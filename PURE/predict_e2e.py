'''
This code predicts the entity mentions and relation types among them
from given input sentence.
'''

import json
import argparse
import os
import logging
from datetime import datetime
import pytz

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from shared.data_structures import Dataset
from shared.const import task_ner_labels, task_rel_labels, task_certainty_labels, get_labelmap
from shared.utils import generate_analysis_csv, decode_sample_id
from entity.utils import convert_dataset_to_samples, batchify, NpEncoder
from entity.models import EntityModel
from relation.models import BertForRelation
from trigger.utils import generate_trigger_data, convert_examples_to_triplet_features
from certainty.models import BertForFactuality
from certainty.utils import generate_certainty_data, convert_examples_to_factuality_features

CLS = "[CLS]"
SEP = "[SEP]"

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    )
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    # general arguments: Task, Directories, Train/Eval, and so on.
    parser.add_argument(
        '--task', 
        type=str, 
        default=None, 
        required=True,
        help=f"Run one of the task in {list(task_ner_labels.keys())}"
    )
    parser.add_argument(
        "--no_cuda", 
        action='store_true',
        help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--extract_trigger", 
        action='store_true',
        help="Whether to extract trigger in NER task"
    )
    parser.add_argument(
        "--untyped_trigger", 
        action='store_true',
        help="Whether to use untyped TRIGGER in NER task"        
    )
    
    # directory and file arguments
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="./output/", 
        # required=True,
        help="Output directory of the experiment outputs"
    )
    parser.add_argument(
        '--ner_pred_filename', 
        type=str, 
        default="ent_pred_test.json", 
        help="Prediction filename for the test set"
    )
    parser.add_argument(
        '--rel_pred_filename', 
        type=str, 
        default="rel_pred_test.json", 
        help="Prediction filename for the test set"
    )
    parser.add_argument(
        '--e2e_pred_filename', 
        type=str, 
        default="e2e_pred_test.json", 
        help="Prediction filename for the test set"
    )    
    parser.add_argument(
        '--e2e_csv_filename', 
        type=str, 
        default="e2e_pred_test.csv", 
        help="Prediction CSV filename for the test set"
    )
    parser.add_argument(
        "--binary_classification", 
        action='store_true',
        help="Whether to run Triplet entailment w/Typed trigger \
            or Multi-class Triplet classification w/Untyped trigger"
    )

    # data-specific arguments:
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default="/data", 
        required=True, 
        help="Path to the preprocessed dataset"
    )
    parser.add_argument(
        '--remove_nested', 
        action='store_true',
        help="Whether to remove nested entities"
    )
    
    # training-specific arguments:
    parser.add_argument(
        '--context_window', 
        type=int, 
        default=100, 
        help="Context window size W for the entity model"
    )
    parser.add_argument(
        '--max_seq_length', 
        type=int, 
        default=300,
        help="Maximum length of tokenized input sequence"
    )
    parser.add_argument(
        '--eval_batch_size', 
        type=int, 
        default=64, 
        help="Batch size during inference"
    )
    parser.add_argument(
        '--print_loss_step', 
        type=int, 
        default=30, 
        help="How often logging the loss value during training"        
    )
    
    # model arguments:
    parser.add_argument(
        '--model', 
        type=str, 
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="Base model name (a huggingface model)"
    )
    parser.add_argument(
        '--entity_model_dir', 
        type=str, 
        default='/jet/home/ghong1/ocean_cis230030p/ghong1/PN/output/best_models/entity',
        help="Output directory of the entity prediction outputs"
    )
    parser.add_argument(
        '--relation_model_dir', 
        type=str, 
        default='/jet/home/ghong1/ocean_cis230030p/ghong1/PN/output/best_models/triplet',
        help="Output directory of the relation prediction outputs"
    )
    parser.add_argument(
        '--certainty_model_dir', 
        type=str, 
        default='/jet/home/ghong1/ocean_cis230030p/ghong1/PN/output/best_models/certainty',
        help="Output directory of the certainty prediction outputs"
    )    
    parser.add_argument(
        '--max_span_length_entity', 
        type=int, 
        default=8, 
        help="Entity Spans w/ length up to max_span_length are considered as candidates"
    )
    parser.add_argument(
        '--max_span_length_trigger', 
        type=int, 
        default=4, 
        help="Trigger Spans w/ length up to max_span_length are considered as candidates"
    )
    parser.add_argument(
        '--dual_classifier', 
        type=bool,
        default=True,
        help="Whether to use different classifiers for Entity and Trigger. \
            If not extracting Triggers, then this also should be off."
    )
    parser.add_argument(
        "--negative_label", default="no_relation", type=str
    )
    parser.add_argument(
        "--negative_label_certainty", default="no_certainty", type=str
    )    
    
    args = parser.parse_args()
    return args


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id, sub_idx, obj_idx, trg_idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sub_idx = sub_idx
        self.obj_idx = obj_idx
        self.trg_idx = trg_idx


def add_marker_tokens(tokenizer, ner_labels):
    new_tokens = [
        '<SUBJ_START>', '<SUBJ_END>', 
        '<OBJ_START>', '<OBJ_END>', 
        '<TRG_START>', '<TRG_END>'
    ]
    for label in ner_labels:
        new_tokens.append('<SUBJ_START=%s>'%label)
        new_tokens.append('<SUBJ_END=%s>'%label)
        new_tokens.append('<OBJ_START=%s>'%label)
        new_tokens.append('<OBJ_END=%s>'%label)
        new_tokens.append('<TRG_START=%s>'%label)
        new_tokens.append('<TRG_END=%s>'%label)
    tokenizer.add_tokens(new_tokens)
    logger.info('# vocab after adding markers: %d'%len(tokenizer))


def output_ner_predictions(model, batches, dataset, ner_id2label, output_file):
    """
    Save the prediction as a json file
    """
    ner_result = {}
    trg_result = {}
    tot_pred_ett = 0
    tot_pred_trg = 0

    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            off = sample['sent_start_in_doc'] - sample['sent_start']
            k = sample['doc_key'] + '-' + str(sample['sentence_ix'])
            ner_result[k] = []
            trg_result[k] = []
            for span, pred in zip(sample['spans'], preds):
                # span_id = '%s::%d::(%d,%d)'%(sample['doc_key'], sample['sentence_ix'], span[0]+off, span[1]+off)
                if pred == 0:
                    continue
                # ner_result[k].append([span[0]+off, span[1]+off, ner_id2label[pred]])
                if not ner_id2label[pred].isupper():
                    ner_result[k].append(
                        [span[0]+off, span[1]+off, ner_id2label[pred]]
                    )
                else:
                    trg_result[k].append(
                        [span[0]+off, span[1]+off, ner_id2label[pred]]
                    )
            tot_pred_ett += len(ner_result[k])
            tot_pred_trg += len(trg_result[k])

    logger.info('Total pred entities: %d'%tot_pred_ett)
    logger.info('Total pred triggers: %d'%tot_pred_trg)

    js = dataset.js
    for i, doc in enumerate(js):
        doc["predicted_ner"] = []
        doc["predicted_triggers"] = []
        doc["predicted_triplets"] = []
        doc["predicted_relations"] = []
        for j in range(len(doc["sentences"])):
            k = doc['doc_key'] + '-' + str(j)
            if k in ner_result:
                doc["predicted_ner"].append(ner_result[k])
            else:
                logger.info('%s not in NER results!'%k)
                doc["predicted_ner"].append([])
            if k in trg_result:
                doc["predicted_triggers"].append(trg_result[k])
            else:
                logger.info('%s not in TRG results!'%k)
                doc["predicted_triggers"].append([])
            
            doc["predicted_triplets"].append([])
            doc["predicted_relations"].append([])

        js[i] = doc

    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(
            json.dumps(doc, cls=NpEncoder) for doc in js
        ))


def output_trg_predictions(
    eval_data, eval_examples, preds, output_file, id2label, args
):

    rels = dict()
    triplets = dict()

    for ex, pred in zip(eval_examples, preds):
        doc_sent, sub, obj, trg = decode_sample_id(ex['id'])
        if doc_sent not in rels:
            rels[doc_sent] = []
            triplets[doc_sent] = []
        if pred != 0:
            if args.binary_classification:
                # 'Entailed' means that it follows the trigger label of example
                predicted_relation = [
                    sub[0], sub[1], obj[0], obj[1], ex['trg_type'], ""
                ]
            else:
                predicted_relation = [
                    sub[0], sub[1], obj[0], obj[1], id2label[pred], ""
                ]
            # Some relations might be duplicated
            # because different trigger could have same relation on same arguments
            if predicted_relation not in rels[doc_sent]:
                rels[doc_sent].append(predicted_relation)
                predicted_triplet = [
                    sub[0], sub[1], obj[0], obj[1], trg[0], trg[1]
                ]
                triplets[doc_sent].append(predicted_triplet)

    # Store triplet prediction
    # (Consider using gold standards for NER)
    js = eval_data.js
    for doc in js:
        doc['predicted_triplets'] = []
        doc['predicted_relations'] = []
        for sid in range(len(doc['sentences'])):
            k = '%s@%d'%(doc['doc_key'], sid)
            doc['predicted_triplets'].append(triplets.get(k, []))
            doc['predicted_relations'].append(rels.get(k, []))

        # if args.eval_with_gold:
        #     doc['predicted_ner'] = doc['ner']
        #     doc['predicted_triggers'] = doc['triggers']

    logger.info('Output predictions to %s.. \n'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc) for doc in js))


def print_pred_json(eval_data, eval_examples, preds, id2label, output_file, args):
    rels = dict()
    for ex, pred in zip(eval_examples, preds):
        doc_sent, sub, obj = decode_sample_id(ex['id'], use_trigger=False)
        rel_type = ex['rel_type']
        if doc_sent not in rels:
            rels[doc_sent] = []
        rels[doc_sent].append(
            [sub[0], sub[1], obj[0], obj[1], rel_type, id2label[pred]]
        )

    js = eval_data.js
    for doc in js:
        doc['predicted_relations'] = []
        for sid in range(len(doc['sentences'])):
            k = '%s@%d'%(doc['doc_key'], sid)
            doc['predicted_relations'].append(rels.get(k, []))

        # if args.eval_with_gold:
        #     doc['predicted_ner'] = doc['ner']
        #     doc['predicted_triggers'] = doc['triggers']
        #     doc['predicted_triplets'] = doc['triplets']
    
    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc) for doc in js))        


def predict(
        model, 
        device, 
        eval_dataloader, 
        verbose=True
):  
    '''
    Function to infer relations/factuality levels
    for examples without ground-truths
    '''
    model.eval()
    nb_eval_steps = 0

    preds = []
    for input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx, trg_idx in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        sub_idx = sub_idx.to(device)
        obj_idx = obj_idx.to(device)
        trg_idx = trg_idx.to(device)

        with torch.no_grad():
            logits = model(
                input_ids, 
                segment_ids, 
                input_mask, 
                labels=None, 
                sub_idx=sub_idx, 
                obj_idx=obj_idx, 
                trg_idx=trg_idx
            )

        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0
            )

    logits = preds[0]
    preds = np.argmax(preds[0], axis=1)

    return preds, logits



def main() -> None:
    args = get_args()
    args.test_data = os.path.join(args.data_dir, 'test.json')

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    cdt_time = datetime.now(pytz.timezone('America/Chicago'))
    timestamp = cdt_time.strftime("%Y-%m-%d_%H-%M-%S")

    args.output_dir = os.path.join(
        args.output_dir, timestamp
    )
    os.mkdir(args.output_dir)

    logger.addHandler(
        logging.FileHandler(os.path.join(
            args.output_dir, f"predict.log"
        ), 'w')
    )

    logger.info(f"args: {json.dumps(args.__dict__, indent=2, sort_keys=True)}")
    logger.info("device: {}, n_gpu: {}".format(device, args.n_gpu))


    ### Step 1. Predict entities and triggers ###

    ner_label2id, ner_id2label, num_entity_labels, num_trigger_labels = get_labelmap(
        [label for label in task_ner_labels[args.task]]
    )
    logger.info(f"NER labels: {ner_label2id}")

    num_entity_labels += 1
    num_trigger_labels += 1

    args.bert_model_dir = args.entity_model_dir
    model_ner = EntityModel(
        args, 
        num_entity_labels=num_entity_labels, 
        num_trigger_labels=num_trigger_labels, 
        evaluation=True
    )

    test_data = Dataset(args.test_data)
    ner_pred_file = os.path.join(
        args.output_dir, args.ner_pred_filename
    )
    test_samples, *_ = convert_dataset_to_samples(
        args, test_data, ner_label2id=ner_label2id
    )
    test_batches = batchify(
        test_samples, args.eval_batch_size, args.model
    )
    output_ner_predictions(
        model_ner, 
        test_batches, 
        test_data, 
        ner_id2label, 
        output_file=ner_pred_file
    )

    logger.info("## FINISHED PREDICTING ENTITIES AND TRIGGERS ##")


    ### Step 2. Predict relations based on the predicted entities and triggers ###

    logger.info("## STARTING RELATION PREDICTION ##")

    label_list = [args.negative_label]
    if args.binary_classification:
        label_list.append("Entailment")
    else:
        label_list.extend(task_rel_labels[args.task])

    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    RelationModel = BertForRelation
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    add_marker_tokens(tokenizer, task_ner_labels[args.task])

    if os.path.exists(os.path.join(args.relation_model_dir, 'special_tokens.json')):
        with open(os.path.join(args.relation_model_dir, 'special_tokens.json'), 'r') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {}

    test_dataset, test_examples, *_ = generate_trigger_data(
        os.path.join(args.output_dir, args.ner_pred_filename), 
        ref_data=args.test_data, 
        use_gold=False, 
        context_window=args.context_window, 
        binary_classification=True
    )
    
    test_features = convert_examples_to_triplet_features(
        test_examples, 
        label2id, 
        tokenizer, 
        special_tokens, 
        args, 
        unused_tokens=False
    )

    logger.info("***** Predict Test set *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    all_sub_idx = torch.tensor([f.sub_idx for f in test_features], dtype=torch.long)
    all_obj_idx = torch.tensor([f.obj_idx for f in test_features], dtype=torch.long)
    all_trg_idx = torch.tensor([f.trg_idx for f in test_features], dtype=torch.long)

    test_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, 
        all_label_ids, all_sub_idx, all_obj_idx, all_trg_idx
    )
    test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size)
    # test_label_ids = all_label_ids

    # Load the fine-tuned model
    model = RelationModel.from_pretrained(
        args.relation_model_dir, 
        num_rel_labels=num_labels, 
        use_trigger=True
    )
    model.to(device)

    preds, logits = predict(
        model, 
        device, 
        test_dataloader, 
        verbose=False
    )

    trg_test_file_path = os.path.join(args.output_dir, args.rel_pred_filename)

    output_trg_predictions(
        test_dataset, 
        test_examples, 
        preds, 
        trg_test_file_path, 
        id2label, 
        args
    )

    logger.info("## FINISHED RELATION PREDICTION ##")


    ### Step 3. Predict factuality level for predicted relations ###

    logger.info("## STARTING FACTUALITY PREDICTION ##")

    label_list = [args.negative_label_certainty]
    label_list.extend(task_certainty_labels[args.task])
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    FactualityModel = BertForFactuality
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    add_marker_tokens(tokenizer, task_ner_labels[args.task])

    if os.path.exists(os.path.join(args.certainty_model_dir, 'special_tokens.json')):
        with open(os.path.join(args.certainty_model_dir, 'special_tokens.json'), 'r') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {}

    test_dataset, test_examples, *_ = generate_certainty_data(
        os.path.join(args.output_dir, args.rel_pred_filename), 
        ref_data=args.test_data, 
        training=False, 
        use_gold=False,
        use_trigger=True, 
        context_window=0
    )    
    
    test_features = convert_examples_to_factuality_features(
        examples=test_examples, 
        label2id=label2id, 
        max_seq_length=200, 
        tokenizer=tokenizer, 
        special_tokens=special_tokens, 
        unused_tokens=False
    )

    logger.info("***** Predict Test set *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    all_sub_idx = torch.tensor([f.sub_idx for f in test_features], dtype=torch.long)
    all_obj_idx = torch.tensor([f.obj_idx for f in test_features], dtype=torch.long)
    all_trg_idx = torch.tensor([f.trg_idx for f in test_features], dtype=torch.long)

    test_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, 
        all_label_ids, all_sub_idx, all_obj_idx, all_trg_idx
    )
    test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size)
    # test_label_ids = all_label_ids

    model = FactualityModel.from_pretrained(
        args.certainty_model_dir, 
        num_certainty_labels=num_labels, 
        use_trigger=True
    )   
    model.to(device)

    preds, logits = predict(
        model, 
        device, 
        test_dataloader,
        verbose=False
    )

    e2e_test_file_path = os.path.join(args.output_dir, args.e2e_pred_filename)
    test_csv_path = os.path.join(args.output_dir, args.e2e_csv_filename)

    print_pred_json(
        eval_data=test_dataset, 
        eval_examples=test_examples, 
        preds=preds, 
        id2label=id2label, 
        output_file=e2e_test_file_path, 
        args=args
    )

    logger.info("## FINISHED END-TO-END RELATION PREDICTION ##")


    ### Step 4. Generate csv file for analysis ###
    logger.info(f'## Generate CSV file for Sent-level Analysis from {e2e_test_file_path} ##')
    final_prediction = Dataset(e2e_test_file_path)
    generate_analysis_csv(final_prediction, test_csv_path)


if __name__ == "__main__":
    main()