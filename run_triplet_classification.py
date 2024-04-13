"""
This code is based on the file in PURE repo: https://github.com/princeton-nlp/PURE/blob/main/run_relation.py
and SpanBERT repo: https://github.com/facebookresearch/SpanBERT/blob/master/code/run_tacred.py
"""

import argparse
import logging
import os
import random
import time
import json
from tqdm import tqdm
import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from torch.nn import CrossEntropyLoss

from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from relation.models_copied import BertForRelation
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from trigger.utils import generate_trigger_data, decode_sample_id
from shared.const import task_rel_labels, task_ner_labels
from shared.utils import set_seed, make_output_dir

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
    parser.add_argument('--task', type=str, default=None, required=True,
                        help=f"Run one of the task in {list(task_ner_labels.keys())}")
    parser.add_argument('--pipeline_task', type=str, default=None, required=True,
                        help=f"Choose what kind of tasks to run: NER, Triplet, and RE")
    parser.add_argument("--do_train", action='store_true', 
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', 
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--do_predict_dev', action='store_true', 
                        help="Whether to run prediction on dev set")
    parser.add_argument('--do_predict_test', action='store_true', 
                        help="Whether to run prediction on test set")
    parser.add_argument("--eval_with_gold", action="store_true", 
                        help="Whether to evaluate the relation model with gold entities provided.")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--binary_classification", action='store_true',
                        help="Whether to run Triplet entailment w/Typed trigger \
                            or Multi-class Triplet classification w/Untyped trigger")
    
    # directory and file arguments
    parser.add_argument('--output_dir', type=str, default=None, required=True,
                        help="Output directory of the experiment outputs")
    parser.add_argument("--entity_output_dir", type=str, default=None,
                        help="Needs to define if you want to use predicted entities \
                            from different folder")
    parser.add_argument("--triplet_output_dir", type=str, default=None,
                        help="Needs to define if you want to use already fine-tuned triplet models")
    parser.add_argument("--entity_output_test_dir", type=str, default=None,
                        help="Needs to define if you want to use predicted entities of test set \
                            from different folder")
    parser.add_argument("--triplet_output_test_dir", type=str, default=None,
                        help="Needs to define if you want to use predicted entities of test set \
                            from different folder")
    parser.add_argument("--entity_predictions_dev", type=str, default="ent_pred_dev.json", 
                        help="The entity prediction file of the dev set")
    parser.add_argument("--entity_predictions_test", type=str, default="ent_pred_test.json", 
                        help="The entity prediction file of the test set")    
    parser.add_argument('--dev_pred_filename', type=str, default="trg_pred_dev.json", 
                        help="the prediction filename for the dev set")
    parser.add_argument('--test_pred_filename', type=str, default="trg_pred_test.json", 
                        help="the prediction filename for the test set")
    
    # data-specific arguments:
    parser.add_argument("--train_file", default=None, type=str, 
                        help="The path of the training data.")
    parser.add_argument("--dev_file", default=None, type=str, 
                        help="The path of the dev data.")
    parser.add_argument("--test_file", default=None, type=str, 
                        help="The path of the test data.")
    parser.add_argument("--train_mode", type=str, default='random_sorted', 
                        choices=['random', 'sorted', 'random_sorted'])

    # training-specific arguments:
    parser.add_argument("--eval_per_epoch", default=1.0, type=float,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument('--context_window', type=int, default=100)
    parser.add_argument("--max_seq_length", default=300, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_metric", default="f1", type=str)
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_epoch", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--max_patience', type=int, default=3)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10\% of training.")
    parser.add_argument("--bertadam", action="store_true", 
                        help="If bertadam, then set correct_bias = False")
    parser.add_argument('--print_loss_step', type=int, default=10, 
                        help="how often logging the loss value during training")
    parser.add_argument("--sampling_method", default="trigger_position", type=str,
                        choices=['random', 'trigger_position'])
    parser.add_argument("--sampling_proportion", default=0.0, type=float)
    
    # model arguments:
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--negative_label", default="no_relation", type=str)
    parser.add_argument("--do_lower_case", default=True, 
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--add_new_tokens', default=True, 
                        help="Whether to add new tokens as marker tokens instead of using [unusedX] tokens.")
    
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
        '<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>', '<TRG_START>', '<TRG_END>'
    ]
    for label in ner_labels:
        new_tokens.append('<SUBJ_START=%s>'%label)
        new_tokens.append('<SUBJ_END=%s>'%label)
        new_tokens.append('<OBJ_START=%s>'%label)
        new_tokens.append('<OBJ_END=%s>'%label)
        new_tokens.append('<TRG_START=%s>'%label)
        new_tokens.append('<TRG_END=%s>'%label)
    # for label in ner_labels:
    #     new_tokens.append('<SUBJ=%s>'%label)
    #     new_tokens.append('<OBJ=%s>'%label)
    #     new_tokens.append('<TRG=%s>'%label)
    tokenizer.add_tokens(new_tokens)
    logger.info('# vocab after adding markers: %d'%len(tokenizer))

def trigger_pos_based_sampling(samples, gold_samples, num_samples_to_keep):
    trg_left_cnt = 0
    trg_right_cnt = 0
    trg_mid_cnt = 0
    for sample in gold_samples:
        if sample["trg_start"] < min(sample["subj_start"], sample["obj_start"]):
            trg_left_cnt += 1
        elif sample["trg_start"] > max(sample["subj_start"], sample["obj_start"]):
            trg_right_cnt += 1
        else:
            trg_mid_cnt += 1

    left_ratio = round(trg_left_cnt/len(gold_samples), 2)
    mid_ratio = round(trg_mid_cnt/len(gold_samples), 2)
    right_ratio = round(trg_right_cnt/len(gold_samples), 2)

    logger.info(f"Left:{trg_left_cnt}({left_ratio*100}%), Mid:{trg_mid_cnt}({mid_ratio*100}%), Right:{trg_right_cnt}({right_ratio*100}%)")

    samples_left = []
    samples_mid = []
    samples_right = []
    for sample in samples:
        if sample["trg_start"] < min(sample["subj_start"], sample["obj_start"]):
            samples_left.append(sample)
        elif sample["trg_start"] > max(sample["subj_start"], sample["obj_start"]):
            samples_right.append(sample)
        else:
            samples_mid.append(sample)
    sampled = random.sample(samples_left, min(int(num_samples_to_keep*left_ratio), len(samples_left)))
    sampled.extend(random.sample(samples_mid, min(int(num_samples_to_keep*mid_ratio), len(samples_mid))))
    sampled.extend(random.sample(samples_right, min(int(num_samples_to_keep*right_ratio), len(samples_right))))
    return sampled

def undersampling(samples, ratio, method):
    class_0_samples = [sample for sample in samples if sample["relation"] == "no_relation"]
    class_1_samples = [sample for sample in samples if sample["relation"] != "no_relation"]

    logger.info(f"Length of No-Relation: {len(class_0_samples)}")
    logger.info(f"Length of Valid Classes: {len(class_1_samples)}")

    # Do undersampling when null cases are more than five times of not-null cases
    if len(class_0_samples) // len(class_1_samples) >= 5:        
        num_samples_to_keep = min(len(class_0_samples), int(len(class_1_samples) / ratio))
        if method == "trigger_position":
            logger.info(f"## Now doing undersampling with {method.upper()} ##")
            class_0_samples_subset = trigger_pos_based_sampling(class_0_samples, class_1_samples, num_samples_to_keep=num_samples_to_keep)
            undersampled_dataset = class_0_samples_subset + class_1_samples
        else:
            logger.info(f"## Now doing RANDOM Undersamping ##")
            class_0_samples_subset = random.sample(class_0_samples, num_samples_to_keep)
            undersampled_dataset = class_0_samples_subset + class_1_samples

        logger.info(f"Length of No-Relation after sampling: {len(class_0_samples_subset)}")

        random.shuffle(undersampled_dataset)
        return undersampled_dataset
    else:
        logger.info("No need for undersampling...")
        return samples

def convert_examples_to_features(examples, label2id, tokenizer, special_tokens, args, unused_tokens=True):
    """
    Loads a data file into a list of `InputBatch`s.
    unused_tokens: whether use [unused1] [unused2] as special tokens
    """

    def get_special_token(w):
        if w not in special_tokens:
            if unused_tokens:
                special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
            else:
                special_tokens[w] = ('<' + w + '>')
        return special_tokens[w]

    num_tokens = 0
    max_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = [CLS]

        TRIGGER_START_NER = get_special_token("TRG_START=%s"%example['trg_type'])
        TRIGGER_END_NER = get_special_token("TRG_END=%s"%example['trg_type'])
        SUBJECT_START_NER = get_special_token("SUBJ_START=%s"%example['subj_type'])
        SUBJECT_END_NER = get_special_token("SUBJ_END=%s"%example['subj_type'])
        OBJECT_START_NER = get_special_token("OBJ_START=%s"%example['obj_type'])
        OBJECT_END_NER = get_special_token("OBJ_END=%s"%example['obj_type'])

        for i, token in enumerate(example['token']):
            if i == example['subj_start']:
                sub_idx = len(tokens)
                tokens.append(SUBJECT_START_NER)
            if i == example['obj_start']:
                obj_idx = len(tokens)
                tokens.append(OBJECT_START_NER)
            if i == example['trg_start']:
                trg_idx = len(tokens)
                tokens.append(TRIGGER_START_NER)
            # for sub_token in tokenizer.tokenize(token):
            for sub_token in tokenizer.tokenize(token.lower()):
                tokens.append(sub_token)
            if i == example['subj_end']:
                tokens.append(SUBJECT_END_NER)
            if i == example['obj_end']:
                tokens.append(OBJECT_END_NER)
            if i == example['trg_end']:
                tokens.append(TRIGGER_END_NER)
        tokens.append(SEP)

        num_tokens += len(tokens)
        max_tokens = max(max_tokens, len(tokens))

        if len(tokens) > args.max_seq_length:
            tokens = tokens[:args.max_seq_length]
            if sub_idx >= args.max_seq_length:
                sub_idx = 0
            if obj_idx >= args.max_seq_length:
                obj_idx = 0
            if trg_idx >= args.max_seq_length:
                trg_idx = 0
        else:
            num_fit_examples += 1

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (args.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id = label2id[example['relation']]
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length

        if num_shown_examples < 20:
            if (ex_index < 5) or (label_id > 0):
                num_shown_examples += 1
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example['id']))
                logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %d)" % (example['relation'], label_id))
                logger.info("sub_idx, obj_idx, trg_idx: %d, %d, %d" % (sub_idx, obj_idx, trg_idx))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                sub_idx=sub_idx,
                obj_idx=obj_idx,
                trg_idx=trg_idx
            )
        )
    logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    logger.info("Max #tokens: %d"%max_tokens)
    logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                num_fit_examples * 100.0 / len(examples), args.max_seq_length))
    return features

def save_div(a, b):
    if b != 0:
        return a / b 
    else:
        return 0.0

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_f1(preds, labels, e2e_ngold, label_cnt_dict, id2label):

    assert e2e_ngold == sum(label_cnt_dict.values())

    result_by_class = {}
    for idx, label in id2label.items():
        if idx != 0:
            result_by_class[label] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 
                                      'gold': 0.0, 'pred': 0.0, 'correct': 0.0}
            result_by_class[label]["gold"] = label_cnt_dict[label]

    n_pred = n_correct = 0
        
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
            result_by_class[id2label[pred]]["pred"] += 1
            if pred == label:
                n_correct += 1
                result_by_class[id2label[pred]]["correct"] += 1
            
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, result_by_class
    else:
        for label in result_by_class:
            counts = result_by_class[label]
            counts["precision"] = save_div(counts["correct"], counts["pred"])
            counts["recall"] = save_div(counts["correct"], counts["gold"])
            counts["f1"] = save_div(2*counts["precision"]*counts["recall"], counts["precision"]+counts["recall"])

        prec = save_div(n_correct, n_pred)
        rec = save_div(n_correct, e2e_ngold)
        f1 = save_div(2*prec*rec, prec+rec)
        logger.info('Prec: %.5f, Rec: %.5f, F1: %.5f'%(prec, rec, f1))

        result = {'precision': prec, 'recall': rec, 'f1': f1, \
        'n_correct': n_correct, 'n_pred': n_pred, 'n_gold': e2e_ngold}        
        return result, result_by_class


def evaluate(model, device, eval_dataloader, eval_label_ids, id2label, label_cnt_dict, e2e_ngold=None, verbose=True):
    
    model.eval()

    num_labels = len(id2label)
    eval_loss = 0
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
            logits = model(input_ids, segment_ids, input_mask, labels=None, sub_idx=sub_idx, obj_idx=obj_idx, trg_idx=trg_idx)
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    logits = preds[0]
    preds = np.argmax(preds[0], axis=1)
    result, result_by_class = compute_f1(
        preds, eval_label_ids.numpy(), e2e_ngold=e2e_ngold, label_cnt_dict=label_cnt_dict, id2label=id2label
        )
    result['accuracy'] = simple_accuracy(preds, eval_label_ids.numpy())
    result['eval_loss'] = eval_loss
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return preds, result, logits, result_by_class


def output_trg_predictions(eval_data, eval_examples, preds, output_file, id2label, args):
    rels = dict()
    triplets = dict()
    for ex, pred in zip(eval_examples, preds):
        doc_sent, sub, obj, trg = decode_sample_id(ex['id'])
        if doc_sent not in rels:
            rels[doc_sent] = []
            triplets[doc_sent] = []
        if pred != 0:
            # predicted_triplet = [sub[0], sub[1], obj[0], obj[1], trg[0], trg[1]]
            # # if len(args.pipeline_task.split("-")) == 3:
            #     rels[doc_sent].append([sub[0], sub[1], obj[0], obj[1], trg[0], trg[1]])
            # else: # Triplet to Relation type, Need to be revised
            if args.binary_classification:
                # Entailed means that it follows the trigger label of example
                predicted_relation = [sub[0], sub[1], obj[0], obj[1], ex['trg_type'], ""]
            else:
                predicted_relation = [sub[0], sub[1], obj[0], obj[1], id2label[pred], ""]
            # Some relations might be duplicated
            # because different trigger could have same relation on same arguments
            if predicted_relation not in rels[doc_sent]:
                rels[doc_sent].append(predicted_relation)
                predicted_triplet = [sub[0], sub[1], obj[0], obj[1], trg[0], trg[1]]
                triplets[doc_sent].append(predicted_triplet)

    # Store triplet prediction
    # Consider using gold standards for NER
    js = eval_data.js
    for doc in js:
        # if args.eval_with_gold:
        #     doc['predicted_triplets'] = doc['triplets']
        # else:
        doc['predicted_triplets'] = []
        for sid in range(len(doc['sentences'])):
            k = '%s@%d'%(doc['doc_key'], sid)
            doc['predicted_triplets'].append(triplets.get(k, []))
        if args.eval_with_gold:
            doc['predicted_ner'] = doc['ner']
            doc['predicted_triggers'] = doc['triggers']

    # If we can get rel type in this task
    if len(args.pipeline_task.split("-")) != 3:
        for doc in js:
            doc['predicted_relations'] = []
            for sid in range(len(doc['sentences'])):
                k = '%s@%d'%(doc['doc_key'], sid)
                doc['predicted_relations'].append(rels.get(k, []))

    logger.info('Output predictions to %s.. \n'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc) for doc in js))


def save_trained_model(model, tokenizer, step, args):
    """
    Save the model to the output directory
    """
    os.makedirs(args.triplet_output_dir, exist_ok=True)
    logger.info('Saving model to %s with Step %d ...'%(args.triplet_output_dir, step))

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.triplet_output_dir)
    tokenizer.save_vocabulary(args.triplet_output_dir)


def main() -> None:
    args = get_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    # Generate specific version of output folder
    if args.triplet_output_dir is None:
        args.triplet_output_dir = make_output_dir(
            args.output_dir, task='triplet', pipeline_task=args.pipeline_task
        )
        os.makedirs(args.triplet_output_dir, exist_ok=True)

    # Specify the entity output folder
    if args.entity_output_dir is None:
        path_components = args.triplet_output_dir.split(os.path.sep)
        path_components[-1] = "entity"
        args.entity_output_dir = os.path.sep.join(path_components)

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.triplet_output_dir, f"train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.triplet_output_dir, f"eval.log"), 'w'))

    # train set
    if args.do_train:
        train_dataset, train_examples, *_ = generate_trigger_data(
            args.train_file, ref_data=args.train_file, use_gold=True, context_window=args.context_window, binary_classification=args.binary_classification
            )
    # dev set
    if args.do_eval or args.do_predict_dev or (args.do_train and args.do_predict_test):
        eval_dataset, eval_examples, eval_ntrg, eval_label_dict = generate_trigger_data(
            os.path.join(args.entity_output_dir, args.entity_predictions_dev), ref_data=args.dev_file, use_gold=args.eval_with_gold, context_window=args.context_window, binary_classification=args.binary_classification
            )
        # incorporate train set with dev set
        if args.do_predict_test:
            logger.info("## Now moving Dev data to Train data... ##")
            train_examples.extend(eval_examples)
            logger.info(f"## Length of Train data: {len(train_examples)} ##")
        if args.sampling_proportion:
            train_examples = undersampling(
                train_examples, ratio=args.sampling_proportion, method=args.sampling_method
            )
    if not args.do_train and not (args.do_predict_dev or args.do_predict_test):
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    logger.info(f"args: {json.dumps(args.__dict__, indent=2, sort_keys=True)}")
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))

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
    if args.add_new_tokens:
        add_marker_tokens(tokenizer, task_ner_labels[args.task])

    if os.path.exists(os.path.join(args.triplet_output_dir, 'special_tokens.json')):
        with open(os.path.join(args.triplet_output_dir, 'special_tokens.json'), 'r') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {}

    if args.do_eval or args.do_predict_dev:
        eval_features = convert_examples_to_features(
            eval_examples, label2id, tokenizer, special_tokens, args, unused_tokens=not(args.add_new_tokens)
        )
        logger.info("***** Dev *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in eval_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in eval_features], dtype=torch.long)
        all_trg_idx = torch.tensor([f.trg_idx for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx, all_trg_idx)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        eval_label_ids = all_label_ids

    with open(os.path.join(args.triplet_output_dir, 'special_tokens.json'), 'w') as f:
        json.dump(special_tokens, f)

    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label2id, tokenizer, special_tokens, args, unused_tokens=not(args.add_new_tokens))
        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in train_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in train_features], dtype=torch.long)
        all_trg_idx = torch.tensor([f.trg_idx for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx, all_trg_idx)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = len(train_dataloader) * args.num_epoch

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_result = None
        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        print("eval step >>> ", eval_step)
        
        lr = args.learning_rate
        model = RelationModel.from_pretrained(
            args.model, 
            cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), 
            num_rel_labels=num_labels,
            use_trigger=True
            )
        if hasattr(model, 'bert'):
            model.bert.resize_token_embeddings(len(tokenizer))
        elif hasattr(model, 'albert'):
            model.albert.resize_token_embeddings(len(tokenizer))
        else:
            raise TypeError("Unknown model class")

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=not(args.bertadam))
        scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_optimization_steps * args.warmup_proportion), num_train_optimization_steps)

        start_time = time.time()
        global_step = 0
        tr_loss = 0
        nb_tr_examples = 0
        nb_tr_steps = 0
        eval_step = len(train_batches) // args.eval_per_epoch

        max_patience, current_patience = args.max_patience, 0
        if_exit = False

        for epoch in range(1, args.num_epoch+1):

            if if_exit:
                logger.info("=== Do EARLY STOPPING ===")
                break

            if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                random.shuffle(train_batches)

            progress = tqdm.tqdm(total=len(train_batches), ncols=150, desc="Epoch: " + str(epoch))
        
            for step, batch in enumerate(train_batches):

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx, trg_idx = batch

                loss = model(input_ids, segment_ids, input_mask, label_ids, sub_idx, obj_idx, trg_idx)
                if n_gpu > 1:
                    loss = loss.mean()

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.print_loss_step == 0:
                    logger.info('Epoch=%d, step=%d, loss=%.5f'%(epoch, step, tr_loss / nb_tr_examples))
                    tr_loss = 0
                    nb_tr_examples = 0

                if args.do_eval and global_step % eval_step == 0:
                    preds, result, logits, result_by_class = evaluate(
                        model, device, eval_dataloader, eval_label_ids, id2label=id2label, label_cnt_dict=eval_label_dict, e2e_ngold=eval_ntrg
                    )
                    model.train()
                    result['global_step'] = global_step
                    result['epoch'] = epoch
                    result['learning_rate'] = lr
                    result['batch_size'] = args.train_batch_size

                    if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                        best_result = result
                        logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                    (args.eval_metric, str(lr), epoch, result[args.eval_metric] * 100.0))
                        save_trained_model(model, tokenizer, global_step, args)
                        current_patience = 0
                    else:
                        if result[args.eval_metric] != 0:
                            current_patience += 1
                            if current_patience >= max_patience:
                                if_exit = True
                progress.update(1)
            progress.close()

    if not args.do_eval and args.do_train:
        logger.info('## Without Validation Set: Saving model to %s... ##'%(args.triplet_output_dir))
        save_trained_model(model, tokenizer, global_step, args)

    logger.info(special_tokens)

    if args.do_predict_dev:
        model = RelationModel.from_pretrained(
            args.triplet_output_dir, num_rel_labels=num_labels, use_trigger=True
        )
        model.to(device)
        # dev dataloader has been already made
        preds, result, logits, result_by_class = evaluate(
                            model, device, eval_dataloader, eval_label_ids, id2label=id2label, label_cnt_dict=eval_label_dict, e2e_ngold=eval_ntrg
                        )
        with open(os.path.join(args.triplet_output_dir, "dev_result_by_class.json"), 'w', encoding='utf-8') as f_out:
            f_out.write(json.dumps(result_by_class, indent=4))

        logger.info('*** Final Dev Set Results ***')
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        trg_pred_file_path = os.path.join(args.triplet_output_dir, args.dev_pred_filename)
        output_trg_predictions(
            eval_dataset, eval_examples, preds, trg_pred_file_path, id2label, args
        )

    if args.do_predict_test:
        test_dataset, test_examples, test_ntrg, test_label_dict = generate_trigger_data(
            os.path.join(args.entity_output_test_dir, args.entity_predictions_test), ref_data=args.test_file, use_gold=args.eval_with_gold, context_window=args.context_window, binary_classification=args.binary_classification
            )
        test_features = convert_examples_to_features(
            test_examples, label2id, tokenizer, special_tokens, args, unused_tokens=not(args.add_new_tokens)
        )
        logger.info("***** Test *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in test_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in test_features], dtype=torch.long)
        all_trg_idx = torch.tensor([f.trg_idx for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx, all_trg_idx)
        test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size)
        test_label_ids = all_label_ids

        # Load the fine-tuned model (TRAIN + DEV)
        model = RelationModel.from_pretrained(
            "gbhong/BiomedBERT-fulltext_finetuned_DiMB-RE", num_rel_labels=num_labels, use_trigger=True
        )
        model.to(device)

        preds, result, logits, result_by_class = evaluate(
            model, device, test_dataloader, test_label_ids, id2label=id2label, label_cnt_dict=test_label_dict, e2e_ngold=test_ntrg, verbose=False
        )
        with open(os.path.join(args.triplet_output_dir, "test_result_by_class.json"), 'w', encoding='utf-8') as f_out:
            f_out.write(json.dumps(result_by_class, indent=4))

        logger.info('*** Final Test Set Results ***')
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        trg_test_file_path = os.path.join(args.triplet_output_dir, args.test_pred_filename)
        output_trg_predictions(
            test_dataset, test_examples, preds, trg_test_file_path, id2label, args
        )


if __name__ == "__main__":
    main()