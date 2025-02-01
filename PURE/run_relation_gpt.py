import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

from sentence_transformers import SentenceTransformer

from icl.models.embeddings import *
from icl.utils.utils import *
from icl.utils.api_call import OpenAIAPI

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    # Directory args
    parser.add_argument("--data_dir", type=str, default="./data/pernut/ner_reduced_v6.1_trg_abs")
    parser.add_argument('--output_dir', type=str, default='./icl/output/',
                        help='Path to save experiment result files')
    parser.add_argument('--ckpt_dir', type=str,
                        help='Designate checkpoint directory if needed.')
    
    parser.add_argument("--prompt_dir", type=str, default="./icl/prompts/")
    parser.add_argument("--prompt_file", type=str, default="desc_guid_1shot_021424",
                        help='File name for prompt text')

    # Model args
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument('--eval_valid_only', action='store_true')
    
    parser.add_argument("--zero_shot", action='store_true')
    parser.add_argument("--nli", action='store_true')
    parser.add_argument("--min_count", type=int, default=3,
                        help='Threshold to exclude low-count entity pair types \
                            from relation structures')
    parser.add_argument("--gpt_model", type=str, default="chatgpt",
                        choices=['chatgpt', 'gpt4', 'gpt4o-mini'])
    parser.add_argument('--use_generated_prompts', action='store_true',
                        help='Load previously generated prompts')
    
    parser.add_argument("--few_shot", action='store_true')
    parser.add_argument("--retrieval_method", type=str, default='knn')
    parser.add_argument("--retrieval_model", type=str, 
                        default="princeton-nlp/sup-simcse-roberta-base")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument('--load_sim', action='store_true',
                        help='Load previously calculated similiarity matrix')
    
    # GPT-related args
    # Please refer to https://platform.openai.com/docs/guides/text-generation/parameter-details
    parser.add_argument('--api_key', type=str, required=True, 
                        help='Input your own API key')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, 
                        default=4096, help='Set max tokens for decoding')
    parser.add_argument('--seed', type=int, 
                        default=42, help='Set max tokens for decoding')

    # Data args
    parser.add_argument('--sampling', type=str, default='full', 
                        choices=['full', 'sequential', 'random'],
                        help='Select types of sampling. \
                            If you sample a susbet of dataset, then choose sequential or random.')
    parser.add_argument('--subset_ratio', type=float, default=1.0,
                        help='Set proportion to make a subset from full dataset')
    parser.add_argument('--null_ratio', type=float, default=0.9,
                        help='Set Null proportion to make a subset from full dataset')
    parser.add_argument('--load_subset', action='store_true',
                        help='Load previously generated subset indices')
    parser.add_argument('--num_debug', type=int, default=0,
                        help='Set the number of samples for debugging')    

    args = parser.parse_args()
    return args


def cosine_sim(source_features, target_features):
    source_features /= np.linalg.norm(np.array(source_features), axis=-1, keepdims=True)
    target_features /= np.linalg.norm(np.array(target_features), axis=-1, keepdims=True)
    similarity = source_features @ target_features.T
    return similarity

def rank_sim(similarity):
    query_len, target_len = similarity.shape
    ranks = np.empty((query_len, target_len))
    top1 = np.zeros(query_len)
    for idx in range(query_len):
        ranked_indices = np.argsort(similarity[idx])[::-1].tolist()
        ranks[idx] = ranked_indices
        top1[idx] = ranked_indices[0]
    return ranks, top1

def print_pred_json(eval_data, eval_examples, preds, output_file, use_gold=False):
    rels = dict()
    for ex, pred in zip(eval_examples, preds):
        pred_rel, pred_modal = pred
        doc_sent, sub, obj = decode_sample_id(ex['id'])
        if doc_sent not in rels:
            rels[doc_sent] = []
        if pred_rel != 'no_relation':
            rels[doc_sent].append([sub[0], sub[1], obj[0], obj[1], pred_rel, pred_modal])

    js = eval_data.js
    for doc in js:
        doc['predicted_relations'] = []
        for sid in range(len(doc['sentences'])):
            k = '%s@%d'%(doc['doc_key'], sid)
            doc['predicted_relations'].append(rels.get(k, []))
    
    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc) for doc in js))

def save_div(a, b):
    if b != 0:
        return a / b 
    else:
        return 0.0

def evaluate(preds, samples, label_list):
    result_by_class = {}
    for label in label_list:
        if label != 'no_relation':
            result_by_class[label] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 
                                      'gold': 0.0, 'pred': 0.0, 'correct': 0.0}

    n_pred = n_gold = n_correct = n_correct_modality = 0
    for pred, sample in zip(preds, samples):

        pred_relation, pred_modality = pred

        logger.info(f"PRED: {pred} | GOLD: {(sample['relation'], sample['factuality'])}")

        if pred_relation != 'no_relation':
            n_pred += 1
            result_by_class[pred_relation]["pred"] += 1
        if sample['relation'] != 'no_relation':
            n_gold += 1
            result_by_class[sample['relation']]['gold'] += 1
            if pred_relation == sample['relation']:
                n_correct += 1
                if pred_modality == sample['factuality']:
                    n_correct_modality += 1
                    result_by_class[pred_relation]["correct"] += 1
            
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, result_by_class
    else:
        for label in result_by_class:
            counts = result_by_class[label]
            counts["precision"] = save_div(counts["correct"], counts["pred"])
            counts["recall"] = save_div(counts["correct"], counts["gold"])
            counts["f1"] = save_div(2*counts["precision"]*counts["recall"], counts["precision"]+counts["recall"])

        prec = save_div(n_correct, n_pred)
        rec = save_div(n_correct, n_gold)
        f1 = save_div(2*prec*rec, prec+rec)
        logger.info('Prec: %.5f, Rec: %.5f, F1: %.5f'%(prec, rec, f1))

        prec_modality = save_div(n_correct_modality, n_pred)
        rec_modality = save_div(n_correct_modality, n_gold)
        f1_modality = save_div(2*prec_modality*rec_modality, prec_modality+rec_modality)
        logger.info('Prec_modal: %.5f, Rec_modal: %.5f, F1_modal: %.5f'%(prec_modality, rec_modality, f1_modality))

        result = {
            'precision': prec, 
            'recall': rec, 
            'f1': f1,
            'precision_modality': prec_modality, 
            'recall_modality': rec_modality, 
            'f1_modality': f1_modality,
            'n_pred': n_pred, 
            'n_gold': n_gold,
            'n_correct': n_correct, 
            'n_correct_modality': n_correct_modality
        }
        return result, result_by_class


def main():
    # Setting Arguments
    args = get_args()
    logger.info(f"args: {json.dumps(args.__dict__, indent=2, sort_keys=True)}")

    args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    args.train_file = os.path.join(args.data_dir, "train.json")
    args.dev_file = os.path.join(args.data_dir, "dev.json")
    args.test_file = os.path.join(args.data_dir, "test.json")

    # args.dev_file = os.path.join(args.data_dir, "dev_sample.json")

    args.k_shot = 0 if args.zero_shot else args.top_k

    if args.do_test:
        args.output_dir = os.path.join(args.output_dir, 'test')
    args.output_dir = os.path.join(
        args.output_dir, 
        f"{args.prompt_file}_{args.gpt_model}_{args.k_shot}_{args.retrieval_model.split('/')[-1]}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    train_data, train_samples = generate_data(args.train_file)
    if not args.do_test:
        dev_data, dev_samples = generate_data(args.dev_file, sent_level=False)
    else:
        dev_data, dev_samples = generate_data(args.dev_file)
    test_data, test_samples = generate_data(args.test_file, sent_level=False)

    args.relation_structure = "rel_structure_abs_train.json" if not args.do_test else "rel_structure_abs_traindev.json"

    # Load predefined relation structures
    with open(os.path.join(args.data_dir, args.relation_structure), 'r') as file:
        structure = json.load(file)
    # Load verbalizers
    with open("./icl/verbalizer.json", 'r') as file:
        verbalizer = json.load(file)
    
    label_list = []  # To store predefined rel labels
    label2statement = {}  # Verbalize labels into statements
    for rel, values in verbalizer.items():
        label_list.append(rel)
        if isinstance(values, dict):
            for factuality, statement in values.items():
                key = '-'.join([rel, factuality])
                label2statement[key] = statement
        else:  # str type
            label2statement[rel] = values
    # To parse statements to map them to original labels        
    statement2label = {v:k for k, v in label2statement.items()}  

    # Filter out pairs from relation structures based on min-count
    filtered_structure = defaultdict(list)
    for rel_type, values in structure.items():
        for pair, cnt in values:
            if cnt >= args.min_count:
                filtered_structure[rel_type].append(pair)

    if args.do_test:
        demo_samples = train_samples + dev_samples
        query_samples = test_samples
    else:
        demo_samples = train_samples
        query_samples = dev_samples

    ranks = None
    if args.retrieval_method == 'knn':

        demo_texts, query_texts = reconstruct_samples(demo_samples, query_samples)

        # Calculate similarity between query and targets
        similarity_dir = os.path.join(args.output_dir, f'similarity_{str(len(query_texts))}.npy')
        if args.load_sim and os.path.exists(similarity_dir):
            similarity = np.load(similarity_dir)
            logger.info(f"Loaded Similiary matrix")
        else:
            if args.retrieval_model.startswith("kamalkraj") or \
                args.retrieval_model.startswith("neuml"):
                model = SentenceTransformer(args.retrieval_model)
                model = model.to(args.device)
                all_query_embeddings, all_target_embeddings = get_embeddings_st(
                    demo_texts, query_texts, model, args.device, args.batch_size
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.retrieval_model)
                model = AutoModel.from_pretrained(args.retrieval_model)
                model = model.to(args.device)

                # targets = tokenizer(demo_texts, padding=True, truncation=True, return_tensors="pt")
                # queries = tokenizer(query_texts, padding=True, truncation=True, return_tensors="pt")
                # targets = targets.to(args.device)
                # queries = queries.to(args.device)

                all_query_embeddings, all_target_embeddings = get_embeddings(
                    demo_texts, query_texts, tokenizer, model, args.device, args.batch_size
                )
                # # all_target_embeddings = all_target_embeddings.detach().cpu().numpy()
                # # all_query_embeddings = all_query_embeddings.detach().cpu().numpy()
            
                similarity = cosine_sim(all_query_embeddings, all_target_embeddings)
                np.save(similarity_dir, similarity)
                logger.info(f"Saved Similiary matrix")

        logger.info(f"Shape of Similarity matrix >>> {similarity.shape}")

        ranks, top1 = rank_sim(similarity)
        logger.info(ranks.shape)
        # logger.info(top1.shape)

    args.prompt_file += '.txt'
    prompt_dir = os.path.join(
        args.prompt_dir, args.prompt_file
    )

    # # To check sim-based demos
    # for i in range(10):
    #     print('QUERY >>', query_texts[i])
    #     for r in ranks[i][:5]:
    #         print(f'DEMO-INDEX: {int(r)}, SIM: {similarity[i][int(r)]}')
    #         print('DEMO >>', demo_texts[int(r)])
    #         print()
    #     print()
    # assert 1==0

    generated_prompts = generate_prompt(
        demo_samples, query_samples, filtered_structure, verbalizer, 
        prompt_dir, nli=args.nli, few_shot=args.few_shot, ranks=ranks
    )

    # Sampling for reduce the number of samples
    # Or to balance the ratio of null and not-null
    added_sents = []
    input_prompts = []
    input_samples = []
    valid_indices = []

    subset_fn = f'subset_indices_{args.subset_ratio}_null_{args.null_ratio}_seed_{args.seed}.txt'
    full_subset_fn = os.path.join('./icl/output/test', subset_fn)

    if args.sampling != 'full':
        if args.load_subset:
            with open(full_subset_fn, 'r') as f:
                subset_indices = [int(line.strip()) for line in f]
                logger.info(len(subset_indices))
                for idx in subset_indices:
                    input_prompts.append(generated_prompts[idx])
                    input_samples.append(query_samples[idx])

        else:
            input_prompts, input_samples, subset_indices = generate_subset(
                generated_prompts, query_samples, args.sampling, 
                args.subset_ratio, args.null_ratio
            )
            with open(full_subset_fn, 'w') as f:
                for index in subset_indices:
                    f.write(f"{index}\n")
    else:
        input_prompts = generated_prompts
        if args.do_test:
            input_samples = test_samples
        else:
            input_samples = dev_samples
        subset_indices = list(range(len(input_samples)))
    
    logger.info(f"Length of input_prompts >>> {len(input_prompts)}")
    logger.info(f"Length of input_samples >>> {len(input_samples)}")

    if args.num_debug:
        input_prompts = input_prompts[:args.num_debug]
        input_samples = input_samples[:args.num_debug]
        subset_indices = subset_indices[:args.num_debug]

    # Save valid indices for evaluation
    valid_indices = [
        idx for idx, sample in enumerate(input_samples) \
            if sample['relation'] != 'no_relation'
    ]
    logger.info(f"Length of not-null relations >>> {len(valid_indices)}")

    # Load GPT API-related settings, then call GPT API
    api = OpenAIAPI(args)

    # If you want to use responses already generated,
    # you just need to slice inputs from the last index of loaded responses
    responses = api.generate_response(
        input_prompts, args.output_dir, ckpt_dir=args.ckpt_dir
    )
    # Shuffled cases
    if len(subset_indices) != len(responses):
        responses = [responses[idx] for idx in subset_indices]
    predictions = extract_predictions(
        responses, statement2label, label_list, two_step=True
    )

    if args.eval_valid_only:
        predictions_ = [predictions[i] for i in valid_indices]
        predictions = predictions_[:]
        input_samples_ = [input_samples[i] for i in valid_indices]
        input_samples = input_samples_[:]
        del predictions_, input_samples_

    # Just for check
    logger.info(f"## CHECK FROM FIRST ROW ##")
    i = 0
    for sample, prompt, response, pred in zip(
        input_samples, input_prompts, responses, predictions
    ):
        if i >= 5:
            break
        logger.info(f"Sample >>> {sample}")
        logger.info(f"Prompt >>> {prompt}")
        logger.info(f"Response >>> {response}\n")
        logger.info(f"Pred >>> {pred}\n")
        i += 1
    # Check in reverse
    logger.info(f"## CHECK FROM LAST ROW ##")
    for sample, prompt, response, pred in zip(
        input_samples[::-1], input_prompts[::-1], responses[::-1], predictions[::-1]
    ):
        if i >= 10:
            break
        logger.info(f"Sample >>> {sample}")
        logger.info(f"Prompt >>> {prompt}")
        logger.info(f"Response >>> {response}\n")
        logger.info(f"Pred >>> {pred}\n")
        i += 1

    # Evaluation
    result, result_by_class = evaluate(predictions, input_samples, label_list)

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    # Save result based on labels
    if args.eval_valid_only:
        with open(os.path.join(
            args.output_dir, "test_result_by_class_dropnull.json"), 'w', encoding='utf-8'
        ) as f_out:
            f_out.write(json.dumps(result_by_class, indent=4))
        # if args.do_test and args.sampling == "full":
        if args.do_test:
            print_pred_json(
                test_data, 
                input_samples, 
                predictions, 
                os.path.join(args.output_dir, 'rel_pred_test_dropnull.json')
            )
    else:
        with open(os.path.join(
            args.output_dir, "test_result_by_class.json"), 'w', encoding='utf-8'
        ) as f_out:
            f_out.write(json.dumps(result_by_class, indent=4))
        if args.do_test and args.sampling == "full":
            print_pred_json(
                test_data, 
                input_samples, 
                predictions, 
                os.path.join(args.output_dir, 'rel_pred_test.json')
            )


if __name__ == "__main__":
    main()