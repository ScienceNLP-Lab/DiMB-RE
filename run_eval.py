import argparse
import os
from shared.data_structures import Dataset, evaluate_predictions
from shared.utils import generate_analysis_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', type=str, default=None, required=True)
    parser.add_argument('--output_dir', type=str, default=None, required=True)
    parser.add_argument('--task', type=str, default=None, required=True)
    parser.add_argument('--dataset_name', type=str, default=None, required=True)
    args = parser.parse_args()

    data = Dataset(args.prediction_file)
    eval_result = evaluate_predictions(
        data, args.output_dir, task=args.task, dataset_name=args.dataset_name
    )

    args.csv_dir = os.path.join(args.output_dir, "analysis.csv")
    generate_analysis_csv(data, args.csv_dir)

    print('Evaluation result %s'%(args.prediction_file))

    print('NER - P: %.4f, R: %.4f, F1: %.4f'%(eval_result['ner']['precision'], eval_result['ner']['recall'], eval_result['ner']['f1']))
    # print('NER - Pred: %.4f, Gold: %.4f, Correct: %.4f'%(eval_result['ner']['n_pred'], eval_result['ner']['n_gold'], eval_result['ner']['n_correct']))
    
    print('NER Relaxed - P: %.4f, R: %.4f, F1: %.4f'%(eval_result['ner_soft']['precision'], eval_result['ner_soft']['recall'], eval_result['ner_soft']['f1']))
    # print('NER Soft - Pred: %.4f, Gold: %.4f, Correct: %.4f'%(eval_result['ner_soft']['n_pred'], eval_result['ner_soft']['n_gold'], eval_result['ner_soft']['n_correct']))
    
    print('TRG - P: %.4f, R: %.4f, F1: %.4f'%(eval_result['trigger']['precision'], eval_result['trigger']['recall'], eval_result['trigger']['f1']))
    # print('TRG - Pred: %.4f, Gold: %.4f, Correct: %.4f'%(eval_result['trigger']['n_pred'], eval_result['trigger']['n_gold'], eval_result['trigger']['n_correct']))
    
    print('TRG Relaxed - P: %.4f, R: %.4f, F1: %.4f'%(eval_result['trigger_soft']['precision'], eval_result['trigger_soft']['recall'], eval_result['trigger_soft']['f1']))
    # print('TRG Soft - Pred: %.4f, Gold: %.4f, Correct: %.4f'%(eval_result['ner_soft']['n_pred'], eval_result['ner_soft']['n_gold'], eval_result['ner_soft']['n_correct']))

    print('REL Relaxed - P: %.4f, R: %.4f, F1: %.4f'%(eval_result['relaxed_relation']['precision'], eval_result['relaxed_relation']['recall'], eval_result['relaxed_relation']['f1']))
    # print('REL Relaxed - Pred: %.4f, Gold: %.4f, Correct: %.4f'%(eval_result['relaxed_relation']['n_pred'], eval_result['relaxed_relation']['n_gold'], eval_result['relaxed_relation']['n_correct']))

    print('REL Strict - P: %.4f, R: %.4f, F1: %.4f'%(eval_result['strict_relation']['precision'], eval_result['strict_relation']['recall'], eval_result['strict_relation']['f1']))
    # print('REL Strict - Pred: %.4f, Gold: %.4f, Correct: %.4f'%(eval_result['strict_relation']['n_pred'], eval_result['strict_relation']['n_gold'], eval_result['strict_relation']['n_correct']))

    print('REL Relaxed+Factuality - P: %.4f, R: %.4f, F1: %.4f'%(eval_result['relaxed_relation_fact']['precision'], eval_result['relaxed_relation_fact']['recall'], eval_result['relaxed_relation_fact']['f1']))
    # print('REL - Pred: %.4f, Gold: %.4f, Correct: %.4f'%(eval_result['relation']['n_pred'], eval_result['relation']['n_gold'], eval_result['relation']['n_correct']))

    print('REL Strict+Factuality - P: %.4f, R: %.4f, F1: %.4f'%(eval_result['strict_relation_fact']['precision'], eval_result['strict_relation_fact']['recall'], eval_result['strict_relation_fact']['f1']))
    # print('REL (strict) - Pred: %.4f, Gold: %.4f, Correct: %.4f'%(eval_result['strict_relation']['n_pred'], eval_result['strict_relation']['n_gold'], eval_result['strict_relation']['n_correct']))
