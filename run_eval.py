import argparse

from shared.data_structures_copied import Dataset, evaluate_predictions

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

    print('Evaluation result %s'%(args.prediction_file))

    print('NER - P: %f, R: %f, F1: %f'%(eval_result['ner']['precision'], eval_result['ner']['recall'], eval_result['ner']['f1']))
    # print('NER - Pred: %f, Gold: %f, Correct: %f'%(eval_result['ner']['n_pred'], eval_result['ner']['n_gold'], eval_result['ner']['n_correct']))
    
    print('NER Relaxed - P: %f, R: %f, F1: %f'%(eval_result['ner_soft']['precision'], eval_result['ner_soft']['recall'], eval_result['ner_soft']['f1']))
    # print('NER Soft - Pred: %f, Gold: %f, Correct: %f'%(eval_result['ner_soft']['n_pred'], eval_result['ner_soft']['n_gold'], eval_result['ner_soft']['n_correct']))
    
    print('TRG - P: %f, R: %f, F1: %f'%(eval_result['trigger']['precision'], eval_result['trigger']['recall'], eval_result['trigger']['f1']))
    # print('TRG - Pred: %f, Gold: %f, Correct: %f'%(eval_result['trigger']['n_pred'], eval_result['trigger']['n_gold'], eval_result['trigger']['n_correct']))
    
    print('TRG Relaxed - P: %f, R: %f, F1: %f'%(eval_result['trigger_soft']['precision'], eval_result['trigger_soft']['recall'], eval_result['trigger_soft']['f1']))
    # print('TRG Soft - Pred: %f, Gold: %f, Correct: %f'%(eval_result['ner_soft']['n_pred'], eval_result['ner_soft']['n_gold'], eval_result['ner_soft']['n_correct']))

    print('REL Relaxed - P: %f, R: %f, F1: %f'%(eval_result['relaxed_relation']['precision'], eval_result['relaxed_relation']['recall'], eval_result['relaxed_relation']['f1']))
    # print('REL - Pred: %f, Gold: %f, Correct: %f'%(eval_result['relation']['n_pred'], eval_result['relation']['n_gold'], eval_result['relation']['n_correct']))

    print('REL Strict - P: %f, R: %f, F1: %f'%(eval_result['strict_relation']['precision'], eval_result['strict_relation']['recall'], eval_result['strict_relation']['f1']))
    # print('REL (strict) - Pred: %f, Gold: %f, Correct: %f'%(eval_result['strict_relation']['n_pred'], eval_result['strict_relation']['n_gold'], eval_result['strict_relation']['n_correct']))

    print('REL Relaxed+Factuality - P: %f, R: %f, F1: %f'%(eval_result['relaxed_relation_fact']['precision'], eval_result['relaxed_relation_fact']['recall'], eval_result['relaxed_relation_fact']['f1']))
    # print('REL - Pred: %f, Gold: %f, Correct: %f'%(eval_result['relation']['n_pred'], eval_result['relation']['n_gold'], eval_result['relation']['n_correct']))

    print('REL Strict+Factuality - P: %f, R: %f, F1: %f'%(eval_result['strict_relation_fact']['precision'], eval_result['strict_relation_fact']['recall'], eval_result['strict_relation_fact']['f1']))
    # print('REL (strict) - Pred: %f, Gold: %f, Correct: %f'%(eval_result['strict_relation']['n_pred'], eval_result['strict_relation']['n_gold'], eval_result['strict_relation']['n_correct']))
