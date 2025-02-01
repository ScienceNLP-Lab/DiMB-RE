#!/bin/bash

source /jet/home/ghong1/miniconda3/bin/activate pure
echo "Activated pure"

# dataset="ner_reduced_v6.1_trg_abs"
dataset="ner_reduced_v6.1_trg_abs_result"

# If TypedTrigger or Gold set Eval
dataset_name="pn_reduced_trg"

# # If Untyped Trigger
# dataset_name="pn_reduced_trg_dummy" 

# indices=(8 9)
indices=(13)
# for i in {183..183}; do
for i in "${indices[@]}"; do
    echo EXP${i}
    output_dir=../ocean_cis230030p/ghong1/PN/output/${dataset}/EXP_${i}
    # output_dir=../ocean_cis230030p/ghong1/PN/output/${dataset}_SEED4/EXP_${i}
    task=dev
    # pred_file=certainty/certainty_pred_$task.json
    # pred_file=relation/rel_pred_$task.json
    pred_file=triplet/trg_pred_$task.json
    # pred_file=entity/ent_pred_$task.json
    python run_eval.py --prediction_file "${output_dir}/${pred_file}" --output_dir ${output_dir} --task $task --dataset_name $dataset_name
    echo ""
done

# for i in {1..4}; do
# # for i in "${indices[@]}"; do
#     echo SEED${i}
#     output_dir=../ocean_cis230030p/ghong1/PN/output/${dataset}_SEED${i}/EXP_3
#     task=test
#     pred_file=certainty/certainty_pred_$task.json
#     # pred_file=relation/rel_pred_$task.json
#     # pred_file=triplet/trg_pred_$task.json
#     # pred_file=entity/ent_pred_$task.json
#     python run_eval.py --prediction_file "${output_dir}/${pred_file}" --output_dir ${output_dir} --task $task --dataset_name $dataset_name
#     echo ""
# done