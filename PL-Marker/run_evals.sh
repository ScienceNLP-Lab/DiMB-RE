#!/bin/bash

source /jet/home/ghong1/miniconda3/bin/activate pure
echo "Activated pure"

# dataset="ner_reduced_v6.1_trg_abs"
# dataset="ner_reduced_v6.1_trg_abs_result"

# If TypedTrigger or Gold set Eval
dataset_name="pn_reduced_trg"

# # If Untyped Trigger
# dataset_name="pn_reduced_trg_dummy" 

# for i in "${indices[@]}"; do
for seed in 0 1 2 3 4; do
# for seed in 0; do

    echo "SEED: $seed"
    # task=dev
    task=test

    # output_dir=../ocean_cis230030p/ghong1/PL-Marker
    # exp_dir=${output_dir}/NER/dimb-re_models_biomedbert_trg_${task}-${seed}
    # exp_dir=${output_dir}/RE/dimb-re_models_biomedbert_trg_inserted_${task}-${seed}
    # exp_dir=${output_dir}/FD/dimb-re_models_biomedbert_trg_inserted_${task}-${seed}

    # output_dir=../ocean_cis230030p/ghong1/PL-Marker/probing_exp
    # exp_dir=${output_dir}/NER/dimb-re_models_biomedbert_${task}-${seed}
    # exp_dir=${output_dir}/RE/dimb-re_models_biomedbert_typedmarker_nosym_${task}-${seed}
    # exp_dir=${output_dir}/FD/dimb-re_models_biomedbert_typedmarker_nosym_${task}-${seed}

    output_dir=../ocean_cis230030p/ghong1/PL-Marker/no-result
    # exp_dir=${output_dir}/NER/dimb-re_models_biomedbert_trg_${task}-${seed}
    # exp_dir=${output_dir}/RE/dimb-re_models_biomedbert_trg_inserted_${task}-${seed}
    exp_dir=${output_dir}/FD/dimb-re_models_biomedbert_trg_inserted_${task}-${seed}

    pred_file=rel_pred_${task}_goldrel.json
    # pred_file=rel_pred_$task.json
    # pred_file=rel_pred_${task}_goldner.json
    # pred_file=ent_pred_$task.json
    python run_eval.py \
        --prediction_file "${exp_dir}/${pred_file}" \
        --output_dir $exp_dir \
        --task $task \
        --dataset_name $dataset_name \
        --do_eval_rel \
        --print_trigger
    echo ""

    pred_file=rel_pred_${task}_goldner.json
    python run_eval.py \
        --prediction_file "${exp_dir}/${pred_file}" \
        --output_dir $exp_dir \
        --task $task \
        --dataset_name $dataset_name \
        --do_eval_rel \
        --print_trigger        
    echo ""

    pred_file=rel_pred_$task.json
    # pred_file=rel_pred_${task}_goldner.json
    # pred_file=ent_pred_$task.json
    python run_eval.py \
        --prediction_file "${exp_dir}/${pred_file}" \
        --output_dir $exp_dir \
        --task $task \
        --dataset_name $dataset_name \
        --do_eval_rel \
        --print_trigger
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