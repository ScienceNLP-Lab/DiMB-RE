#!/bin/bash

source /jet/home/ghong1/miniconda3/bin/activate PL-Marker
echo "Activated PL-Marker"

# --use_ner_results: use the original entity type predicted by NER models
# DiMB-RE
# mkdir dimb-re_models

# split=dev
split=test
output_dir=/jet/home/ghong1/ocean_cis230030p/ghong1/PL-Marker/probing_exp

for seed in 0 1 2 3 4; do 
    python3 run_fd_trg_marked.py \
    --model_type bertsub \
    --model_name_or_path /jet/home/ghong1/ocean_cis230030p/ghong1/BiomedBERT-fulltext \
    --do_lower_case \
    --data_dir ../data/DiMB-RE/ner_reduced_v6.1_trg_abs_result \
    --learning_rate 2e-5 --num_train_epochs 3 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 256 \
    --max_pair_length 16 \
    --save_steps 221 \
    --eval_logsoftmax \
    --seed $seed \
    --output_dir $output_dir/FD/dimb-re_models_biomedbert_typedmarker_nosym_${split}-${seed} \
    --test_file $output_dir/RE/dimb-re_models_biomedbert_typedmarker_nosym_${split}-${seed}/rel_pred_${split}.json \
    --overwrite_output_dir \
    --use_ner_results \
    --use_typemarker \
    --lminit \
    --eval_all_checkpoints \
    --no_sym 
    # --do_train
    # --evaluate_during_training \
    # --do_eval \    
    # --no_test \

    # --test_file $output_dir/RE/dimb-re_models_biomedbert_typedmarker_nosym_${split}-${seed}/rel_pred_${split}_goldner.json \
    
    # --dev_file $output_dir/RE/dimb-re_models_biomedbert_trg_inserted_dev-$seed/rel_pred_dev.json \
    # --test_file $output_dir/RE/dimb-re_models_biomedbert_trg_${split}-${seed}/ent_pred_${split}.json \
    # --eval_unidirect \
done;

