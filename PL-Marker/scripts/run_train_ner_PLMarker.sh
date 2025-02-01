#!/bin/bash

source /jet/home/ghong1/miniconda3/bin/activate PL-Marker
echo "Activated PL-Marker"

GPU_ID=0

# split=dev
split=test

### SET YOUR OWN OUTPUT DIR ###
# output_dir=/jet/home/ghong1/ocean_cis230030p/ghong1/PL-Marker/no-result
output_dir=./output

# DiMB-RE
# mkdir dimb-re_models
for seed in 0 1 2 3 4; do 
# for seed in 0; do 
    python run_acener_trg_modified.py \
        --model_type bertspanmarker \
        --model_name_or_path microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
        --do_lower_case \
        --data_dir ../data/DiMB-RE/ner_reduced_v6.1_trg_abs_result \
        --train_file train.json --dev_file dev.json --test_file test.json  \
        --learning_rate 2e-5 \
        --num_train_epochs 28 \
        --per_gpu_train_batch_size 8 \
        --per_gpu_eval_batch_size 16 \
        --gradient_accumulation_steps 1  \
        --max_seq_length 512  --save_steps 1308 \
        --max_pair_length 256  --max_mention_ori_length 8 \
        --seed $seed \
        --onedropout \
        --lminit \
        --output_dir $output_dir/NER/dimb-re_models_biomedbert_trg_$split-$seed \
        --overwrite_output_dir \
        --output_results \
        --do_train \
        --do_test
        # --do_eval \
        # --evaluate_during_training 
        
done;
