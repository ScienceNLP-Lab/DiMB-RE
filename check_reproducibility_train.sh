#!/bin/bash

ROOT=$PWD

# Designate your own path for virtual environment
venv_path=""

source $venv_path
echo "Activated DiMB-RE environment"

# Set the data directory and the version of dataset
data_dir=./data/DiMB-RE/
dataset=ner_reduced_v6.1_trg_abs_result

# Set LLM for huggingface archive
MODEL=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

# Pipeline directory
output_dir=./output
entity_output_dir="${output_dir}/entity"
entity_output_test_dir="${output_dir}/entity"
triplet_output_dir="${output_dir}/triplet"
triplet_output_test_dir="${output_dir}/triplet"
certainty_output_dir="${output_dir}/certainty"

# Set different seed number at here
SEED=2025

# Step 1. Train and Inference for Entity & Trigger Extraction model
task=pn_reduced_trg
pipeline_task=entity

ner_plm_lr=1e-5
ner_task_lr=1e-3
ner_cw=300
max_seq_length=512
max_span_len_ent=8
max_span_len_trg=4
ner_patience=4
n_epochs=78
ner_bs=128

python run_entity_trigger.py \
  --task $task --pipeline_task $pipeline_task \
  --do_train --do_predict_test \
  --output_dir $output_dir \
  --entity_output_dir $entity_output_dir \
  --data_dir "${data_dir}${dataset}" \
  --context_window $ner_cw --max_seq_length $max_seq_length \
  --train_batch_size $ner_bs  --eval_batch_size $ner_bs \
  --learning_rate $ner_plm_lr --task_learning_rate $ner_task_lr \
  --num_epoch $n_epochs --eval_per_epoch 0.33 --max_patience $ner_patience \
  --model $MODEL \
  --max_span_length_entity $max_span_len_ent --max_span_length_trigger $max_span_len_trg \
  --extract_trigger --dual_classifier \
  --seed $SEED

# Evaluation
dataset_name=pn_reduced_trg
task=test
pred_file=ent_pred_${task}.json

python run_eval.py \
  --prediction_file "${entity_output_dir}/${pred_file}" \
  --output_dir ${entity_output_dir} \
  --task $task \
  --dataset_name $dataset_name


# Step 2. Train and Inference for RE model
task=pn_reduced_trg
pipeline_task=triplet

# Optimal hyperparams for RE w/ Typed Trigger
re_lr=2e-5
re_cw=100
re_max_len=300
re_patience=4
sampling_p=0.0
n_epochs=12
re_bs=128

python run_triplet_classification.py \
  --task $task --pipeline_task $pipeline_task \
  --do_train --do_predict_test \
  --output_dir $output_dir --entity_output_dir $entity_output_dir \
  --entity_output_test_dir $entity_output_test_dir \
  --triplet_output_dir $triplet_output_dir \
  --train_file "${data_dir}${dataset}"/train.json \
  --dev_file "${data_dir}${dataset}"/dev.json \
  --test_file "${data_dir}${dataset}"/test.json \
  --context_window $re_cw --max_seq_length $re_max_len \
  --train_batch_size $re_bs --eval_batch_size $re_bs --learning_rate $re_lr \
  --num_epoch $n_epochs  --max_patience $re_patience --sampling_proportion $sampling_p \
  --model $MODEL \
  --binary_classification \
  --seed $SEED
  
# RE evaluation
dataset_name=pn_reduced_trg
task=test
pred_file=trg_pred_${task}.json

python run_eval.py \
  --prediction_file "${triplet_output_dir}/${pred_file}" \
  --output_dir ${triplet_output_dir} \
  --task $task \
  --dataset_name $dataset_name


# Step 3. Train and Inference for Factuality Detection (end-to-end)
task=pn_reduced_trg
pipeline_task=certainty

fd_lr=3e-5
fd_cw=0
fd_max_len=200
fd_patience=4
sampling_p=0.0
n_epochs=7
fd_bs=128

python run_certainty_detection.py \
  --task $task --pipeline_task $pipeline_task \
  --do_train --do_predict_test \
  --output_dir $output_dir --relation_output_dir $triplet_output_dir \
  --relation_output_test_dir $triplet_output_test_dir \
  --certainty_output_dir $certainty_output_dir \
  --train_file "${data_dir}${dataset}"/train.json \
  --dev_file "${data_dir}${dataset}"/dev.json \
  --test_file "${data_dir}${dataset}"/test.json \
  --context_window $fd_cw --max_seq_length $fd_max_len \
  --train_batch_size $fd_bs --eval_batch_size $fd_bs --learning_rate $fd_lr \
  --num_epoch $n_epochs  --max_patience $fd_patience --sampling_proportion $sampling_p \
  --model $MODEL \
  --use_trigger \
  --seed $SEED
  
# End-to-end evaluation
dataset_name=pn_reduced_trg
task=test
pred_file=certainty_pred_${task}.json

python run_eval.py \
  --prediction_file "${certainty_output_dir}/${pred_file}" \
  --output_dir ${certainty_output_dir} \
  --task $task \
  --dataset_name $dataset_name
