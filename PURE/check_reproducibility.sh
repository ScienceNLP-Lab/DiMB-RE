#!/bin/bash

ROOT=$PWD

# Assign your venv directory after setting up your virtual environment
venv_path=""

source $venv_path
echo "Activated virtual environment for DiMB-RE"

# Set the data directory and the version of dataset
data_dir=../data/DiMB-RE/
dataset=ner_reduced_v6.1_trg_abs_result

# Assign dirs for pipeline outputs
output_dir=./output
entity_output_dir="${output_dir}/entity"
entity_output_test_dir="${output_dir}/entity"
triplet_output_dir="${output_dir}/triplet"
triplet_output_test_dir="${output_dir}/triplet"
certainty_output_dir="${output_dir}/certainty"

# Step 1. Reproducibility check for NER
task=pn_reduced_trg
pipeline_task=entity

ner_cw=300
max_seq_length=512
max_span_len_ent=8
max_span_len_trg=4
ner_bs=128
MODEL=gbhong/BiomedBERT-fulltext_finetuned_DiMB-RE_NER

python run_entity_trigger.py \
  --task $task --pipeline_task $pipeline_task \
  --do_predict_test \
  --output_dir $output_dir \
  --entity_output_dir $entity_output_dir \
  --data_dir "${data_dir}${dataset}" \
  --context_window $ner_cw --max_seq_length $max_seq_length \
  --eval_batch_size $ner_bs \
  --model $MODEL \
  --max_span_length_entity $max_span_len_ent --max_span_length_trigger $max_span_len_trg \
  --extract_trigger --dual_classifier \
  --seed $SEED \

# Evaluation
dataset_name=pn_reduced_trg
task=test
pred_file=ent_pred_${task}.json

python run_eval.py \
  --prediction_file "${entity_output_dir}/${pred_file}" \
  --output_dir ${entity_output_dir} \
  --task $task \
  --dataset_name $dataset_name


# Step 2. Reproducibility check for RE
# Optimal hyperparams for RE w/ Typed Trigger
re_cw=100
re_max_len=300
batch_size=32
MODEL=gbhong/BiomedBERT-fulltext_finetuned_DiMB-RE_RE

python run_triplet_classification.py \
  --task $task --pipeline_task $pipeline_task \
  --do_predict_test \
  --output_dir $output_dir \
  --entity_output_test_dir $entity_output_test_dir \
  --triplet_output_dir $triplet_output_dir \
  --train_file "${data_dir}${dataset}"/train.json \
  --dev_file "${data_dir}${dataset}"/dev.json \
  --test_file "${data_dir}${dataset}"/test.json \
  --context_window $re_cw --max_seq_length $re_max_len \
  --eval_batch_size $batch_size \
  --model $MODEL \
  --binary_classification
  
# RE evaluation
dataset_name=pn_reduced_trg
task=test
pred_file=trg_pred_${task}.json

python run_eval.py \
  --prediction_file "${triplet_output_dir}/${pred_file}" \
  --output_dir ${triplet_output_dir} \
  --task $task \
  --dataset_name $dataset_name


# Step 3. Reproducibility check for FD (end-to-end)
task=pn_reduced_trg
pipeline_task=certainty

fd_cw=0
fd_max_len=200
batch_size=32
MODEL=gbhong/BiomedBERT-fulltext_finetuned_DiMB-RE_FD

python run_certainty_detection.py \
  --task $task --pipeline_task $pipeline_task \
  --do_predict_test \
  --output_dir $output_dir \
  --relation_output_test_dir $relation_output_test_dir \
  --certainty_output_dir $certainty_output_dir \
  --train_file "${data_dir}${dataset}"/train.json \
  --dev_file "${data_dir}${dataset}"/dev.json \
  --test_file "${data_dir}${dataset}"/test.json \
  --context_window $fd_cw --max_seq_length $fd_max_len \
  --eval_batch_size $batch_size \
  --model $MODEL \
  --use_trigger
  
# End-to-end evaluation
dataset_name=pn_reduced_trg
task=test
pred_file=certainty_pred_${task}.json

python run_eval.py \
  --prediction_file "${certainty_output_dir}/${pred_file}" \
  --output_dir ${certainty_output_dir} \
  --task $task \
  --dataset_name $dataset_name

