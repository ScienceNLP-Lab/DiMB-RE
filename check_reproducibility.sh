#!/bin/bash

task=pn_reduced_trg
data_dir=./data/pernut/
dataset=ner_reduced_v6.1_trg_abs
pipeline_task=triplet
MODEL=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

output_dir=./output
entity_output_test_dir="${output_dir}/pred_ent_trg_result"
triplet_output_dir="${entity_output_test_dir}/triplet"

# Optimal hyperparams for RE w/ Typed Trigger
re_lr=2e-5
re_cw=100
re_max_len=300
re_patience=4
sampling_p=0.0
n_epochs=12
batch_size=32

python run_triplet_classification.py \
  --task $task --pipeline_task $pipeline_task \
  --do_predict_test \
  --output_dir $output_dir \
  --entity_output_test_dir $entity_output_test_dir \
  --train_file "${data_dir}${dataset}"/train.json \
  --dev_file "${data_dir}${dataset}"/dev.json \
  --test_file "${data_dir}${dataset}"/test.json \
  --context_window $re_cw --max_seq_length $re_max_len \
  --train_batch_size 32 --eval_batch_size 32 --learning_rate $re_lr \
  --num_epoch $n_epochs  --max_patience $re_patience --sampling_proportion $sampling_p \
  --model $MODEL \
  --finetuned_model gbhong/BiomedBERT-fulltext_finetuned_DiMB-RE \
  --binary_classification
  
# Output end-to-end evaluation results for RE
dataset_name=pn_reduced_trg
task=test
pred_file=trg_pred_${task}.json

python run_eval.py \
  --prediction_file "${triplet_output_dir}/${pred_file}" \
  --output_dir ${triplet_output_dir} \
  --task $task \
  --dataset_name $dataset_name
