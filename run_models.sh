#!/bin/bash

ROOT=$PWD

source /jet/home/ghong1/miniconda3/bin/activate pure
echo "Activated pure"

# NER -> RE Model
task=pn_reduced_trg
# task=pn_reduced_trg_dummy

data_dir=./data/DiMB-RE/
# dataset="ner_reduced_v6.1_trg_abs"
dataset="ner_reduced_v6.1_trg_abs_result"

# FIXED NER Hyperparameters
# n_epochs=200
# ner_plm_lr=1e-5
# ner_task_lr=5e-4
# ner_cw=300
# max_seq_length=512
# max_span_len_ent=8 # FIXED
# max_span_len_trg=4 # FIXED
# ner_patience=4 # FIXED

# FIXED RE Hyperparameters
# n_epochs=20
# re_cw=100
# re_max_len=300
# re_lr=3e-5
# sampling_p=0.0  # FIXED
# re_patience=4

# #### TASK 4: RE with Trigger (Typed and Untyped) ####
# pipeline_task='triplet'
# MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# output_dir=../ocean_cis230030p/ghong1/PN/output/${dataset}
# entity_output_dir="${output_dir}/EXP_2/entity"
# triplet_output_dir="${output_dir}/EXP_4/triplet"

# re_lr=2e-5
# re_cw=0
# re_max_len=200
# sampling_p=0.0
# n_epochs=12
# python run_triplet_classification.py \
#     --task $task --pipeline_task $pipeline_task \
#     --do_predict_dev \
#     --output_dir $output_dir --entity_output_dir $entity_output_dir \
#     --triplet_output_dir $triplet_output_dir \
#     --train_file "${data_dir}${dataset}"/train.json \
#     --dev_file "${data_dir}${dataset}"/dev.json \
#     --test_file "${data_dir}${dataset}"/test.json \
#     --context_window $re_cw --max_seq_length $re_max_len \
#     --train_batch_size 128 --eval_batch_size 32 --learning_rate $re_lr \
#     --num_epoch $n_epochs \
#     --model $MODEL \
#     --binary_classification --sampling_method trigger_position \
#     --load_saved_model


#### TASK 5: Certainty Detection with Trigger provided ####
output_dir=../ocean_cis230030p/ghong1/PN/output/${dataset}

cer_cw=0 # FIXED
cer_max_len=200 # FIXED

cer_lr=3e-5
n_epochs=7

MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
pipeline_task=certainty

relation_output_dir="${output_dir}/EXP_13/triplet"
# # For Test set
# relation_output_test_dir="${output_dir}/EXP_2/relation"
# To use fine-tuned model
certainty_output_dir="${output_dir}/EXP_14/certainty"

python run_certainty_detection.py \
    --task $task --pipeline_task $pipeline_task \
    --do_predict_dev \
    --output_dir $output_dir \
    --relation_output_dir $relation_output_dir \
    --train_file "${data_dir}${dataset}"/train.json \
    --dev_file "${data_dir}${dataset}"/dev.json \
    --test_file "${data_dir}${dataset}"/test.json \
    --context_window $cer_cw --max_seq_length $cer_max_len \
    --train_batch_size 64 --eval_batch_size 32 \
    --learning_rate $cer_lr \
    --num_epoch $n_epochs \
    --model $MODEL --do_lower_case --add_new_tokens \
    --certainty_output_dir $certainty_output_dir \
    --use_trigger \
    # --eval_with_gold \
    # --load_saved_model \

# seeds=(1 2 3 4)
# for SEED in "${seeds[@]}"; do

#     #### TASK 5: Certainty Detection with Trigger provided ####
#     dataset=ner_reduced_v6.1_trg_abs
#     output_dir=../ocean_cis230030p/ghong1/PN/output/${dataset}_SEED${SEED}

#     cer_cw=0 # FIXED
#     cer_max_len=200 # FIXED

#     cer_lr=2e-5
#     n_epochs=11

#     MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
#     pipeline_task=certainty
    
#     relation_output_dir="${output_dir}/EXP_2/relation"
#     # For Test set
#     relation_output_test_dir="${output_dir}/EXP_2/relation"
#     # To use fine-tuned model
#     certainty_output_dir="${output_dir}/EXP_6/certainty"

#     python run_certainty_detection.py \
#     --task $task --pipeline_task $pipeline_task \
#     --do_predict_test \
#     --output_dir $output_dir --relation_output_dir $relation_output_dir \
#     --train_file "${data_dir}${dataset}"/train.json \
#     --dev_file "${data_dir}${dataset}"/dev.json \
#     --test_file "${data_dir}${dataset}"/test.json \
#     --context_window $cer_cw --max_seq_length $cer_max_len \
#     --train_batch_size 64 --eval_batch_size 64 --learning_rate $cer_lr \
#     --num_epoch $n_epochs \
#     --model $MODEL --do_lower_case --add_new_tokens \
#     --relation_output_test_dir $relation_output_test_dir \
#     --seed $SEED \
#     --certainty_output_dir $certainty_output_dir \
#     --eval_with_gold \
#     # --load_saved_model \
#     # --use_trigger \

#     relation_output_dir="${output_dir}/EXP_4/triplet"
#     relation_output_test_dir="${output_dir}/EXP_4/triplet"
#     certainty_output_dir="${output_dir}/EXP_7/certainty"

#     cer_lr=3e-5
#     n_epochs=7
#     python run_certainty_detection.py \
#     --task $task --pipeline_task $pipeline_task \
#     --do_predict_test \
#     --output_dir $output_dir --relation_output_dir $relation_output_dir \
#     --train_file "${data_dir}${dataset}"/train.json \
#     --dev_file "${data_dir}${dataset}"/dev.json \
#     --test_file "${data_dir}${dataset}"/test.json \
#     --context_window $cer_cw --max_seq_length $cer_max_len \
#     --train_batch_size 64 --eval_batch_size 64 --learning_rate $cer_lr \
#     --num_epoch $n_epochs \
#     --model $MODEL --do_lower_case --add_new_tokens \
#     --use_trigger \
#     --relation_output_test_dir $relation_output_test_dir \
#     --seed $SEED \
#     --certainty_output_dir $certainty_output_dir \
#     --eval_with_gold \
#     # --load_saved_model \


#     dataset="ner_reduced_v6.1_trg_abs_result"
#     output_dir=../ocean_cis230030p/ghong1/PN/output/${dataset}_SEED${SEED}

#     relation_output_dir="${output_dir}/EXP_1/triplet"
#     relation_output_test_dir="${output_dir}/EXP_1/triplet"
#     certainty_output_dir="${output_dir}/EXP_3/certainty"
#     cer_lr=3e-5
#     n_epochs=5
#     python run_certainty_detection.py \
#     --task $task --pipeline_task $pipeline_task \
#     --do_predict_test \
#     --output_dir $output_dir --relation_output_dir $relation_output_dir \
#     --train_file "${data_dir}${dataset}"/train.json \
#     --dev_file "${data_dir}${dataset}"/dev.json \
#     --test_file "${data_dir}${dataset}"/test.json \
#     --context_window $cer_cw --max_seq_length $cer_max_len \
#     --train_batch_size 64 --eval_batch_size 64 --learning_rate $cer_lr \
#     --num_epoch $n_epochs \
#     --model $MODEL --do_lower_case --add_new_tokens \
#     --relation_output_test_dir $relation_output_test_dir \
#     --use_trigger \
#     --seed $SEED \
#     --certainty_output_dir $certainty_output_dir \
#     --eval_with_gold \
#     # --load_saved_model \
# done


################## PRESET TASKS ######################

# #### TASK 1: NER with vanilla PURE model (without Trigger) ####
# pipeline_task='entity'
# MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# ner_plm_lr=1e-5
# ner_task_lr=1e-3
# ner_cw=100
# max_seq_length=300
# max_span_len_ent=8
# n_epochs=102
# python run_entity_trigger.py \
# --task $task --pipeline_task $pipeline_task \
# --do_train --do_predict_test \
# --output_dir $output_dir \
# --data_dir "${data_dir}${dataset}" \
# --context_window $ner_cw --max_seq_length $max_seq_length \
# --train_batch_size 64  --eval_batch_size 64 \
# --learning_rate $ner_plm_lr --task_learning_rate $ner_task_lr \
# --num_epoch $n_epochs --eval_per_epoch 0.33 --max_patience $ner_patience \
# --model $MODEL \
# --max_span_length_entity $max_span_len_ent --max_span_length_trigger $max_span_len_trg \
# # --extract_trigger --untyped_trigger \



# #### TASK 2: RE with vanilla PURE model (without Trigger) ####
# pipeline_task='relation'

# re_lr=5e-5
# re_cw=0
# re_max_len=200
# sampling_proportion=0
# n_epochs=17
# MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# entity_output_dir="${output_dir}/EXP_16/entity"
# entity_output_test_dir="${output_dir}/EXP_24/entity"
# python run_relation_with_trigger.py \
# --task $task --pipeline_task $pipeline_task \
# --do_train --do_predict_test \
# --output_dir $output_dir --entity_output_dir $entity_output_dir \
# --entity_output_test_dir $entity_output_test_dir \
# --train_file "${data_dir}${dataset}"/train.json \
# --context_window $re_cw --max_seq_length $re_max_len \
# --train_batch_size 128 --eval_batch_size 128 --learning_rate $re_lr \
# --num_epoch $n_epochs  --max_patience 4 --sampling_proportion $sampling_proportion \
# --model $MODEL


# #### TASK 3: NER with Trigger ####
# pipeline_task='entity'

# MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# ner_plm_lr=1e-5
# ner_task_lr=1e-3
# ner_cw=300
# max_seq_length=512
# n_epochs=78

# entity_output_dir="${output_dir}/EXP_68/entity"

# python run_entity_trigger.py \
# --task $task --pipeline_task $pipeline_task \
# --do_train --do_predict_test \
# --output_dir $output_dir \
# --entity_output_dir $entity_output_dir \
# --data_dir "${data_dir}${dataset}" \
# --context_window $ner_cw --max_seq_length $max_seq_length \
# --train_batch_size 128  --eval_batch_size 128 \
# --learning_rate $ner_plm_lr --task_learning_rate $ner_task_lr \
# --num_epoch $n_epochs --eval_per_epoch 0.33 --max_patience $ner_patience \
# --model $MODEL \
# --max_span_length_entity $max_span_len_ent --max_span_length_trigger $max_span_len_trg \
# --extract_trigger --dual_classifier


# task=pn_reduced_trg_dummy
# ner_plm_lr=5e-5
# ner_task_lr=5e-4
# ner_cw=300
# max_seq_length=512
# n_epochs=87

# entity_output_dir="${output_dir}/EXP_66/entity"

# python run_entity_trigger.py \
# --task $task --pipeline_task $pipeline_task \
# --do_train --do_predict_test \
# --output_dir $output_dir \
# --entity_output_dir $entity_output_dir \
# --data_dir "${data_dir}${dataset}" \
# --context_window $ner_cw --max_seq_length $max_seq_length \
# --train_batch_size 128  --eval_batch_size 128 \
# --learning_rate $ner_plm_lr --task_learning_rate $ner_task_lr \
# --num_epoch $n_epochs --eval_per_epoch 0.33 --max_patience $ner_patience \
# --model $MODEL \
# --max_span_length_entity $max_span_len_ent --max_span_length_trigger $max_span_len_trg \
# --extract_trigger --untyped_trigger --dual_classifier


# #### TASK 4: RE with Trigger (Typed and Untyped) ####
# pipeline_task='triplet'
# MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# entity_output_dir="${output_dir}/EXP_68/entity"
# entity_output_test_dir="${output_dir}/EXP_77/entity"
# re_lr=2e-5
# re_cw=0
# re_max_len=200
# sampling_p=0.0
# n_epochs=12
# python run_triplet_classification.py \
# --task $task --pipeline_task $pipeline_task \
# --do_train --do_predict_test \
# --output_dir $output_dir --entity_output_dir $entity_output_dir \
# --entity_output_test_dir $entity_output_test_dir \
# --train_file "${data_dir}${dataset}"/train.json \
# --dev_file "${data_dir}${dataset}"/dev.json \
# --test_file "${data_dir}${dataset}"/test.json \
# --context_window $re_cw --max_seq_length $re_max_len \
# --train_batch_size 128 --eval_batch_size 128 --learning_rate $re_lr \
# --num_epoch $n_epochs  --max_patience $re_patience --sampling_proportion $sampling_p \
# --model $MODEL \
# --binary_classification --sampling_method trigger_position


# #### TASK 5: Certainty Detection with Trigger provided ####
# cer_cw=0
# cer_max_len=200
# cer_lr=2e-5
# n_epochs=20
# MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# # make model-specific output folder and put everything in there
# output_dir=../tmp_ondemand_ocean_cis230030p_symlink/ghong1/PN/output/${dataset}

# task=pn_reduced_trg
# pipeline_task="certainty"
# relation_output_dir="${output_dir}/EXP_88/triplet"

# # For Test set
# relation_output_test_dir="${output_dir}/EXP_98/triplet"
# # certainty_output_dir="${output_dir}/EXP_###/certainty"

# sampling_p=0.0
# n_epochs=3
# python run_certainty_detection.py \
# --task $task --pipeline_task $pipeline_task \
# --do_train --do_predict_test --eval_with_gold \
# --output_dir $output_dir --relation_output_dir $relation_output_dir \
# --train_file "${data_dir}${dataset}"/train.json \
# --dev_file "${data_dir}${dataset}"/dev.json \
# --test_file "${data_dir}${dataset}"/test.json \
# --context_window $cer_cw --max_seq_length $cer_max_len \
# --train_batch_size 64 --eval_batch_size 64 --learning_rate $cer_lr \
# --num_epoch $n_epochs  --max_patience $re_patience \
# --sampling_proportion $sampling_p \
# --model $MODEL --do_lower_case --add_new_tokens \
# --use_trigger \
# --relation_output_test_dir $relation_output_test_dir \
# # --certainty_output_dir $certainty_output_dir \

