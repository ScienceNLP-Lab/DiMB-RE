#!/bin/bash

ROOT=$PWD

source /jet/home/ghong1/miniconda3/bin/activate pure
echo "Activated pure"

# NER -> RE Model
task=pn_reduced_trg
# task=pn_reduced_trg_dummy

data_dir=./data/pernut/
dataset="ner_reduced_v6.1_trg_abs"
# dataset="ner_reduced_v6.1_trg_abs_result"
# output_dir=../tmp_ondemand_ocean_cis230030p_symlink/ghong1/PN/output/${dataset}

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
re_patience=4

#### TASK 5: Certainty Detection with Trigger provided ####
cer_cw=0
cer_max_len=200
cer_lr=3e-5
# n_epochs=20
MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# make model-specific output folder and put everything in there
output_dir=../tmp_ondemand_ocean_cis230030p_symlink/ghong1/PN/output/${dataset}

task=pn_reduced_trg
pipeline_task=certainty
relation_output_dir="${output_dir}/EXP_159/triplet"

# For Test set
relation_output_test_dir="${output_dir}/EXP_161/triplet"
certainty_output_dir="${output_dir}/EXP_183/certainty"

sampling_p=0.0
n_epochs=7
python run_certainty_detection.py \
    --task $task --pipeline_task $pipeline_task \
    --do_predict_test --eval_with_gold \
    --output_dir $output_dir --relation_output_dir $relation_output_dir \
    --relation_output_test_dir $relation_output_test_dir \
    --train_file "${data_dir}${dataset}"/train.json \
    --dev_file "${data_dir}${dataset}"/dev.json \
    --test_file "${data_dir}${dataset}"/test.json \
    --context_window $cer_cw --max_seq_length $cer_max_len \
    --train_batch_size 64 --eval_batch_size 64 --learning_rate $cer_lr \
    --num_epoch $n_epochs  --max_patience $re_patience \
    --sampling_proportion $sampling_p \
    --model $MODEL --do_lower_case --add_new_tokens \
    --use_trigger \
    --certainty_output_dir $certainty_output_dir \
    # --load_saved_model


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







# #### TASK: Iterate Triplet Classification with Typed Triggers ####
#     task=pn_reduced_trg
#     pipeline_task="triplet"
#     entity_output_dir="${output_dir}/EXP_151/entity"
#     # For Test set
#     entity_output_test_dir="${output_dir}/EXP_179/entity"
    
    # n_epoch=6
    # sampling_p=0.0
    # python run_triplet_classification.py \
    # --task $task --pipeline_task $pipeline_task \
    # --do_train --do_predict_test \
    # --binary_classification \
    # --output_dir $output_dir --entity_output_dir $entity_output_dir \
    # --entity_output_test_dir $entity_output_test_dir \
    # --train_file "${data_dir}${dataset}"/train.json \
    # --context_window $re_cw --max_seq_length $re_max_len \
    # --train_batch_size 64 --eval_batch_size 64 --learning_rate $re_lr \
    # --num_epoch $n_epoch  --max_patience $re_patience --sampling_proportion $sampling_p \
    # --model $MODEL --do_lower_case --add_new_tokens

    # sampling_p=0.2
    # sampling_method='random'
    # n_epoch=9
    # python run_triplet_classification.py \
    # --task $task --pipeline_task $pipeline_task \
    # --do_train --do_predict_test \
    # --binary_classification \
    # --output_dir $output_dir --entity_output_dir $entity_output_dir \
    # --entity_output_test_dir $entity_output_test_dir \
    # --train_file "${data_dir}${dataset}"/train.json \
    # --context_window $re_cw --max_seq_length $re_max_len \
    # --train_batch_size 64 --eval_batch_size 64 --learning_rate $re_lr \
    # --num_epoch $n_epoch  --max_patience $re_patience \
    # --sampling_method $sampling_method --sampling_proportion $sampling_p \
    # --model $MODEL --do_lower_case --add_new_tokens

    # sampling_method_set=('trigger_position' 'random')
    # sampling_p_set=(0.2 0.5 1.0)
    # for sampling_method in "${sampling_method_set[@]}"; do
    #     for sampling_p in "${sampling_p_set[@]}"; do
    #         python run_triplet_classification.py \
    #         --task $task --pipeline_task $pipeline_task \
    #         --do_train --do_eval --do_predict_dev \
    #         --binary_classification \
    #         --output_dir $output_dir --entity_output_dir $entity_output_dir \
    #         --train_file "${data_dir}${dataset}"/train.json \
    #         --context_window $re_cw --max_seq_length $re_max_len \
    #         --train_batch_size 128 --eval_batch_size 128 --learning_rate $re_lr \
    #         --num_epoch $n_epoch  --max_patience $re_patience \
    #         --sampling_method $sampling_method --sampling_proportion $sampling_p \
    #         --model $MODEL --do_lower_case --add_new_tokens
    #     done
    # done


    # #### TASK: (Iterate) Triplet Classification with UnTyped Triggers ####

    # task=pn_reduced_trg_dummy
    # pipeline_task="triplet"
    # entity_output_dir="${output_dir}/EXP_168/entity"
    # # For TEST set
    # entity_output_test_dir="${output_dir}/EXP_182/entity"

    # sampling_p=0.0
    # n_epoch=9
    # python run_triplet_classification.py \
    # --task $task --pipeline_task $pipeline_task \
    # --do_train --do_predict_test \
    # --output_dir $output_dir --entity_output_dir $entity_output_dir \
    # --entity_output_test_dir $entity_output_test_dir \
    # --train_file "${data_dir}${dataset}"/train.json \
    # --context_window $re_cw --max_seq_length $re_max_len \
    # --train_batch_size 64 --eval_batch_size 64 --learning_rate $re_lr \
    # --num_epoch $n_epoch  --max_patience $re_patience --sampling_proportion $sampling_p \
    # --model $MODEL --do_lower_case --add_new_tokens

    # sampling_method_set=('trigger_position' 'random')
    # sampling_p_set=(0.2 0.5)
    # for sampling_method in "${sampling_method_set[@]}"; do
    #     for sampling_p in "${sampling_p_set[@]}"; do
    #         python run_triplet_classification.py \
    #         --task $task --pipeline_task $pipeline_task \
    #         --do_train --do_eval --do_predict_dev \
    #         --output_dir $output_dir --entity_output_dir $entity_output_dir \
    #         --train_file "${data_dir}${dataset}"/train.json \
    #         --context_window $re_cw --max_seq_length $re_max_len \
    #         --train_batch_size 128 --eval_batch_size 128 --learning_rate $re_lr \
    #         --num_epoch $n_epoch  --max_patience $re_patience \
    #         --sampling_method $sampling_method --sampling_proportion $sampling_p \
    #         --model $MODEL --do_lower_case --add_new_tokens
    #     done
    # done


    # #### TASK: Triplet Classification with GOLD Typed Triggers ####
    # task=pn_reduced_trg
    # pipeline_task="triplet"
    # entity_output_dir="${output_dir}/EXP_151/entity"
    # # For Test set
    # entity_output_test_dir="${output_dir}/EXP_179/entity"
    # triplet_output_dir="${output_dir}/EXP_209/triplet"
    
    # n_epoch=8
    # sampling_p=0.0
    # python run_triplet_classification.py \
    # --task $task --pipeline_task $pipeline_task \
    # --do_predict_test \
    # --eval_with_gold \
    # --binary_classification \
    # --output_dir $output_dir --entity_output_dir $entity_output_dir \
    # --entity_output_test_dir $entity_output_test_dir --triplet_output_dir $triplet_output_dir\
    # --train_file "${data_dir}${dataset}"/train.json \
    # --context_window $re_cw --max_seq_length $re_max_len \
    # --train_batch_size 64 --eval_batch_size 32 --learning_rate $re_lr \
    # --num_epoch $n_epoch  --max_patience $re_patience \
    # --sampling_proportion $sampling_p \
    # --model $MODEL --do_lower_case --add_new_tokens

    # #### TASK: Triplet Classification with GOLD UnTyped Triggers ####
    # task=pn_reduced_trg_dummy
    # pipeline_task="triplet"
    # entity_output_dir="${output_dir}/EXP_168/entity"
    # # For Test set
    # entity_output_test_dir="${output_dir}/EXP_182/entity"
    # triplet_output_dir="${output_dir}/EXP_210/triplet"
    
    # n_epoch=5
    # sampling_p=0.0
    # python run_triplet_classification.py \
    # --task $task --pipeline_task $pipeline_task \
    # --do_predict_test \
    # --eval_with_gold \
    # --output_dir $output_dir --entity_output_dir $entity_output_dir \
    # --entity_output_test_dir $entity_output_test_dir --triplet_output_dir $triplet_output_dir \
    # --train_file "${data_dir}${dataset}"/train.json \
    # --context_window $re_cw --max_seq_length $re_max_len \
    # --train_batch_size 64 --eval_batch_size 32 --learning_rate $re_lr \
    # --num_epoch $n_epoch  --max_patience $re_patience \
    # --sampling_proportion $sampling_p \
    # --model $MODEL --do_lower_case --add_new_tokens




    # ner_cw=300
    # ner_lr=5e-5
    # n_epoch=112
    # # # task=pn_reduced_trg_dummy
    # python run_entity_trigger.py --do_train --do_predict_test \
    # --max_span_length_entity $max_span_len_ent --max_span_length_trigger $max_span_len_trg \
    # --learning_rate $ner_lr --task_learning_rate 5e-4 --context_window $ner_cw \
    # --train_batch_size 64  --eval_batch_size 64 --num_epoch $n_epoch \
    # --eval_per_epoch 0.25 --max_patience $ner_patience \
    # --task $task --pipeline_task $pipeline_task --model $MODEL \
    # --data_dir "${data_dir}${dataset}" --output_dir $output_dir \
    # --extract_trigger
    # # --untyped_trigger

    # if [ "$MODEL" = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" ]; then
    #     entity_output_dir="${output_dir}/EXP_35/entity"
    # else
    #     entity_output_dir="${output_dir}/EXP_69/entity"
    # fi

    # n_epoch=11
    # sampling_p_trg=0.0
    # sampling_method='random'
    # python run_triplet_classification.py --do_train --do_predict_test \
    # --do_lower_case --add_new_tokens \
    # --learning_rate 2e-5 --max_seq_length $re_max_len --context_window $re_cw \
    # --num_epoch $n_epoch --train_batch_size 128 --eval_batch_size 128 --max_patience $re_patience \
    # --task $task --model $MODEL --output_dir $output_dir \
    # --sampling_proportion $sampling_p_trg --sampling_method $sampling_method \
    # --pipeline_task $pipeline_task --entity_output_dir $entity_output_dir \
    # --train_file "${data_dir}${dataset}"/train.json \
    # --binary_classification
    # # --trigger_output_dir $trigger_output_dir \
    # # --use_trigger

    # MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    # entity_output_dir="${output_dir}/EXP_17/entity"
    # re_cw=300
    # re_max_len=512
    # re_lr=1e-5
    # n_epoch=11
    # python run_relation_with_trigger.py --do_train --do_predict_test \
    # --do_lower_case --add_new_tokens \
    # --learning_rate $re_lr --max_seq_length $re_max_len --context_window $re_cw \
    # --num_epoch $n_epoch --train_batch_size 64 --eval_batch_size 64 --max_patience 4 \
    # --task $task --model $MODEL --output_dir $output_dir --sampling_proportion $sampling_p \
    # --pipeline_task $pipeline_task --entity_output_dir $entity_output_dir \
    # --train_file "${data_dir}${dataset}"/train.json \
    # # --trigger_output_dir $trigger_output_dir \
    # # --use_trigger

    # # task=pn_reduced_trg_dummy
    # python run_entity_trigger.py --do_train --do_predict_test \
    # --max_span_length_entity $max_span_len_ent --max_span_length_trigger $max_span_len_trg \
    # --learning_rate $ner_lr --task_learning_rate 5e-4 --context_window $ner_cw \
    # --train_batch_size 64  --eval_batch_size 64 --num_epoch 84 \
    # --eval_per_epoch 0.25 --max_patience $ner_patience \
    # --task $task --pipeline_task $pipeline_task --model $MODEL \
    # --data_dir "${data_dir}${dataset}" --output_dir $output_dir \
    # # --extract_trigger
    # # --untyped_trigger

    # entity_output_dir="${output_dir}/EXP_38/entity"
    # entity_output_test_dir="${output_dir}/EXP_177/entity"
    # relation_output_dir="${output_dir}/EXP_178/relation"
    # re_cw=100
    # re_max_len=300
    # re_lr=5e-5
    # n_epoch=10
    # python run_relation_with_trigger.py --do_predict_test \
    # --do_lower_case --add_new_tokens \
    # --learning_rate $re_lr --max_seq_length $re_max_len --context_window $re_cw \
    # --num_epoch $n_epoch --train_batch_size 128 --eval_batch_size 128 --max_patience 4 \
    # --task $task --model $MODEL --output_dir $output_dir --sampling_proportion $sampling_p \
    # --pipeline_task $pipeline_task --entity_output_dir $entity_output_dir \
    # --train_file "${data_dir}${dataset}"/train.json \
    # --relation_output_dir $relation_output_dir \
    # --entity_output_test_dir $entity_output_test_dir \
    # # --trigger_output_dir $trigger_output_dir \
    # # --use_trigger


    # pipeline_task="entity"
    # # task=pn_reduced_trg_dummy
    # python run_entity_trigger.py --do_train --do_predict_test \
    # --max_span_length_entity $max_span_len_ent --max_span_length_trigger $max_span_len_trg \
    # --learning_rate $ner_lr --task_learning_rate $task_lr --context_window $ner_cw \
    # --train_batch_size 256  --eval_batch_size 256 --num_epoch 200 \
    # --eval_per_epoch 0.25 --max_patience $ner_patience \
    # --task $task --pipeline_task $pipeline_task --model $MODEL \
    # --data_dir "${data_dir}${dataset}" --output_dir $output_dir \
    # --extract_trigger
    # # --untyped_trigger

    # entity_output_dir="${output_dir}/EXP_151/entity"
    # entity_output_test_dir="${output_dir}/EXP_179/entity"
    # triplet_output_dir="${output_dir}/EXP_180/triplet"

    # re_cw=100
    # re_max_len=300
    # re_lr=2e-5
    # n_epoch=11
    # sampling_p_trg=0.0
    # sampling_method='random'
    # python run_triplet_classification.py --do_predict_test \
    # --do_lower_case --add_new_tokens \
    # --learning_rate $re_lr --max_seq_length $re_max_len --context_window $re_cw \
    # --num_epoch $n_epoch --train_batch_size 128 --eval_batch_size 128 --max_patience $re_patience \
    # --task $task --model $MODEL --output_dir $output_dir \
    # --sampling_proportion $sampling_p_trg --sampling_method $sampling_method \
    # --pipeline_task $pipeline_task --entity_output_dir $entity_output_dir \
    # --entity_output_test_dir $entity_output_test_dir \
    # --train_file "${data_dir}${dataset}"/train.json \
    # --binary_classification \
    # --triplet_output_dir $triplet_output_dir
    # # --trigger_output_dir $trigger_output_dir \
    # # --use_trigger

    # pipeline_task="triplet"
    # entity_output_dir="${output_dir}/EXP_168/entity"
    # entity_output_test_dir="${output_dir}/EXP_182/entity"

    # sampling_p_trg=0.0
    # sampling_method='random'
    # python run_triplet_classification.py --do_train --do_predict_test \
    # --do_lower_case --add_new_tokens \
    # --learning_rate 2e-5 --max_seq_length $re_max_len --context_window $re_cw \
    # --num_epoch 17 --train_batch_size 64 --eval_batch_size 64 --max_patience $re_patience \
    # --task $task --model $MODEL --output_dir $output_dir \
    # --sampling_proportion $sampling_p_trg --sampling_method $sampling_method \
    # --pipeline_task $pipeline_task --entity_output_dir $entity_output_dir \
    # --entity_output_test_dir $entity_output_test_dir \
    # --train_file "${data_dir}${dataset}"/train.json
