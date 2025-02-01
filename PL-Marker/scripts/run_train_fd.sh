#!/bin/bash

source /jet/home/ghong1/miniconda3/bin/activate PL-Marker
echo "Activated PL-Marker"

# --use_ner_results: use the original entity type predicted by NER models
# DiMB-RE
# mkdir dimb-re_models

# split=dev
split=test

### SET YOUR OWN OUTPUT DIR ###
# output_dir=/jet/home/ghong1/ocean_cis230030p/ghong1/PL-Marker/no-result
output_dir=./output

for seed in 0 1 2 3 4; do 
    python3 run_fd_trg_inserted.py \
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
    --save_steps 255 \
    --eval_logsoftmax \
    --seed $seed \
    --output_dir $output_dir/FD/dimb-re_models_biomedbert_trg_inserted_${split}-${seed} \
    --test_file $output_dir/RE/dimb-re_models_biomedbert_trg_inserted_${split}-${seed}/rel_pred_${split}_goldner.json \
    --overwrite_output_dir \
    --use_ner_results \
    --use_trigger \
    --use_typemarker \
    --lminit \
    --eval_all_checkpoints \
    --no_sym
    # --do_train
    # --evaluate_during_training \
    # --do_eval \
    # --no_test \

    # --test_file $output_dir/RE/dimb-re_models_biomedbert_trg_inserted_${split}-${seed}/rel_pred_${split}_goldner.json \
    
    # --dev_file $output_dir/RE/dimb-re_models_biomedbert_trg_inserted_dev-$seed/rel_pred_dev.json \
    # --test_file $output_dir/RE/dimb-re_models_biomedbert_trg_${split}-${seed}/ent_pred_${split}.json \
    # --eval_unidirect \
done;


# GPU_ID=0

# # For ALBERT-xxlarge, change learning_rate from 2e-5 to 1e-5

# # ACE05
# mkdir ace05re_models
# for seed in 42 43 44 45 46; do 
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_re.py  --model_type bertsub  \
#     --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
#     --data_dir ace05  \
#     --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
#     --max_seq_length 256  --max_pair_length 32  --save_steps 5000  \
#     --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
#     --fp16  --seed $seed    \
#     --test_file ace05ner_models/PL-Marker-ace05-bert-$seed/ent_pred_test.json  \
#     --output_dir ace05re_models/ace05re-bert-$seed  --overwrite_output_dir
# done;
# # Average the scores
# python3 sumup.py ace05re ace05re-bert


# # SciERC,  --use_ner_results: use the original entity type predicted by NER models
# mkdir scire_models
# for seed in 42 43 44 45 46; do 
# CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_re.py  --model_type bertsub  \
#     --model_name_or_path  bert_models/scibert-uncased  --do_lower_case  \
#     --data_dir scierc  \
#     --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
#     --max_seq_length 256  --max_pair_length 16  --save_steps 2500  \
#     --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
#     --fp16  --seed $seed      \
#     --test_file sciner_models/PL-Marker-scierc-scibert-$seed/ent_pred_test.json  \
#     --use_ner_results \
#     --output_dir scire_models/scire-scibert-$seed  --overwrite_output_dir
# done;
# # Average the scores
# python3 sumup.py scire scire-scibert
