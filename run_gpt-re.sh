#!/bin/bash

ROOT=$PWD

# User-specific dir is needed
source /jet/home/ghong1/miniconda3/bin/activate pure
echo "Activated pure"

python run_relation_gpt.py \
    --data_dir ./data/pernut/ner_reduced_v6.1_trg_abs \
    --prompt_file desc_guid_1shot_021424 \
    --few_shot --retrieval_method knn --top_k 1 \
    --min_count 3 --gpt_model chatgpt \
    --temperature 0.0 --max_tokens 4096 --seed 42 \
    --retrieval_model princeton-nlp/sup-simcse-roberta-base \
    --do_test \
    # --api_key \
    # --sampling --num_debug_samples 3 --num_debug_null 1 \
    # --ckpt_dir "./icl/output/prompt_nli_desc_guid_model_chatgpt_k_0/ckpt/result_150.pkl"
