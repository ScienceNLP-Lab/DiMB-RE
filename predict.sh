#!/bin/bash

ROOT=$PWD

source /jet/home/ghong1/miniconda3/bin/activate pure
echo "Activated pure"

task=pn_reduced_trg
# Assign your own dataset dir
data_dir=./data/pilot

python predict_e2e.py \
    --task $task --extract_trigger --binary_classification \
    --data_dir $data_dir --eval_batch_size 64