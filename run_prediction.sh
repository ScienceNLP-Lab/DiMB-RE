#!/bin/bash

ROOT=$PWD

source /jet/home/ghong1/miniconda3/bin/activate pure
echo "Activated pure"

task=pn_reduced_trg
data_dir=./data/pilot

python predict.py \
    --task $task --extract_trigger --binary_classification \
    --data_dir $data_dir --eval_batch_size 64