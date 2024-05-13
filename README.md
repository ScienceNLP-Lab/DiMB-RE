# DiMB-RE: Mining the Scientific Literature for Diet-Microbiome Associations

This repository contains (PyTorch) code, dataset, and fine-tuned models for DiMB-RE (**Di**et-**M**icro**B**iome dataset for **R**elation **E**xtraction).

## Quick links
- [DiMB-RE: Mining the Scientific Literature for Diet-Microbiome Associations](#dimb-re-mining-the-scientific-literature-for-diet-microbiome-associations)
  - [Quick links](#quick-links)
  - [Overview](#overview)
  - [1. Setup](#1-setup)
    - [Install dependencies](#install-dependencies)
  - [2. Reproduce the Training and Inference for Pipeline RE system](#2-reproduce-the-training-and-inference-for-pipeline-re-system)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Details for Entity and Trigger Extraction Model](#4-details-for-entity-and-trigger-extraction-model)
  - [5. Details for RE Model (Under construction)](#5-details-for-re-model)
  - [6. Details for Factuality Detection Model (Under construction)](#6-details-for-factuality-detection-model)
  - [7. Fine-tuned Models](#7-fine-tuned-models)

## Overview
![](./figs/annotation-example-new-wb.png)
In this work, we annotate new benchmark corpus for entity and relation extraction, as well as factuality detection with diet-microbiome related entities. Our contributions are as follow:

1. We present the first diverse, multi-layered, publicly available corpus that focuses on diet and microbiome interactions in the scientific literature.
2. We present comprehensive NLP experiments based on state-of-the-art pretrained language models to establish baselines on this corpus.
3. We present a detailed error analysis of NLP results to identify challenges and improvement areas.

<!-- Please find more details of this work in our [paper](https://arxiv.org/pdf/2010.12812.pdf). -->

## 1. Setup

### Install dependencies
Please install all the dependency packages using the following command lines:
```bash
conda create -n DiMB-RE python=3.8
conda activate DiMB-RE
conda install --file requirements.txt
```
or
```bash
conda create -n DiMB-RE python=3.8
conda activate DiMB-RE
conda install pip
pip install -r requirements.txt
```

*Note*: We employed and modified the existing codes from [PURE](https://github.com/princeton-nlp/PURE) as a baseline, while employing the preprocessing scripts from [DeepEventMine](https://github.com/aistairc/DeepEventMine/tree/master/scripts).


## 2. Reproduce the Training and Inference for Pipeline RE system
<!-- ## Quick Start -->

<!-- For simple reproducibility check, you can run this [Colab Notebook](https://colab.research.google.com/drive/1abCEYlFOCmu7yO7TQQeHOwVPCDX8H4Rs?usp=sharing) which is to train end-to-end system from NER-RE-Factuality Detection. -->

The final end-to-end result would approximate to the following scores, which are the best performance of PURE-based RE models from our paper.

```plaintext
NER - P: 0.773, R: 0.745, F1: 0.760
NER Relaxed - P: 0.848, R: 0.795, F1: 0.820
TRG - P: 0.708, R: 0.628, F1: 0.666
TRG Relaxed - P: 0.757, R: 0.671, F1: 0.711
REL Relaxed - P: 0.460, R: 0.377, F1: 0.414
REL Strict - P: 0.417, R: 0.341, F1: 0.375
REL Relaxed+Factuality - P: 0.446, R: 0.365, F1: 0.401
REL Strict+Factuality - P: 0.402, R: 0.329, F1: 0.362

```

If you want to run check reproducibility in your own environment, first you need to follow the instructions in [1. Setup](#1-setup). And then, all you need to do is to run the bash command below. In the shell script, you can check all the arguments for training and making inference for each different model.

```bash
bash check_reproducibility_train.sh

```

## 3. Data Preprocessing
We preprocessed our raw dataset to fit into the input format of PURE, the SpanNER model we are based on. We uploaded the processed dataset for training PURE-based model, so we recommend you to use the processed input files in the `./data/DiMB-RE` folder. 

We also put our raw dataset which are formatted in BRAT style. If you want to check or modify the preprocessing code, please refer to the notebook files in the `./preprocess` directory. 

### Input data format
We follow the protocol of the original PURE paper to construct the input: each line of the input file contains one document.

```bash
{
  # PMID (please make sure doc_key can be used to identify a certain document)
  "doc_key": "34143954",

  # sentences in the document, each sentence is a list of tokens
  "sentences": [
    [...],
    [...],
    ["Here", "we", "show", "that", "a", "lack", "of", "bifidobacteria", ...],
    ...
  ],

  # entities (boundaries and entity type) in each sentence
  "ner": [
    [...],
    [...],
    [[78, 78, "Microorganism"], [88, 90, "Nutrient"], ...], # start and end indices are document-level, token-based spans 
    ...,
  ],

  # triggers (boundaries and entity type) in each sentence
  "triggers": [
    [...],
    [...],
    [[100, 101, "NEG_ASSOCIATED_WITH"]], # Same with the format of ner values.
    ...,
  ],

  # relations (spans of entity pair (in the order of Agent -> Theme), relation type, and factuality value) in each sentence
  "relations": [
    [...],
    [...],
    [[78, 78, 102, 103, "NEG_ASSOCIATED_WITH", "Factual"], [78, 78, 105, 106, "NEG_ASSOCIATED_WITH", "Factual"]], 
    ...
  ],

  # triplets (spans of entity pair (in the order of Agent -> Theme) and trigger mention, relation type) in each sentence
  "triplets": [
    [...],
    [...],
    [[78, 78, 102, 103, 100, 101, "NEG_ASSOCIATED_WITH"], [78, 78, 105, 106, 100, 101, "NEG_ASSOCIATED_WITH"]], # We require the information of trigger mention spans to pass them as inputs for triplet classification
    ...
  ]
}
```

## 4. Details for Entity and Trigger Extraction Model

### Train/evaluate

You can use `run_entity_trigger.py` with `--do_train` to train an entity model and with `--do_eval` to evaluate an entity model.

A training command template is as follow:
```bash
python run_entity.py \
    --do_train --do_eval [--eval_test] \
    --learning_rate=1e-5 --task_learning_rate=5e-4 \
    --train_batch_size=16 \
    --context_window {0 | 100 | 300} \
    --task {ace05 | ace04 | scierc} \
    --data_dir {directory of preprocessed dataset} \
    --model {bert-base-uncased | albert-xxlarge-v1 | allenai/scibert_scivocab_uncased} \
    --output_dir {directory of output files}
```
Arguments:
* `--learning_rate`: the learning rate for BERT encoder parameters.
* `--task_learning_rate`: the learning rate for task-specific parameters, i.e., the classifier head after the encoder.
* `--context_window`: the context window size used in the model. `0` means using no contexts. In our cross-sentence entity experiments, we use `--context_window 300` for BERT models and SciBERT models and use `--context_window 100` for ALBERT models.
* `--model`: the base transformer model. We use `bert-base-uncased` and `albert-xxlarge-v1` for ACE04/ACE05 and use `allenai/scibert_scivocab_uncased` for SciERC.
* `--eval_test`: whether evaluate on the test set or not.

The predictions of the entity model will be saved as a file (`ent_pred_dev.json`) in the `output_dir` directory. If you set `--eval_test`, the predictions (`ent_pred_test.json`) are on the test set. The prediction file of the entity model will be the input file of the relation model.

<!-- ## 3. Details for Training Model (Under construction): -->
<!-- ### Input data format for the relation model
The input data format of the relation model is almost the same as that of the entity model, except that there is one more filed `."predicted_ner"` to store the predictions of the entity model.
```bash
{
  "doc_key": "CNN_ENG_20030306_083604.6",
  "sentences": [...],
  "ner": [...],
  "relations": [...],
  "predicted_ner": [
    [...],
    [...],
    [[26, 26, "LOC"], [14, 15, "PER"], ...],
    ...
  ]
}
```

### Train/evaluate the relation model (Under construction):
You can use `run_relation.py` with `--do_train` to train a relation model and with `--do_eval` to evaluate a relation model. A trianing command template is as follow:
```bash
python run_relation.py \
  --task {ace05 | ace04 | scierc} \
  --do_train --train_file {path to the training json file of the dataset} \
  --do_eval [--eval_test] [--eval_with_gold] \
  --model {bert-base-uncased | albert-xxlarge-v1 | allenai/scibert_scivocab_uncased} \
  --do_lower_case \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --context_window {0 | 100} \
  --max_seq_length {128 | 228} \
  --entity_output_dir {path to output files of the entity model} \
  --output_dir {directory of output files}
```
Arguments:
* `--eval_with_gold`: whether evaluate the model with the gold entities provided.
* `--entity_output_dir`: the output directory of the entity model. The prediction files (`ent_pred_dev.json` or `ent_pred_test.json`) of the entity model should be in this directory.

The prediction results will be stored in the file `predictions.json` in the folder `output_dir`, and the format will be almost the same with the output file from the entity model, except that there is one more field `"predicted_relations"` for each document.

You can run the evaluation script to output the end-to-end performance  (`Ent`, `Rel`, and `Rel+`) of the predictions.
```bash
python run_eval.py --prediction_file {path to output_dir}/predictions.json
```

*Note*: Training/evaluation performance might be slightly different from the reported numbers in the paper, depending on the number of GPUs, batch size, and so on. -->

<!-- ### Approximation relation model
You can use the following command to train an approximation model.
```bash
python run_relation_approx.py \
 --task {ace05 | ace04 | scierc} \
 --do_train --train_file {path to the training json file of the dataset} \
 --do_eval [--eval_with_gold] \
 --model {bert-base-uncased | allenai/scibert_scivocab_uncased} \
 --do_lower_case \
 --train_batch_size 32 \
 --eval_batch_size 32 \
 --learning_rate 2e-5 \
 --num_train_epochs 10 \
 --context_window {0 | 100} \
 --max_seq_length {128 | 228} \
 --entity_output_dir {path to output files of the entity model} \
 --output_dir {directory of output files}
```

Once you have a trained approximation model, you can enable efficient batch computation during inference with `--batch_computation`:
```bash
python run_relation_approx.py \
 --task {ace05 | ace04 | scierc} \
 --do_eval [--eval_test] [--eval_with_gold] \
 --model {bert-base-uncased | allenai/scibert_scivocab_uncased} \
 --do_lower_case \
 --eval_batch_size 32 \
 --context_window {0 | 100} \
 --max_seq_length 250 \
 --entity_output_dir {path to output files of the entity model} \
 --output_dir {directory of output files} \
 --batch_computation
```
*Note*: the current code does not support approximation models based on ALBERT. -->

## Fine-tuned Models
We release our fine-tuned relation models, and factuality detection models for our dataset in HuggingFace with the model name of gbhong/BiomedBERT-fulltext_finetuned_DiMB-RE and gbhong/BiomedBERT-fulltext_finetuned_DiMB-RE_FD.

<!-- ## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Zexuan Zhong `(zzhong@cs.princeton.edu)`. If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker! -->

## Citation
If you use our code in your research, please cite our work:
```bibtex
@inproceedings{zhong2021frustratingly,
   title={A Frustratingly Easy Approach for Entity and Relation Extraction},
   author={Zhong, Zexuan and Chen, Danqi},
   booktitle={North American Association for Computational Linguistics (NAACL)},
   year={2021}
}
```
