# DiMB-RE: Mining the Scientific Literature for Diet-Microbiome Associations

This repository contains (PyTorch) code, dataset, and fine-tuned models for DiMB-RE (**Di**et-**M**icro**B**iome dataset for **R**elation **E**xtraction).

**Note**: (On Oct 1st, 2024) Some functions in the repository are still under construction and will be updated soon. Stay tuned for further improvements and updates.

## Quick links
- [DiMB-RE: Mining the Scientific Literature for Diet-Microbiome Associations](#dimb-re-mining-the-scientific-literature-for-diet-microbiome-associations)
  - [Quick links](#quick-links)
  - [Overview](#overview)
  - [1. Setup](#1-setup)
    - [Install dependencies](#install-dependencies)
  - [2. Reproduce the Training and Inference for Pipeline RE system](#2-reproduce-the-training-and-inference-for-pipeline-re-system)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Details for Pipeline Model](#4-details-for-pipeline-model)
  - [5. Fine-tuned Models](#5-fine-tuned-models)

## Overview
![](./figs/annotation-example-new-wb.png)
DiMB-RE is a corpus of 165 nutrition and microbiome-related publications, and we validate its usefulness with state-of-the-art pretrained language models. Specifically, we make the following contributions:

1. We annotated titles and abstracts of 165 publications with 15 entity types and 13 relation types that hold between them. To our knowledge, DiMB-RE is the largest and most diverse corpus focusing on this domain in terms of the number of entities and relations it contains.
2. In addition to titles and abstracts, we annotated Results sections of 30 articles (out of 165) to assess the impact of the information from full text.
3. To ground and contextualize relations, we annotated relation triggers and certainty information, which were previously included only in the biological event extraction corpora.
4. We normalized entity mentions to standard database identifiers (e.g., MeSH, CheBI, FoodOn) to allow aggregation for further study.
5. We trained and evaluated NER and RE models based on the state-of-the-art pretrained language models to establish robust baselines for this corpus. 

Further details regarding this study are available in our [paper](https://arxiv.org/pdf/2409.19581.pdf).

## 1. Setup

### Install dependencies
Please install all the dependency packages using the following command lines to replicate training process, or just use the fine-tuned model:

<!-- ```bash
conda create -n DiMB-RE python=3.8
conda activate DiMB-RE
conda install --file requirements.txt
```
or -->

```bash
conda create -n DiMB-RE python=3.8
conda activate DiMB-RE
conda install pip
pip install -r requirements.txt
```

*Note*: We employed and modified the existing codes from [PURE](https://github.com/princeton-nlp/PURE) as a baseline, while employing the preprocessing scripts from [DeepEventMine](https://github.com/aistairc/DeepEventMine/tree/master/scripts).


## 2. Replicate the Training process for End-to-end RE system

### Training pipeline
To train our end-to-end pipeline (NER-RE-FD), you can simply run the shell script like below:

```bash
bash train.sh
```

We currently use optimal hyperparameter set specific to our dataset, DiMB-RE. If you plan to train on a different dataset, please adjust the hyperparameters accordingly. You may also modify the scripts if you’re training only part of the pipeline. Also, before running the model we strongly recommend to assign your own specific directories to save models and prediction files in the shell script.

The final end-to-end results for DiMB-RE test set would approximate to the following scores, which are reported as our main result in the paper. Confidence intervals for each P/R/F1 in our original paper are not included for brevity.

```plaintext
NER Strict - P: 0.777, R: 0.745, F1: 0.760
NER Relaxed - P: 0.852, R: 0.788, F1: 0.819
TRG Strict - P: 0.691, R: 0.631, F1: 0.660
TRG Relaxed - P: 0.742, R: 0.678, F1: 0.708
REL Strict - P: 0.416, R: 0.336, F1: 0.371
REL Relaxed - P: 0.448, R: 0.370, F1: 0.409
REL Strict+Factuality - P: 0.399, R: 0.322, F1: 0.356
REL Relaxed+Factuality - P: 0.440, R: 0.355, F1: 0.393
```

### Using fine-tuned models (for reproducibility)
If you want to check whether our result for test set is reproducible for our main model, just run the command line below and check the final result:

```bash
bash check_reproducibility.sh
```

### End-to-end prediction with unlabeled dataset
If you want to predict relation with our main model, please run the command line below:

```bash
bash predict.sh
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

## 4. Details for Pipeline Model

The predictions of the entity model will be saved as a file (`ent_pred_dev.json`) in the `./output/entity` directory if you set `--do_predict_dev`. The predictions (`ent_pred_test.json`) would be generated if you set `--do_predict_test`. The prediction file of the entity model will be the input file of the relation extraction model. This goes same with the relation extraction model: `trg_pred_{dev|test}.json` file would be saved after running the model, and those files will be inputs for factuality detection model, which is our last step for the pipeline.

For more details about the arguments in each model, please refer to the `run_entity_trigger.py` for entity and trigger extraction, `run_triplet_classification.py` for relation extraction with Typed trigger, and `run_certainty_detection.py` for factuality detection model. 

And for evaluation, we recommend you test your prediction file with `run_eval.py` or `run_evals.sh` in order to consider the directionality of predicted relations.

<!-- ## 4. Details for Entity and Trigger Extraction Model -->

<!-- Below is the python command to run training/evaluation with different kinds of arguments:

```bash
python run_entity_trigger.py \
  --task pn_reduced_trg --pipeline_task entity \
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
```

Arguments:
* `--task`: Related with constant variables (task-specific labels). Check `./shared/const.py` for more details.
* `--pipeline_task`: Specify what kind of task to perform among the three pipeline tasks.
* `--do_train`, `--do_eval`: Wge
* `--learning_rate`: the learning rate for BERT encoder parameters.
* `--learning_rate`: the learning rate for BERT encoder parameters.
* `--task_learning_rate`: the learning rate for task-specific parameters, i.e., the classifier head after the encoder.
* `--context_window`: the context window size used in the model. `0` means using no contexts. In our cross-sentence entity experiments, we use `--context_window 300` for BERT models and SciBERT models and use `--context_window 100` for ALBERT models.
* `--model`: the base transformer model. We use `bert-base-uncased` and `albert-xxlarge-v1` for ACE04/ACE05 and use `allenai/scibert_scivocab_uncased` for SciERC.
* `--eval_test`: whether evaluate on the test set or not. -->

<!-- The predictions of the entity model will be saved as a file (`ent_pred_dev.json`) in the `./output/entity` directory if you set `--do_predict_dev`. The predictions (`ent_pred_test.json`) would be generated if you set `--do_predict_test`. The prediction file of the entity model will be the input file of the relation extraction model.  -->

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

## 5. Fine-tuned Models
We released our best fine-tuned [NER model](https://huggingface.co/gbhong/BiomedBERT-fulltext_finetuned_DiMB-RE_NER), [Relation Extraction model](https://huggingface.co/gbhong/BiomedBERT-fulltext_finetuned_DiMB-RE_RE), and [Factuality Detection model](https://huggingface.co/gbhong/BiomedBERT-fulltext_finetuned_DiMB-RE_FD) trained for our DiMB-RE dataset in HuggingFace.

<!-- ## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Zexuan Zhong `(zzhong@cs.princeton.edu)`. If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker! -->

## Citation
If you utilize our code in your research, please reference our work:
```bibtex
@misc{hong2024dimbreminingscientificliterature,
      title={DiMB-RE: Mining the Scientific Literature for Diet-Microbiome Associations}, 
      author={Gibong Hong and Veronica Hindle and Nadine M. Veasley and Hannah D. Holscher and Halil Kilicoglu},
      year={2024},
      eprint={2409.19581},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.19581}, 
}
```
