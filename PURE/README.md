This repository contains PyTorch code to implement PURE for DiMB-RE (**Di**et-**M**icro**B**iome dataset for **R**elation **E**xtraction) dataset.

**Note**: (On Feb 1st, 2025) Some functions in the repository are still under construction and will be updated soon. Stay tuned for further improvements and updates.

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

## 3. Details for Pipeline Model

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

## 4. Fine-tuned Models
We released our best fine-tuned [NER model](https://huggingface.co/gbhong/BiomedBERT-fulltext_finetuned_DiMB-RE_NER), [Relation Extraction model](https://huggingface.co/gbhong/BiomedBERT-fulltext_finetuned_DiMB-RE_RE), and [Factuality Detection model](https://huggingface.co/gbhong/BiomedBERT-fulltext_finetuned_DiMB-RE_FD) trained for our DiMB-RE dataset in HuggingFace.


