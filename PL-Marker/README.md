This repository contains PyTorch code to implement PL-Marker for DiMB-RE (**Di**et-**M**icro**B**iome dataset for **R**elation **E**xtraction) dataset.

**Note**: (On Feb 1st, 2025) Some functions in the repository are still under construction and will be updated soon. Stay tuned for further improvements and updates.

## 1. Setup

### Install dependencies
Please install all the dependency packages using the following command lines to replicate training process, or just use the fine-tuned model:

```bash
conda create -n *your-venv-name* python=3.8
conda activate *your-venv-name*
conda install pip

pip install -r requirements.txt
pip install --editable ./transformers
```

*Note*: We employed and modified the existing codes from [PL-Marker](https://github.com/thunlp/PL-Marker) as a baseline.

## 2. Replicate the Training process for End-to-end RE system

Note that some parts of the code may need to be modified to match your directory or specific setup for data.

### Training NER and Trigger Extraction model

```bash
bash ./scripts/run_train_ner_PLMarker.sh
```

Check the `run_acener_trg_modified.py` for hyperparameter tuning, code reference, or modifications.
Most of our default parameters follow the original settings suggested by PL-Marker paper.

### Training Relation Extraction model

```bash
bash ./scripts/run_train_re.sh
```

Check the `run_re_trg_inserted.py` for hyperparameter tuning, code reference, or modifications.
Most of our default parameters follow the original settings suggested by PL-Marker paper.

### Training Factuality Detection model

```bash
bash ./scripts/run_train_fd.sh
```

Check the `run_fd_trg_inserted.py` for hyperparameter tuning, code reference, or modifications.


## 3. Details for Pipeline Model

The predictions of the entity model will be saved as a file (`ent_pred_dev.json`) in the `./output` directory if you set `--do_eval`. The predictions (`ent_pred_test.json`) would be generated if you set `--do_test`. The prediction file of the entity model will be the input file of the relation extraction model. This goes same with the relation extraction model: `trg_pred_{dev|test}.json` file would be saved after running the model, and those files will be inputs for factuality detection model, which is our last step for the pipeline.

And for evaluation, we recommend you test your prediction file with `run_eval.py` or `run_evals.sh` in order to consider the directionality of predicted relations.