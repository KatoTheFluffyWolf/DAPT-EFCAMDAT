# README

## Overview
This zip contains the code used in our study on domain-adaptive continued pretraining for automated essay scoring (AES). The project investigates whether continued pretraining on learner writing improves AES performance on English proficiency test datasets, and whether it helps reduce performance degradation under cross-dataset transfer.

The code is organised around three stages:
1. **Dataset preprocessing** for IELTS, FCE, and EFCAMDAT.
2. **Continued pretraining (DAPT)** of BERT on learner writing from EFCAMDAT.
3. **Downstream AES experiments**, including in-domain fine-tuning and cross-dataset few-shot transfer.

## Important note on data availability
The raw datasets are **not included** in this zip. This is intentional because some datasets used in the study have redistribution restrictions. To reproduce the experiments, users should obtain the datasets from their original sources and then run the preprocessing scripts provided here.

---

## Folder structure

```text
code_submission/
├── cross-dataset-transfer.py
├── dapt_bert_efcamdat.py
├── finetuning_FCE.py
├── finetuning_IELTS.py
├── requirements.txt
├── preprocessing/
│   ├── EFCAMDAT_split.py
│   ├── FCE_preprocessing.py
│   └── IELTS_preprocessing.py
├── results/
│   ├── Ablation.xlsx
│   ├── Baseline Comparison.xlsx
│   └── Cross-dataset Transfer.xlsx
└── runs/
    ├── Ablation/
    │   ├── fce_A1A2/
    │   ├── fce_B1B2C1/
    │   ├── ielts_A1A2/
    │   └── ielts_B1B2C1/
    ├── Baseline Comparison/
    │   ├── fce_bert/
    │   ├── fce_dapt_bert/
    │   ├── fce_distilbert/
    │   ├── fce_roberta/
    │   ├── ielts_bert/
    │   ├── ielts_dapt_bert/
    │   ├── ielts_distilbert/
    │   └── ielts_roberta/
    └── Cross-dataset Transfer/
        ├── FCE to IELTS/
        │   ├── bert/
        │   └── dapt-bert/
        └── IELTS to FCE/
            ├── bert/
            └── dapt-bert/
```

**Note:** Each specific run in the "runs" folder contains two files: test-predictions and test-metrics. Test metrics contains the evaluation metrics reported in the paper, while test predictions comprises the raw prediction on the test set that was used to compute the metrics.

---

## File descriptions

### 1. `cross-dataset-transfer.py`
This script runs the **cross-dataset transfer experiment**. It loads a checkpoint that was already fine-tuned on a **source dataset**, resets the regression head, samples a **few-shot subset** from the **target training set**, fine-tunes on that subset, and evaluates on the target development and test sets. It reports RMSE, Pearson, Spearman, and optionally QWK depending on the target score set, and saves predictions and metrics as output files.

Use this script for experiments such as:
- source = IELTS, target = FCE
- source = FCE, target = IELTS

### 2. `dapt_bert_efcamdat.py`
This script performs **domain-adaptive pretraining (DAPT)** of BERT using a masked language modeling (MLM) objective on the cleaned EFCAMDAT corpus. It follows the setup described in this study by Stearns et al. (2024)

Bernardo Stearns, Nicolas Ballier, Thomas Gaillat, Andrew Simpkin, and John P. McCrae. 2024. Evaluating the Generalisation of an Artificial Learner. In Proceedings of the 13th Workshop on Natural Language Processing for Computer Assisted Language Learning, pages 199–208, Rennes, France. LiU Electronic Press.

The pretraining setup includes:
- BERT base as the starting checkpoint
- maximum sequence length filtering
- 15% masking probability
- train/eval split
- optional deduplication
- optional token-budget capping for ablation settings

The script saves intermediate checkpoints and the final domain-adapted model. This is the script used to produce the domain-adapted checkpoints before downstream AES fine-tuning.

### 3. `finetuning_FCE.py`
This script fine-tunes a transformer model for **AES regression on the FCE dataset**. It loads the processed train/dev/test CSV files, tokenizes the essays, fine-tunes the model, evaluates on development and test sets, and saves predictions and metrics.

It uses the FCE score mapping and computes:
- RMSE
- Pearson correlation
- Spearman correlation
- Quadratic Weighted Kappa (QWK)

This script can be used with either:
- a general-domain base checkpoint such as `bert-base-uncased`, or
- a domain-adapted checkpoint produced by `dapt_bert_efcamdat.py`

### 4. `finetuning_IELTS.py`
This script fine-tunes a transformer model for **AES regression on the IELTS dataset**. It is the IELTS counterpart to `finetuning_FCE.py`. It loads processed train/dev/test CSV files, fine-tunes the model, evaluates performance, and saves predictions and metrics.

It computes:
- RMSE
- Pearson correlation
- Spearman correlation
- Quadratic Weighted Kappa (QWK)

Like the FCE script, it supports both general-domain checkpoints and DAPT checkpoints.

### 5. `requirements.txt`
This file lists the Python package versions used in the environment for the experiments. It is included to help reproduce the software setup as closely as possible.

---

## Folder descriptions

### `preprocessing/`
This folder contains the dataset preparation scripts used before training.

#### `preprocessing/EFCAMDAT_split.py`
This script merges the two excel files of the EFCAMDAT corpus, filters rows with valid text and CEFR labels, and splits the corpus by proficiency level. In its current form, it creates:
- `A1_A2.xlsx`
- `B1_B2.xlsx`
- `C1.xlsx`
- `split_summary.csv`

It also counts BERT tokens for each split, which is useful for reporting corpus size and for constructing token-controlled ablation studies.

#### `preprocessing/FCE_preprocessing.py`
This script parses the raw **FCE XML files**, reconstructs the learner text from the error-annotated XML, maps the original FCE holistic score labels to the numerical scale used in the study, removes duplicates, and creates **train/dev/test CSV splits**. It uses **group-based splitting by script ID** to reduce leakage between splits.

#### `preprocessing/IELTS_preprocessing.py`
This script preprocesses the raw IELTS CSV data. It removes missing values and duplicates, drops unused columns, filters invalid scores, removes ultra-short essays, cleans non-English characters, filters to the selected task type, and then creates **train/dev/test CSV splits**.

This script is intended to produce the processed IELTS files used by `finetuning_IELTS.py`.

### `runs/`
This folder contains the saved **test set outputs** for the experiments reported in the paper. Each run directory contains two files:
- `test_metrics.json`: the evaluation metrics for that run
- `test_predictions.csv`: the raw test predictions used to compute the reported results

The folder is organised by experiment type:
- `runs/Baseline Comparison/` for in-domain baseline comparison runs on IELTS and FCE.
- `runs/Cross-dataset Transfer/` for few-shot cross-dataset transfer in both directions (IELTS→FCE and FCE→IELTS).
- `runs/Ablation/` for the EFCAMDAT subset ablation runs (A1–A2 only, B1–B2–C1 only) on IELTS and FCE.

### `results/`
This folder contains spreadsheet summaries of the main reported results:
- `Baseline Comparison.xlsx`
- `Cross-dataset Transfer.xlsx`
- `Ablation.xlsx`

These files are included for convenience and present the reported results in table form.

---

## Typical workflow
A typical reproduction workflow is:

1. Obtain the raw datasets from their original sources.
2. Run the preprocessing scripts in `preprocessing/` to prepare train/dev/test files.
3. Run `dapt_bert_efcamdat.py` to create domain-adapted BERT checkpoints on EFCAMDAT.
4. Run `finetuning_FCE.py` and `finetuning_IELTS.py` to fine-tune models on the FCE and IELTS datasets, respectively. These scripts can be used for both the domain-adapted checkpoint and the baseline models.
5. Run `cross-dataset-transfer.py` for the few-shot cross-dataset transfer experiments.

---

## Reproducibility notes
- Random seeds are set in the training scripts for more stable reproduction.
- The scripts save metrics and predictions to disk so results can be inspected after each run.
- The raw datasets are not redistributed in this package; users must obtain them from the corresponding provider.

For the EFCAMDAT corpus, please refer to: https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html
For the CLC FCE dataset: https://ilexir.co.uk/datasets/index.html
For the IELTS writings scored essays dataset: https://www.kaggle.com/datasets/mazlumi/ielts-writing-scored-essays-dataset/


---

## Notes for reviewers
This package is intended to make the experimental pipeline transparent and reproducible without redistributing restricted datasets. The code covers the main stages of the study: preprocessing, domain-adaptive pretraining, fine-tuning on downstream datasets for AES, and cross-dataset transfer evaluation.

In addition to the code, the package includes the saved test metrics and prediction files corresponding to the reported experiments under `runs/`, as well as spreadsheet summaries of the main results under `results/`.
