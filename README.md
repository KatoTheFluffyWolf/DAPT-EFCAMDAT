# DAPT-EFCAMDAT

This repository contains the code, experiment outputs, and result summaries for a study on **learner-domain domain-adaptive continued pretraining (DAPT)** for **automated essay scoring (AES)** on English proficiency test datasets.

The study investigates whether continued pretraining on learner writing from **EFCAMDAT** improves transformer-based AES performance, and whether such adaptation improves cross-test transfer between two English proficiency assessment datasets: **FCE** and **IELTS Task 2**.

## Overview

This repository supports a study on whether **learner-domain domain-adaptive continued pretraining (DAPT)** improves **automated essay scoring (AES)** on English proficiency test datasets.

The study first performs DAPT on three transformer encoders, **BERT**, **RoBERTa**, and **DistilBERT**, using the **EFCAMDAT** learner corpus. The adapted models are then evaluated through two main experiments:

1. **Baseline comparison**  
   The domain-adapted models are fine-tuned and evaluated separately on **FCE** and **IELTS Task 2**, then compared against their corresponding non-adapted base checkpoints.

2. **Cross-dataset transfer**  
   Models are first fine-tuned on one English proficiency test dataset, then adapted to the other dataset under few-shot settings using 50, 100, and 200 target training samples.

Because full-corpus DAPT produced mixed results, the study further investigates whether these outcomes are related to corpus alignment. This is done through:

3. **Vocabulary distribution analysis**  
   Jensen-Shannon divergence is used to compare lexical distributions between EFCAMDAT proficiency subsets and the downstream AES datasets.

4. **Syntactic complexity analysis**  
   NeoSCA is used to compare syntactic complexity profiles across EFCAMDAT, FCE, and IELTS.

5. **Proficiency-based ablation**  
   BERT is further pretrained on selected CEFR-based EFCAMDAT subsets, such as A1–A2, B1–B2, and B2–C1, to test whether better-aligned pretraining data improves downstream AES performance.

Overall, the repository includes code for preprocessing, DAPT, downstream fine-tuning, cross-dataset transfer, alignment analysis, ablation experiments, and saved experiment outputs.

## Repository structure

```text
DAPT-EFCAMDAT/
├── analysis/
│   ├── JSD.py
│   ├── syntactic_complexity.py
│   └── top-k_tokens.py
│
├── preprocessing/
│   ├── EFCAMDAT_split.py
│   ├── FCE_preprocessing.py
│   └── IELTS_preprocessing.py
│
├── results/
│   ├── Ablation.xlsx
│   ├── Analysis.xlsx
│   ├── Baseline Comparison.xlsx
│   ├── Cross-test transferability.xlsx
│   ├── jsd_fce_efcamdat.xlsx
│   └── jsd_ielts_efcamdat.xlsx
│
├── runs/
│   ├── Ablation/
│   ├── Downstream Fine-tuning/
│   └── Transfer/
│
├── cross-dataset-transfer.py
├── dapt_bert_efcamdat.py
├── finetuning_FCE.py
├── finetuning_IELTS.py
├── manuscript.pdf
├── requirements.txt
└── README.md
```

## Main scripts

### `dapt_bert_efcamdat.py`

Runs domain-adaptive continued pretraining on EFCAMDAT using a masked language modelling objective.

The script supports:

- BERT-style masked language modelling
- 15% masking probability
- maximum WordPiece sequence filtering
- train/evaluation split
- optional deduplication
- token-budget capping for controlled ablation experiments
- checkpoint saving and evaluation during training

Example:

```bash
python dapt_bert_efcamdat.py \
  --xlsx path/to/efcamdat.xlsx \
  --text_col text \
  --out outputs/dapt_bert_efcamdat \
  --model bert-base-uncased \
  --max_len 512 \
  --mlm_prob 0.15 \
  --epochs 1 \
  --train_bs 4 \
  --eval_bs 8 \
  --grad_acc 16 \
  --lr 5e-5 \
  --warmup_ratio 0.06 \
  --fp16
```

### `finetuning_FCE.py`

Fine-tunes a transformer encoder for AES regression on the FCE dataset.

The script:

- loads processed FCE train/dev/test files
- normalises FCE scores
- fine-tunes a sequence classification model with a regression head
- computes RMSE, Pearson, Spearman, and Quadratic Weighted Kappa
- saves test metrics and predictions

It can be used with either a base transformer checkpoint or a DAPT checkpoint.

### `finetuning_IELTS.py`

Fine-tunes a transformer encoder for AES regression on the IELTS Task 2 dataset.

The script:

- loads processed IELTS train/dev/test files
- normalises IELTS scores
- fine-tunes a regression model
- computes RMSE, Pearson, Spearman, and Quadratic Weighted Kappa
- saves test metrics and predictions

It can be used with either a base transformer checkpoint or a DAPT checkpoint.

### `cross-dataset-transfer.py`

Runs few-shot cross-dataset transfer experiments.

The script:

- loads a model checkpoint fine-tuned on a source dataset
- samples a small target-domain training subset
- fine-tunes on the target few-shot subset
- evaluates on the target development and test sets
- saves metrics and predictions

This is used for both transfer directions:

- IELTS → FCE
- FCE → IELTS

## Analysis scripts

### `analysis/JSD.py`

Computes Jensen-Shannon divergence between vocabulary distributions.

This is used to examine lexical alignment between EFCAMDAT proficiency subsets and the downstream AES datasets.

### `analysis/top-k_tokens.py`

Extracts the most frequent tokens from selected corpora.

This provides a qualitative view of lexical distribution differences between EFCAMDAT, FCE, and IELTS.

### `analysis/syntactic_complexity.py`

Computes syntactic complexity features for corpus comparison.

This is used to examine whether EFCAMDAT proficiency subsets are syntactically closer to FCE or IELTS.

## Preprocessing scripts

### `preprocessing/EFCAMDAT_split.py`

Processes and splits the EFCAMDAT corpus by CEFR proficiency level.

The resulting proficiency subsets are used for full-corpus DAPT and proficiency-based ablation experiments.

### `preprocessing/FCE_preprocessing.py`

Processes the FCE dataset by extracting learner text, mapping holistic scores to the numerical scale used in the study, removing duplicates, and creating train/dev/test splits.

### `preprocessing/IELTS_preprocessing.py`

Processes the IELTS writing dataset by cleaning essays, filtering unsuitable examples, selecting IELTS Task 2 essays, and creating train/dev/test splits.

## Results folder

The `results/` folder contains spreadsheet summaries of the main reported results and corpus-alignment analyses.

It includes:

- `Baseline Comparison.xlsx`: summary of in-domain AES fine-tuning results.
- `Cross-test transferability.xlsx`: summary of few-shot cross-dataset transfer results.
- `Ablation.xlsx`: summary of proficiency-based DAPT ablation results.
- `Analysis.xlsx`: additional analysis summary tables.
- `jsd_fce_efcamdat.xlsx`: Jensen-Shannon divergence results comparing FCE and EFCAMDAT variants.
- `jsd_ielts_efcamdat.xlsx`: Jensen-Shannon divergence results comparing IELTS and EFCAMDAT variants.

These spreadsheets are included for convenience and present the reported results in table form.

## Experiment outputs

The `runs/` directory contains saved outputs from the experiments reported in the study.

Each run folder typically contains files such as:

```text
test_metrics.json
test_predictions.csv
```

`test_metrics.json` contains the evaluation metrics for the run, while `test_predictions.csv` contains the gold scores and model predictions used to compute the reported results.

### `runs/Downstream Fine-tuning/`

Contains in-domain AES fine-tuning outputs for FCE and IELTS.

These runs correspond to the baseline comparison experiment, where each DAPT model is compared against its corresponding non-adapted base model.

### `runs/Transfer/`

Contains few-shot cross-dataset transfer outputs for:

- IELTS → FCE
- FCE → IELTS

These runs correspond to the cross-dataset transfer experiment.

### `runs/Ablation/`

Contains outputs for proficiency-based DAPT ablation experiments.

The ablation experiments compare DAPT variants trained on different EFCAMDAT proficiency subsets, such as:

- A1–A2
- B1–B2
- B2–C1

The ablation outputs include both downstream fine-tuning and cross-dataset transfer results.

## Installation

Clone the repository:

```bash
git clone https://github.com/KatoTheFluffyWolf/DAPT-EFCAMDAT.git
cd DAPT-EFCAMDAT
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Using a virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data availability

The raw datasets are not included in this repository because some of them have redistribution restrictions.

To reproduce the experiments, users should obtain the datasets from their original sources and then run the preprocessing scripts in `preprocessing/`.

Datasets used in the study include:

- EFCAMDAT learner corpus
- Cambridge Learner Corpus FCE dataset
- IELTS Writing Scored Essays dataset

After preprocessing, the expected workflow is to create train/dev/test files for the downstream AES experiments and EFCAMDAT proficiency subsets for DAPT.

## Typical reproduction workflow

A typical workflow is:

1. Obtain the raw datasets from their original sources.
2. Run the preprocessing scripts in `preprocessing/`.
3. Run `dapt_bert_efcamdat.py` to create full-corpus EFCAMDAT DAPT checkpoints.
4. Run `finetuning_FCE.py` and `finetuning_IELTS.py` to perform the baseline comparison experiment.
5. Run `cross-dataset-transfer.py` to perform few-shot cross-dataset transfer experiments.
6. Run the scripts in `analysis/` to compute vocabulary distribution and syntactic complexity analyses.
7. Run proficiency-based DAPT variants for the ablation study.
8. Inspect `results/` for spreadsheet summaries of the reported results.
9. Inspect `runs/` for saved metrics and prediction files from individual runs.

## Evaluation metrics

The experiments report:

- RMSE: Root Mean Squared Error
- Pearson correlation
- Spearman correlation
- Quadratic Weighted Kappa

RMSE is treated as the primary metric for model selection and interpretation, while Pearson, Spearman, and QWK are reported for comparability with prior AES studies.

Predictions are saved so that reported metrics can be inspected or recomputed.

## Notes for reviewers

This repository is intended to support transparency and reproducibility without redistributing restricted datasets.

The code covers the main experimental pipeline:

- dataset preprocessing
- learner-domain DAPT
- downstream AES fine-tuning
- few-shot cross-dataset transfer
- proficiency-based ablation
- lexical and syntactic alignment analyses

The `runs/` folder contains saved metric and prediction files for individual experiments, while the `results/` folder contains spreadsheet summaries of the main reported results and analyses.

## Citation

If you use this repository, please cite the accompanying manuscript.

```bibtex
@misc{dapt_efcamdat_aes,
  title        = {Does Continued Pretraining on a Learner Corpus Improve Automated Essay Scoring on English Proficiency Tests? Evidence from EFCAMDAT},
  author       = {Nguyen, D. Anh},
  year         = {2026},
  note         = {Code repository for DAPT-EFCAMDAT AES experiments}
}
```

## License

The code in this repository is released under the MIT License. See the `LICENSE` file for details.

This license applies only to the code in this repository. It does not apply to the datasets, trained model checkpoints, or any text-derived artifacts from EFCAMDAT, IELTS, or FCE. These resources are governed by their respective original licenses and access conditions.
