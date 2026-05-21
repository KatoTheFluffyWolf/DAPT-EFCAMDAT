import os
import math
import numpy as np
import pandas as pd
import json
import argparse

from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, cohen_kappa_score

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# -----------------------------
# Reproducibility
# -----------------------------
def set_seeds(seed=42):
    set_seed(seed)

# -----------------------------
# Load CSV
# -----------------------------
def load_dataset_csv(path):
    return pd.read_csv(path)

# -----------------------------
# IELTS score normalization
# -----------------------------
IELTS_MIN_SCORE = 1.0
IELTS_MAX_SCORE = 9.0

def normalize_ielts_score(values):
    values = np.asarray(values, dtype=float)
    return (values - IELTS_MIN_SCORE) / (IELTS_MAX_SCORE - IELTS_MIN_SCORE)

def denormalize_ielts_score(values):
    values = np.asarray(values, dtype=float)
    return values * (IELTS_MAX_SCORE - IELTS_MIN_SCORE) + IELTS_MIN_SCORE

# -----------------------------
# Load model + tokenizer
# -----------------------------
def load_from_pretrained(
    checkpoint_path=None,
    model_name="bert-base-uncased",
    num_labels=1,
):

    """ Replace with

        google-bert/bert-base-uncased
        FacebookAI/roberta-base
        microsoft/deberta-v3-base
        distilbert/distilbert-base-uncased

        for other baselines"""

    model_source = checkpoint_path if checkpoint_path is not None else model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=num_labels,
    )

    model.config.problem_type = "regression"
    return tokenizer, model

# -----------------------------
# Convert pandas -> HF Dataset
# -----------------------------
def build_hf_dataset(df, text_col="Essay", label_col="Overall"):
    df = df[[text_col, label_col]].dropna().copy()
    df[label_col] = df[label_col].astype(float)

    # Normalize IELTS scores from raw 1.0–9.0 scale to 0.0–1.0 scale
    df[label_col] = normalize_ielts_score(df[label_col])

    return Dataset.from_pandas(df, preserve_index=False)

# -----------------------------
# Tokenization
# -----------------------------
def tokenize_dataset(ds, tokenizer, text_col="Essay", label_col="Overall", max_length=512):
    def tokenize_batch(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
        )

    ds = ds.map(tokenize_batch, batched=True)
    ds = ds.rename_column(label_col, "labels")

    columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in ds.column_names:
        columns.insert(2, "token_type_ids")

    ds.set_format(type="torch", columns=columns)
    return ds

# -----------------------------
# Metrics
# -----------------------------

IELTS_VALID_SCORES = np.arange(1.0, 9.5, 0.5) #In this dataset, IELTS score ranges from 1.0 to 9.0

def snap_to_valid_scores(values, valid_scores):
    values = np.asarray(values)
    valid_scores = np.asarray(valid_scores)
    idx = np.abs(values[:, None] - valid_scores[None, :]).argmin(axis=1)
    return valid_scores[idx]

def ielts_band_to_class(values):
    # 1.0 -> 0, 1.5 -> 1, ..., 9.0 -> 16
    values = np.asarray(values, dtype=float)
    return ((values - 1.0) * 2).astype(int)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.squeeze(preds).astype(float)
    labels = np.squeeze(labels).astype(float)

    # Convert normalized predictions and labels back to raw IELTS scale
    preds = denormalize_ielts_score(preds)
    labels = denormalize_ielts_score(labels)

    rmse = math.sqrt(mean_squared_error(labels, preds))

    pearson = pearsonr(labels, preds)[0]

    spearman = spearmanr(labels, preds)[0]

    # Snap both to valid IELTS scores
    preds_qwk = snap_to_valid_scores(preds, IELTS_VALID_SCORES)
    labels_qwk = snap_to_valid_scores(labels, IELTS_VALID_SCORES)

    # Convert to ordinal integer classes
    preds_qwk = ielts_band_to_class(preds_qwk)
    labels_qwk = ielts_band_to_class(labels_qwk)

    qwk = cohen_kappa_score(labels_qwk, preds_qwk, weights="quadratic")

    return {
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
        "qwk": qwk,
    }

# -----------------------------
# One full training run
# -----------------------------
def run_experiment(
    train_df,
    dev_df,
    test_df,
    output_dir,
    checkpoint_path=None,
    model_name="bert-base-uncased",
    text_col="Essay",
    label_col="Overall",
    max_length=512,
    learning_rate=2e-5,
    train_batch_size=8,
    eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    seed=42,
):
    set_seeds(seed)

    tokenizer, model = load_from_pretrained(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        num_labels=1,
    )

    train_ds = build_hf_dataset(train_df, text_col=text_col, label_col=label_col)
    dev_ds   = build_hf_dataset(dev_df,   text_col=text_col, label_col=label_col)
    test_ds  = build_hf_dataset(test_df,  text_col=text_col, label_col=label_col)

    train_ds = tokenize_dataset(train_ds, tokenizer, text_col=text_col, label_col=label_col, max_length=max_length)
    dev_ds   = tokenize_dataset(dev_ds,   tokenizer, text_col=text_col, label_col=label_col, max_length=max_length)
    test_ds  = tokenize_dataset(test_ds,  tokenizer, text_col=text_col, label_col=label_col, max_length=max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        save_total_limit=1,
        report_to="none",
        seed=seed,
        disable_tqdm=True,
    )

    class ModernSaveTrainer(Trainer): #For loading BERT in the modern format with weight and bias instead of gamma and beta
      def _save(self, output_dir=None, state_dict=None):
          output_dir = output_dir if output_dir is not None else self.args.output_dir
          os.makedirs(output_dir, exist_ok=True)

          model_to_save = self.model
          if hasattr(self, "accelerator"):
              model_to_save = self.accelerator.unwrap_model(self.model)

          model_to_save.save_pretrained(
              output_dir,
              state_dict=state_dict,
              save_original_format=False,
          )

          if self.processing_class is not None:
              self.processing_class.save_pretrained(output_dir)

    trainer = ModernSaveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save best model + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Dev metrics
    dev_metrics = trainer.evaluate(dev_ds)

    # Test metrics
    test_metrics = trainer.evaluate(test_ds)

    with open(os.path.join(output_dir, "dev_metrics.json"), "w") as f:
      json.dump(dev_metrics, f, indent=2)

    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
      json.dump(test_metrics, f, indent=2)

    # Save predictions
    preds_output = trainer.predict(test_ds)
    preds = np.squeeze(preds_output.predictions)

    # Convert normalized predictions back to raw IELTS scale
    preds = denormalize_ielts_score(preds)

    pred_df = pd.DataFrame({
        "gold": test_df[label_col].astype(float).values,
        "pred": preds
    })

    pred_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    return dev_metrics, test_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune/evaluate a sequence-regression model on IELTS Task 2.")

    parser.add_argument("--checkpoint_path", type=str, default="/content/drive/MyDrive/DAPT_BERT_EFCAMDAT/DAPT_checkpoints/B1-B2-BERT-efcamdat/checkpoint-974")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")

    parser.add_argument("--train_csv", type=str, default="/content/drive/MyDrive/DAPT_BERT_EFCAMDAT/Datasets/Downstream/processed/IELTS_processed/task2_train.csv")
    parser.add_argument("--dev_csv", type=str, default="/content/drive/MyDrive/DAPT_BERT_EFCAMDAT/Datasets/Downstream/processed/IELTS_processed/task2_dev.csv")
    parser.add_argument("--test_csv", type=str, default="/content/drive/MyDrive/DAPT_BERT_EFCAMDAT/Datasets/Downstream/processed/IELTS_processed/task2_test.csv")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/DAPT_BERT_EFCAMDAT/runs/Ablation/ielts_b1b2")

    parser.add_argument("--text_col", type=str, default="Essay")
    parser.add_argument("--label_col", type=str, default="Overall")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=float, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    checkpoint_path = args.checkpoint_path if args.checkpoint_path not in (None, "", "None") else None

    train_task2 = load_dataset_csv(args.train_csv)
    dev_task2 = load_dataset_csv(args.dev_csv)
    test_task2 = load_dataset_csv(args.test_csv)

    # Task 2
    dev_t2, test_t2 = run_experiment(
        train_df=train_task2,
        dev_df=dev_task2,
        test_df=test_task2,
        output_dir=args.output_dir,
        checkpoint_path=checkpoint_path,
        model_name=args.model_name,
        text_col=args.text_col,
        label_col=args.label_col,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
