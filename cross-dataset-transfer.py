import math
import numpy as np
import pandas as pd
import os
import json
import argparse

from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, cohen_kappa_score
from sklearn.model_selection import train_test_split
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


# =========================================================
# Normalization helpers
# =========================================================
def normalize_scores(values, score_min, score_max):
    values = np.asarray(values, dtype=float)
    return (values - score_min) / (score_max - score_min)


def denormalize_scores(values, score_min, score_max):
    values = np.asarray(values, dtype=float)
    return values * (score_max - score_min) + score_min


def normalize_label_column(df, label_col, score_min, score_max):
    df = df.copy()
    df[label_col] = normalize_scores(df[label_col].astype(float), score_min, score_max)
    return df


# =========================================================
# Metrics
# =========================================================
def snap_to_valid_scores(values, valid_scores):
    values = np.asarray(values, dtype=float)
    valid_scores = np.asarray(valid_scores, dtype=float)
    idx = np.abs(values[:, None] - valid_scores[None, :]).argmin(axis=1)
    snapped = valid_scores[idx]
    return snapped, idx


def make_compute_metrics(
    valid_scores=None,
    normalize_labels=False,
    score_min=None,
    score_max=None,
):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.squeeze(preds).astype(float)
        labels = np.squeeze(labels).astype(float)

        # Convert normalized predictions/labels back to raw score scale
        # before calculating final metrics.
        if normalize_labels:
            preds = denormalize_scores(preds, score_min, score_max)
            labels = denormalize_scores(labels, score_min, score_max)

        rmse = math.sqrt(mean_squared_error(labels, preds))

        try:
            pearson = pearsonr(labels, preds)[0]
        except Exception:
            pearson = 0.0

        try:
            spearman = spearmanr(labels, preds)[0]
        except Exception:
            spearman = 0.0

        metrics = {
            "rmse": rmse,
            "pearson": pearson,
            "spearman": spearman,
        }

        if valid_scores is not None:
            _, preds_qwk = snap_to_valid_scores(preds, valid_scores)
            _, labels_qwk = snap_to_valid_scores(labels, valid_scores)
            qwk = cohen_kappa_score(labels_qwk, preds_qwk, weights="quadratic")
            metrics["qwk"] = qwk

        return metrics

    return compute_metrics


# =========================================================
# Data helpers
# =========================================================
def load_csv(path, text_col, label_col):
    df = pd.read_csv(path)
    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(float)
    return df


def sample_few_shot(df, label_col, n_shot=50, seed=42, stratify=True):
    if n_shot >= len(df):
        return df.copy()

    if stratify:
        try:
            few_shot_df, _ = train_test_split(
                df,
                train_size=n_shot,
                random_state=seed,
                stratify=df[label_col]
            )
            return few_shot_df.reset_index(drop=True)
        except Exception:
            print("Stratified sampling failed, falling back to random sampling.")

    return df.sample(n=n_shot, random_state=seed).reset_index(drop=True)


def build_dataset(df, tokenizer, text_col, label_col, max_length=512):
    ds = Dataset.from_pandas(df, preserve_index=False)

    def tokenize_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
        )

    ds = ds.map(tokenize_fn, batched=True)
    ds = ds.rename_column(label_col, "labels")

    columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in ds.column_names:
        columns.append("token_type_ids")

    ds.set_format(type="torch", columns=columns)
    return ds

# =========================================================
# Main few-shot transfer function
# =========================================================
def run_few_shot_transfer(
    source_checkpoint,
    target_train_csv,
    target_dev_csv,
    target_test_csv,
    output_dir,
    text_col,
    label_col,
    n_shot=50,
    seed=42,
    max_length=512,
    num_train_epochs=5,
    learning_rate=2e-5,
    train_batch_size=8,
    eval_batch_size=16,
    weight_decay=0.01,
    target_valid_scores=None,
    tokenizer_name=None,

    # Normalization options
    normalize_labels=False,
    score_min=None,
    score_max=None,
):
    set_seed(seed)

    if normalize_labels:
        if score_min is None or score_max is None:
            raise ValueError("score_min and score_max must be provided when normalize_labels=True.")

    # 1) Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(source_checkpoint, use_fast=True)
    except Exception:
        if tokenizer_name is None:
            raise ValueError(
                "Could not load tokenizer from checkpoint. "
                "Please provide tokenizer_name, e.g. 'bert-base-uncased'."
            )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # 2) Load source-trained checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(
        source_checkpoint,
        num_labels=1,
        problem_type="regression",
    )

    # 3) Load target data
    target_train_df_raw = load_csv(target_train_csv, text_col, label_col)
    target_dev_df_raw = load_csv(target_dev_csv, text_col, label_col)
    target_test_df_raw = load_csv(target_test_csv, text_col, label_col)

    # 4) Sample few-shot subset from target train on raw labels
    few_shot_df_raw = sample_few_shot(
        target_train_df_raw,
        label_col=label_col,
        n_shot=n_shot,
        seed=seed,
        stratify=True,
    )

    print(f"\nFew-shot subset size: {len(few_shot_df_raw)}")
    print("Few-shot label distribution:")
    print(few_shot_df_raw[label_col].value_counts().sort_index())

    # 5) Normalize labels for training/evaluation if requested
    if normalize_labels:
        few_shot_df = normalize_label_column(few_shot_df_raw, label_col, score_min, score_max)
        target_dev_df = normalize_label_column(target_dev_df_raw, label_col, score_min, score_max)
        target_test_df = normalize_label_column(target_test_df_raw, label_col, score_min, score_max)
    else:
        few_shot_df = few_shot_df_raw
        target_dev_df = target_dev_df_raw
        target_test_df = target_test_df_raw

    # 6) Convert to HF datasets
    train_ds = build_dataset(few_shot_df, tokenizer, text_col, label_col, max_length)
    dev_ds = build_dataset(target_dev_df, tokenizer, text_col, label_col, max_length)
    test_ds = build_dataset(target_test_df, tokenizer, text_col, label_col, max_length)

    # 7) Metrics
    compute_metrics = make_compute_metrics(
        valid_scores=target_valid_scores,
        normalize_labels=normalize_labels,
        score_min=score_min,
        score_max=score_max,
    )

    # 8) Training arguments
    metric_for_best_model = "rmse"
    greater_is_better = False

    training_args = TrainingArguments(
        output_dir=output_dir,

        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=weight_decay,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
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

    # 9) Trainer
    trainer = ModernSaveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # 10) Fine-tune on few-shot target subset
    trainer.train()

    # 11) Evaluate on target dev and target test
    dev_metrics = trainer.evaluate(dev_ds, metric_key_prefix="dev")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    # 12) Save predictions
    preds_output = trainer.predict(test_ds)
    preds = np.squeeze(preds_output.predictions).astype(float)

    if normalize_labels:
        preds = denormalize_scores(preds, score_min, score_max)

    pred_df = pd.DataFrame({
        "gold": target_test_df_raw[label_col].astype(float).values,
        "pred": preds
    })

    pred_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    with open(os.path.join(output_dir, "dev_metrics.json"), "w") as f:
      json.dump(dev_metrics, f, indent=2)

    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
      json.dump(test_metrics, f, indent=2)

    print("\nDev metrics:")
    for k, v in dev_metrics.items():
        print(f"{k}: {v}")

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    return dev_metrics, test_metrics, few_shot_df_raw


# Change the score range to match the transfer target.

FCE_VALID_SCORES = np.array([
    0.0, 1.0, 4.0, 7.0, 9.0, 10.0, 11.0, 11.5,
    12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0
])

IELTS_VALID_SCORES = np.arange(1.0, 9.5, 0.5)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run few-shot cross-dataset transfer for AES."
    )

    parser.add_argument(
        "--source_checkpoint",
        type=str,
        default="/content/drive/MyDrive/DAPT_BERT_EFCAMDAT/runs/Fine-tuning/IELTS/ielts_bert",
        help="Path to the model fine-tuned on the source dataset.",
    )

    parser.add_argument(
        "--target_train_csv",
        type=str,
        default="/content/drive/MyDrive/DAPT_BERT_EFCAMDAT/Datasets/Downstream/processed/FCE_processed/train.csv",
        help="Path to the target train CSV.",
    )
    parser.add_argument(
        "--target_dev_csv",
        type=str,
        default="/content/drive/MyDrive/DAPT_BERT_EFCAMDAT/Datasets/Downstream/processed/FCE_processed/dev.csv",
        help="Path to the target dev CSV.",
    )
    parser.add_argument(
        "--target_test_csv",
        type=str,
        default="/content/drive/MyDrive/DAPT_BERT_EFCAMDAT/Datasets/Downstream/processed/FCE_processed/test.csv",
        help="Path to the target test CSV.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/DAPT_BERT_EFCAMDAT/runs/Transfer/IELTS->FCE/base/BERT_50",
        help="Directory where outputs, metrics, and predictions will be saved.",
    )

    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="mapped_score")
    parser.add_argument("--n_shot", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument(
        "--target_valid_scores",
        type=str,
        choices=["fce", "ielts", "none"],
        default="fce",
        help="Valid score set used for QWK snapping.",
    )

    parser.set_defaults(normalize_labels=True)
    parser.add_argument(
        "--normalize_labels",
        dest="normalize_labels",
        action="store_true",
        help="Normalize target labels before training.",
    )
    parser.add_argument(
        "--no_normalize_labels",
        dest="normalize_labels",
        action="store_false",
        help="Do not normalize target labels before training.",
    )

    parser.add_argument("--score_min", type=float, default=0.0)
    parser.add_argument("--score_max", type=float, default=20.0)

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer name to use if tokenizer cannot be loaded from source_checkpoint.",
    )

    return parser.parse_args()


def get_valid_scores(name):
    if name == "fce":
        return FCE_VALID_SCORES
    if name == "ielts":
        return IELTS_VALID_SCORES
    return None


if __name__ == "__main__":
    args = parse_args()

    dev, test, fewshot = run_few_shot_transfer(
        source_checkpoint=args.source_checkpoint,

        target_train_csv=args.target_train_csv,
        target_dev_csv=args.target_dev_csv,
        target_test_csv=args.target_test_csv,
        output_dir=args.output_dir,
        text_col=args.text_col,
        label_col=args.label_col,
        n_shot=args.n_shot,
        seed=args.seed,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,

        target_valid_scores=get_valid_scores(args.target_valid_scores),
        normalize_labels=args.normalize_labels,
        score_min=args.score_min,
        score_max=args.score_max,

        tokenizer_name=args.tokenizer_name,
    )
