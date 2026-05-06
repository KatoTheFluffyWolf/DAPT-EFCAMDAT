
import argparse
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
# -----------------------------
# Stearns-style DAPT defaults
# -----------------------------
DEFAULT_MODEL = "bert-base-uncased"
DEFAULT_MAX_LEN = 512          # BERT base limit (Stearns filtered >512 WordPiece tokens)
DEFAULT_MLM_PROB = 0.15        # "masked 15% of WordPiece tokens"
DEFAULT_VAL_RATIO = 0.05


@dataclass
class DAPTConfig:
    model_name_or_path: str = DEFAULT_MODEL
    max_seq_length: int = DEFAULT_MAX_LEN
    mlm_probability: float = DEFAULT_MLM_PROB

    num_train_epochs: float = 1.0
    max_steps: int = 0

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 5154
    lr_scheduler_type: str = "linear"

    fp16: bool = True
    bf16: bool = False

    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2

    seed: int = 42
    val_ratio: float = DEFAULT_VAL_RATIO

    deduplicate: bool = True
    min_char_len: int = 1

    subset_frac: float = 1.0
    max_train_tokens: int = 0


def read_xlsx_texts(xlsx_paths, text_col, sheet_name=None):
    texts = []
    for path in xlsx_paths:
        # If sheet_name is None, read the first sheet (0) instead of "all sheets"
        df = pd.read_excel(path, sheet_name=0 if sheet_name is None else sheet_name, engine="openpyxl")

        if text_col not in df.columns:
            raise ValueError(f"[{path}] Column '{text_col}' not found. Columns: {list(df.columns)}")

        col = df[text_col].dropna().astype(str).map(lambda x: x.strip())
        texts.extend([t for t in col.tolist() if len(t) > 0])

    return texts


def maybe_dedup(texts: List[str]) -> List[str]:
    # Preserve order while deduplicating
    seen = set()
    out = []
    for t in texts:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def build_dataset(texts: List[str]) -> Dataset:
    return Dataset.from_dict({"text": texts})

def cap_dataset_by_tokens(ds: Dataset, max_tokens: int) -> Dataset:
    """
    Keep examples from the shuffled dataset until cumulative wp_len reaches max_tokens.
    Assumes ds already has a 'wp_len' column and has already been filtered by max length.
    """
    if max_tokens <= 0:
        return ds

    kept_indices = []
    total_tokens = 0

    for i, ex in enumerate(ds):
        ex_len = int(ex["wp_len"])

        # skip weird empty examples just in case
        if ex_len <= 0:
            continue

        if total_tokens + ex_len > max_tokens:
            break

        kept_indices.append(i)
        total_tokens += ex_len

    if len(kept_indices) == 0:
        raise ValueError(
            f"max_train_tokens={max_tokens} is too small; not enough to keep even one example."
        )

    capped = ds.select(kept_indices)
    print(f"Capped training set to {len(capped)} texts / {total_tokens} WordPiece tokens")
    return capped

def add_wordpiece_len_column(ds: Dataset, tokenizer, batch_size: int = 512) -> Dataset:
    """
    Compute WordPiece token length without truncation.
    We count WordPiece tokens (excluding special tokens).
    """
    def _len_fn(batch):
        enc = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        lengths = [len(ids) for ids in enc["input_ids"]]
        return {"wp_len": lengths}

    return ds.map(_len_fn, batched=True, batch_size=batch_size)


def filter_by_wordpiece_len(ds: Dataset, max_len: int) -> Dataset:
    # Stearns: "filtered out texts with more than 512 WordPiece tokens"
    return ds.filter(lambda ex: ex["wp_len"] <= max_len)


def tokenize_for_mlm(ds: Dataset, tokenizer, max_len: int, batch_size: int = 256) -> Dataset:
    """
    Tokenize examples (no masking here).
    Masking is done dynamically by DataCollatorForLanguageModeling during training.
    """
    def _tok_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,  # safe even after filtering
            max_length=max_len,
            padding=False,    # dynamic padding (more efficient than padding to max_length)
            return_special_tokens_mask=True,
        )

    tok = ds.map(_tok_fn, batched=True, batch_size=batch_size, remove_columns=["text"])
    tok.set_format(type="torch", columns=["input_ids", "attention_mask", "special_tokens_mask"])
    return tok


def auto_precision_flags(cfg: DAPTConfig) -> DAPTConfig:
    # If user leaves fp16=True, keep it (works on most Colab GPUs).
    # If fp16=False, optionally enable bf16 if supported.
    if cfg.fp16:
        cfg.bf16 = False
        return cfg
    if torch.cuda.is_available():
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                cfg.bf16 = True
        except Exception:
            pass
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Stearns-style DAPT (MLM) for BERT on XLSX text.")
    parser.add_argument("--xlsx", type=str, nargs="+", required=True, help="One or more .xlsx paths")
    parser.add_argument("--text_col", type=str, required=True, help="Text column name in the .xlsx")
    parser.add_argument("--sheet", type=str, default=None, help="Sheet name (optional). Default: first sheet.")
    parser.add_argument("--out", type=str, required=True, help="Output directory for checkpoints/model")

    # Core Stearns-style DAPT knobs
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--mlm_prob", type=float, default=DEFAULT_MLM_PROB)

    # Training budget
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=0)

    # Memory/throughput
    parser.add_argument("--train_bs", type=int, default=1)
    parser.add_argument("--eval_bs", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=8)

    # Optim
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=5154)

    # Precision
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 (recommended on Colab)")
    parser.add_argument("--no_fp16", action="store_true", help="Disable fp16")

    # Capping max token
    parser.add_argument("--max_train_tokens", type=int, default=0, help="Cap the total number of WordPiece training tokens. 0 means no cap.")
    # Split / hygiene
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_dedup", action="store_true", help="Disable deduplication")
    parser.add_argument("--min_char_len", type=int, default=1)

    # Logging/checkpoints
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in --out if present")

    parser.add_argument("--subset_frac", type=float, default=1.0, help="Fraction of the training pool to use for DAPT (e.g. 0.25, 0.5, 0.75, 1.0)")

    parser.add_argument("--resume_from_ckpt", type=str, default=None, help="Path to a specific checkpoint to resume the SAME run from")

    args = parser.parse_args()


    cfg = DAPTConfig(
    model_name_or_path=args.model,
    max_seq_length=args.max_len,
    mlm_probability=args.mlm_prob,
    num_train_epochs=args.epochs,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.train_bs,
    per_device_eval_batch_size=args.eval_bs,
    gradient_accumulation_steps=args.grad_acc,
    learning_rate=args.lr,
    weight_decay=args.weight_decay,
    warmup_steps=args.warmup_steps,
    fp16=(False if args.no_fp16 else True if args.fp16 or not args.no_fp16 else False),
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    eval_steps=args.eval_steps,
    save_total_limit=args.save_total_limit,
    seed=args.seed,
    val_ratio=args.val_ratio,
    deduplicate=(not args.no_dedup),
    min_char_len=args.min_char_len,
    subset_frac=args.subset_frac,
    max_train_tokens=args.max_train_tokens,
)
    cfg = auto_precision_flags(cfg)

    os.makedirs(args.out, exist_ok=True)

    # Reproducibility
    set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # 1) Load texts from XLSX
    sheet = None if args.sheet in (None, "", "None") else args.sheet
    texts = read_xlsx_texts(args.xlsx, args.text_col, sheet_name=sheet)
    texts = [t for t in texts if len(t) >= cfg.min_char_len]

    if cfg.deduplicate:
        texts = maybe_dedup(texts)

    if len(texts) < 100:
        raise ValueError(f"Too few texts loaded ({len(texts)}). Check xlsx/text_col/sheet.")

    # 2) Build dataset + split
    ds = build_dataset(texts).shuffle(seed=cfg.seed)

    # Fixed train/eval split first, so all ablations use the same eval set
    split = ds.train_test_split(test_size=cfg.val_ratio, seed=cfg.seed)
    train_pool = split["train"]
    eval_ds = split["test"]

    # Nested subset from the same shuffled training pool
    if not (0 < cfg.subset_frac <= 1.0):
        raise ValueError(f"subset_frac must be in (0, 1], got {cfg.subset_frac}")

    subset_n = max(1, int(len(train_pool) * cfg.subset_frac))
    train_ds = train_pool.select(range(subset_n))

    print(f"Using {subset_n}/{len(train_pool)} training texts "
          f"({cfg.subset_frac*100:.0f}% of train pool)")

    # 3) Tokenizer + model (MLM head)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(cfg.model_name_or_path)

    # Optional memory saver (especially for 512)
    model.gradient_checkpointing_enable()

    # 4) Compute WordPiece length (no truncation), then filter >512 (Stearns)
    train_ds = add_wordpiece_len_column(train_ds, tokenizer)
    eval_ds = add_wordpiece_len_column(eval_ds, tokenizer)

    before_train = len(train_ds)
    before_eval = len(eval_ds)

    train_ds = filter_by_wordpiece_len(train_ds, cfg.max_seq_length)
    eval_ds = filter_by_wordpiece_len(eval_ds, cfg.max_seq_length)

    after_train = len(train_ds)
    after_eval = len(eval_ds)

    train_ds = cap_dataset_by_tokens(train_ds, cfg.max_train_tokens)

    if after_train < 100:
        raise ValueError(
            f"After filtering >{cfg.max_seq_length} WordPiece tokens, train set is too small "
            f"({after_train}/{before_train}). Consider using max_len=256 or providing shorter text units."
        )
    final_train_tokens = sum(train_ds["wp_len"])
    final_eval_tokens = sum(eval_ds["wp_len"])

    print(f"Final train WordPiece tokens: {final_train_tokens}")
    print(f"Final eval WordPiece tokens:  {final_eval_tokens}")

    # Remove helper column
    train_ds = train_ds.remove_columns(["wp_len"])
    eval_ds = eval_ds.remove_columns(["wp_len"])

    # 5) Tokenize (no padding here; dynamic padding in collator)
    train_tok = tokenize_for_mlm(train_ds, tokenizer, cfg.max_seq_length)
    eval_tok = tokenize_for_mlm(eval_ds, tokenizer, cfg.max_seq_length)

    # 6) Dynamic masking collator (Stearns: 15% WordPiece masked)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.mlm_probability,
        pad_to_multiple_of=8,  # GPU-friendly
    )

    # 7) Training arguments (research-friendly)
    # Prefer max_steps for reproducible compute budgets.
    # If max_steps > 0, HF Trainer ignores num_train_epochs.
    training_args = TrainingArguments(

        output_dir=args.out,

        # Train/Eval switches (optional but clear)
        do_train=True,
        do_eval=True,

        # Budget: if max_steps > 0, it overrides epochs
        num_train_epochs=cfg.num_train_epochs if cfg.max_steps <= 0 else 1.0,
        max_steps=cfg.max_steps if cfg.max_steps > 0 else -1,

        # Batching
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,

        # Optim
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,

        # Precision
        fp16=cfg.fp16,
        bf16=cfg.bf16,

        # Logging
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,

        #Eval
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,

        # Checkpointing
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,

        # Misc
        seed=cfg.seed,
        dataloader_num_workers=2,
        report_to="none",
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


    # 8) Trainer
    trainer = ModernSaveTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=data_collator,
    processing_class=tokenizer,
)

    # 9) Train (optionally resume)
    resume_ckpt = None
    if args.resume_from_ckpt is not None:
        resume_ckpt = args.resume_from_ckpt
    elif args.resume:
        resume_ckpt = True

    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.out)


    # 10) Eval + perplexity
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss", None)
    if eval_loss is not None and eval_loss < 50:
        eval_metrics["perplexity"] = float(math.exp(eval_loss))

    # Print useful summary
    print("\n===== DAPT Summary =====")
    print(f"Model: {cfg.model_name_or_path}")
    print(f"Max length: {cfg.max_seq_length} (filtered >{cfg.max_seq_length} WordPiece tokens)")
    print(f"MLM prob: {cfg.mlm_probability}")
    print(f"Train texts: {after_train}/{before_train} after filtering")
    print(f"Eval texts:  {after_eval}/{before_eval} after filtering")
    if cfg.max_steps > 0:
        print(f"Training budget: max_steps={cfg.max_steps}")
    else:
        print(f"Training budget: epochs={cfg.num_train_epochs}")
    print(f"Saved to: {args.out}")

    if eval_loss is not None:
        ppl = eval_metrics.get("perplexity", None)
        if ppl is not None:
            print(f"Eval loss: {eval_loss:.4f} | Perplexity: {ppl:.2f}")
        else:
            print(f"Eval loss: {eval_loss:.4f}")
    print("========================\n")

# 21019615

if __name__ == "__main__":
    main()
