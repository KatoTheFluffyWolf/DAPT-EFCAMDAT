import os
import argparse
import pandas as pd
from transformers import AutoTokenizer


def load_efcamdat(path1, path2):
    file1 = pd.read_excel(path1)
    file2 = pd.read_excel(path2)
    df = pd.concat([file1, file2], ignore_index=True)
    return df


def count_bert_tokens(df, text_col="text", model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    total_tokens = 0
    for text in df[text_col].astype(str):
        total_tokens += len(tokenizer.tokenize(text))

    return total_tokens


def split_efcamdat_by_proficiency(
    path1,
    path2,
    output_dir="efcamdat_splits",
    text_col="text",
    cefr_col="cefr",
    model_name="bert-base-uncased"
):
    os.makedirs(output_dir, exist_ok=True)

    # Load and merge
    df = load_efcamdat(path1, path2)

    # Keep only rows with text and CEFR
    df = df.dropna(subset=[text_col, cefr_col]).copy()

    # Normalize CEFR labels
    df[cefr_col] = df[cefr_col].astype(str).str.upper().str.strip()

    # Split
    a1_a2 = df[df[cefr_col].isin(["A1", "A2"])].copy()
    b1_b2 = df[df[cefr_col].isin(["B1", "B2"])].copy()
    c1 = df[df[cefr_col] == "C1"].copy()

    # Save
    a1_a2.to_excel(os.path.join(output_dir, "A1_A2.xlsx"), index=False)
    b1_b2.to_excel(os.path.join(output_dir, "B1_B2.xlsx"), index=False)
    c1.to_excel(os.path.join(output_dir, "C1.xlsx"), index=False)

    # Token logs
    a1_a2_tokens = count_bert_tokens(a1_a2, text_col=text_col, model_name=model_name)
    b1_b2_tokens = count_bert_tokens(b1_b2, text_col=text_col, model_name=model_name)
    c1_tokens = count_bert_tokens(c1, text_col=text_col, model_name=model_name)

    # Summary table
    summary = pd.DataFrame({
        "split": ["A1_A2", "B1_B2", "C1"],
        "n_rows": [len(a1_a2), len(b1_b2), len(c1)],
        "total_bert_tokens": [a1_a2_tokens, b1_b2_tokens, c1_tokens]
    })

    summary.to_csv(os.path.join(output_dir, "split_summary.csv"), index=False, encoding="utf-8-sig")

    print("Saved split files to:", output_dir)
    print(summary)

    return a1_a2, b1_b2, c1, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split EFCAMDAT by CEFR proficiency and save Excel files plus token-count summary."
    )

    parser.add_argument(
        "--path1",
        type=str,
        required=True,
        help="Path to the first Excel file."
    )
    parser.add_argument(
        "--path2",
        type=str,
        required=True,
        help="Path to the second Excel file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="efcamdat_splits",
        help="Directory to save the split Excel files and summary CSV."
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="text",
        help="Name of the text column."
    )
    parser.add_argument(
        "--cefr_col",
        type=str,
        default="cefr",
        help="Name of the CEFR column."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Tokenizer model name used for token counting."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    a1_a2, b1_b2, c1, summary = split_efcamdat_by_proficiency(
        args.path1,
        args.path2,
        output_dir=args.output_dir,
        text_col=args.text_col,
        cefr_col=args.cefr_col,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()