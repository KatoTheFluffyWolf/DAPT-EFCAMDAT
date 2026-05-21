import argparse
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NeoSCA/L2SCA-style syntactic complexity analysis on a CSV/XLSX corpus."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV or XLSX file."
    )

    parser.add_argument(
        "--text_col",
        required=True,
        help="Name of the column containing essay/text content."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV file with L2SCA/NeoSCA features."
    )

    parser.add_argument(
        "--id_col",
        default=None,
        help="Optional document ID column. If omitted, row index will be used."
    )

    parser.add_argument(
        "--sheet",
        default=0,
        help="XLSX sheet name or index. Default: 0."
    )

    parser.add_argument(
        "--group_col",
        default=None,
        help="Optional column for filtering groups, e.g. CEFR/proficiency column."
    )

    parser.add_argument(
        "--group_values",
        nargs="*",
        default=None,
        help="Optional group values to keep, e.g. A1 A2 or B1 B2."
    )

    parser.add_argument(
        "--sample_n",
        type=int,
        default=None,
        help="Optional number of documents to sample after filtering."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling."
    )

    parser.add_argument(
        "--min_words",
        type=int,
        default=1,
        help="Minimum number of whitespace-tokenized words required."
    )

    parser.add_argument(
        "--max_words",
        type=int,
        default=None,
        help="Optional maximum number of whitespace-tokenized words allowed."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of text files to send to NeoSCA per subprocess call."
    )

    parser.add_argument(
        "--neosca_cmd",
        default="python -m neosca sca",
        help=(
            "Command used to run NeoSCA. Default: 'python -m neosca sca'. "
            "If your installation uses nsca, try: --neosca_cmd 'nsca'"
        )
    )

    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary text files after running."
    )

    parser.add_argument(
        "--temp_dir",
        default=None,
        help="Optional directory for temporary text files."
    )

    return parser.parse_args()


def read_input(path, sheet=0):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        # Convert numeric-looking sheet value to int.
        try:
            sheet_value = int(sheet)
        except Exception:
            sheet_value = sheet
        return pd.read_excel(path, sheet_name=sheet_value, engine="openpyxl")

    raise ValueError("Input file must be CSV, XLSX, or XLS.")


def clean_text(text):
    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_filename(value, fallback):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("_")
    if not value:
        value = str(fallback)
    return value[:120]


def filter_dataframe(df, args):
    if args.text_col not in df.columns:
        raise ValueError(
            f"text_col '{args.text_col}' not found. Available columns: {list(df.columns)}"
        )

    df = df.copy()

    if args.id_col is not None and args.id_col not in df.columns:
        raise ValueError(
            f"id_col '{args.id_col}' not found. Available columns: {list(df.columns)}"
        )

    if args.group_col is not None:
        if args.group_col not in df.columns:
            raise ValueError(
                f"group_col '{args.group_col}' not found. Available columns: {list(df.columns)}"
            )

        if args.group_values:
            df = df[df[args.group_col].astype(str).isin([str(v) for v in args.group_values])]

    df = df.dropna(subset=[args.text_col]).copy()
    df[args.text_col] = df[args.text_col].map(clean_text)

    df["word_count_for_filter"] = df[args.text_col].str.split().map(len)
    df = df[df["word_count_for_filter"] >= args.min_words]

    if args.max_words is not None:
        df = df[df["word_count_for_filter"] <= args.max_words]

    if args.sample_n is not None and args.sample_n < len(df):
        df = df.sample(n=args.sample_n, random_state=args.seed)

    df = df.reset_index(drop=True)

    if args.id_col is None:
        df["doc_id"] = [f"doc_{i:06d}" for i in range(len(df))]
    else:
        df["doc_id"] = df[args.id_col].astype(str)

    return df


def write_text_files(df, text_col, temp_dir):
    file_records = []

    for i, row in df.iterrows():
        doc_id = row["doc_id"]
        fname = safe_filename(doc_id, fallback=i) + ".txt"
        fpath = Path(temp_dir) / fname

        # NeoSCA expects plain text files.
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(row[text_col])

        file_records.append({
            "doc_id": doc_id,
            "temp_file": str(fpath),
            "temp_filename": fname,
        })

    return pd.DataFrame(file_records)


def run_neosca_batch(neosca_cmd, files, out_csv):
    cmd = shlex.split(neosca_cmd)
    cmd.extend(files)
    cmd.extend(["-o", str(out_csv)])

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "NeoSCA failed.\n\n"
            f"Command:\n{' '.join(cmd)}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    if not Path(out_csv).exists():
        raise FileNotFoundError(
            f"NeoSCA finished but output file was not created: {out_csv}"
        )


def read_neosca_output(path):
    df = pd.read_csv(path)

    # NeoSCA output column names may vary slightly by version.
    # We keep all columns and later try to identify the file/name column.
    return df


def detect_filename_column(df):
    candidates = [
        "Filename", "filename", "File", "file", "Filepath", "filepath",
        "Path", "path", "Source", "source"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # Fallback: first object/string column that looks file-like.
    for col in df.columns:
        if df[col].dtype == object:
            sample_values = df[col].dropna().astype(str).head(10).tolist()
            if any(v.endswith(".txt") or ".txt" in v for v in sample_values):
                return col

    return None


def merge_features_with_metadata(feature_df, file_df, original_df):
    filename_col = detect_filename_column(feature_df)

    if filename_col is not None:
        feature_df = feature_df.copy()
        feature_df["temp_filename"] = feature_df[filename_col].astype(str).map(
            lambda x: Path(x).name
        )

        merged = feature_df.merge(
            file_df[["doc_id", "temp_filename"]],
            on="temp_filename",
            how="left"
        )
    else:
        # If NeoSCA does not write file names, assume output row order matches input order.
        # This is less ideal but usable for controlled batch calls.
        feature_df = feature_df.copy()
        feature_df["doc_id"] = file_df["doc_id"].values[:len(feature_df)]
        merged = feature_df

    metadata_cols = ["doc_id"]

    for col in original_df.columns:
        if col not in [original_df.attrs.get("text_col"), "word_count_for_filter"]:
            if col != "doc_id":
                metadata_cols.append(col)

    metadata = original_df[metadata_cols].drop_duplicates(subset=["doc_id"])

    merged = merged.merge(metadata, on="doc_id", how="left")

    # Put doc_id first.
    cols = ["doc_id"] + [c for c in merged.columns if c != "doc_id"]
    return merged[cols]


def main():
    args = parse_args()

    df = read_input(args.input, sheet=args.sheet)
    df = filter_dataframe(df, args)
    df.attrs["text_col"] = args.text_col

    if len(df) == 0:
        raise ValueError("No documents left after filtering.")

    print(f"Documents to process: {len(df)}")

    if args.temp_dir is not None:
        temp_root = Path(args.temp_dir)
        temp_root.mkdir(parents=True, exist_ok=True)
        cleanup_temp = False
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="l2sca_neosca_"))
        cleanup_temp = not args.keep_temp

    text_dir = temp_root / "texts"
    batch_out_dir = temp_root / "batch_outputs"
    text_dir.mkdir(parents=True, exist_ok=True)
    batch_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        file_df = write_text_files(df, args.text_col, text_dir)

        all_feature_dfs = []

        file_paths = file_df["temp_file"].tolist()
        total_batches = int(np.ceil(len(file_paths) / args.batch_size))

        for batch_idx in range(total_batches):
            start = batch_idx * args.batch_size
            end = start + args.batch_size
            batch_files = file_paths[start:end]

            batch_out = batch_out_dir / f"neosca_batch_{batch_idx:04d}.csv"

            print(
                f"Running NeoSCA batch {batch_idx + 1}/{total_batches} "
                f"({len(batch_files)} files)..."
            )

            run_neosca_batch(
                neosca_cmd=args.neosca_cmd,
                files=batch_files,
                out_csv=batch_out,
            )

            batch_features = read_neosca_output(batch_out)
            all_feature_dfs.append(batch_features)

        features = pd.concat(all_feature_dfs, ignore_index=True)

        merged = merge_features_with_metadata(
            feature_df=features,
            file_df=file_df,
            original_df=df,
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)

        print(f"\nSaved L2SCA/NeoSCA features to: {output_path}")
        print(f"Rows: {len(merged)}")
        print(f"Columns: {len(merged.columns)}")

        if args.keep_temp or args.temp_dir is not None:
            print(f"Temporary files kept at: {temp_root}")

    finally:
        if cleanup_temp:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
