import argparse
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


# =========================================================
# Corpus specification
# =========================================================

@dataclass
class CorpusSpec:
    name: str
    paths: List[str]


def default_corpus_name(path: str) -> str:
    p = Path(path)
    return p.stem if p.is_file() else p.name


def load_manifest(manifest_path: str) -> List[CorpusSpec]:
    """
    Manifest CSV format:
    name,path

    Multiple rows with the same name are merged into one corpus.
    """
    df = pd.read_csv(manifest_path)

    required = {"name", "path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest {manifest_path} is missing columns: {sorted(missing)}. "
            "Expected columns: name,path"
        )

    grouped = defaultdict(list)
    order = []

    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        path = str(row["path"]).strip()

        if not name or not path:
            continue

        if name not in grouped:
            order.append(name)

        grouped[name].append(path)

    return [CorpusSpec(name=name, paths=grouped[name]) for name in order]


def make_corpus_specs(paths: Optional[List[str]], names: Optional[List[str]], manifest: Optional[str]) -> List[CorpusSpec]:
    if manifest is not None:
        return load_manifest(manifest)

    if not paths:
        raise ValueError("Provide either --paths or --manifest.")

    if names is not None and len(names) != len(paths):
        raise ValueError(
            f"--names must have the same length as --paths. "
            f"Got {len(names)} names and {len(paths)} paths."
        )

    specs = []
    for i, path in enumerate(paths):
        name = names[i] if names is not None else default_corpus_name(path)
        specs.append(CorpusSpec(name=name, paths=[path]))

    return specs


# =========================================================
# Stopwords and tokenization
# =========================================================

def load_stopwords(stopword_library: str = "nltk") -> set:
    stopword_library = stopword_library.lower()

    if stopword_library == "nltk":
        try:
            import nltk
            from nltk.corpus import stopwords

            try:
                return set(stopwords.words("english"))
            except LookupError:
                print("NLTK stopwords not found. Downloading now...")
                nltk.download("stopwords", quiet=True)
                return set(stopwords.words("english"))

        except Exception as e:
            print(f"Could not use NLTK stopwords: {e}")
            print("Falling back to sklearn English stopwords.")
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            return set(ENGLISH_STOP_WORDS)

    if stopword_library == "sklearn":
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return set(ENGLISH_STOP_WORDS)

    raise ValueError("Unsupported stopword_library. Choose either 'nltk' or 'sklearn'.")


def simple_word_tokenize(text: str, lowercase: bool = True) -> List[str]:
    if not isinstance(text, str):
        return []

    if lowercase:
        text = text.lower()

    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)


def remove_stopwords_from_tokens(tokens: List[str], stopword_set: set, min_word_len: int = 3) -> List[str]:
    return [
        token for token in tokens
        if token.lower() not in stopword_set and len(token) >= min_word_len
    ]


# =========================================================
# File loading
# =========================================================

def find_supported_files(input_path: str) -> List[Path]:
    input_path = Path(input_path)
    supported_suffixes = {".txt", ".csv", ".xlsx", ".xls"}

    if input_path.is_file():
        if input_path.suffix.lower() not in supported_suffixes:
            raise ValueError(f"Unsupported file type: {input_path}")
        return [input_path]

    if input_path.is_dir():
        files = [
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in supported_suffixes
        ]

        if not files:
            raise ValueError(f"No supported files found in directory: {input_path}")

        return sorted(files)

    raise ValueError(f"Path does not exist: {input_path}")


def iter_texts_from_file(file_path: Path, text_col: Optional[str] = None, sheet_name=0) -> Iterable[str]:
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

    elif suffix == ".csv":
        if text_col is None:
            raise ValueError(f"text_col must be provided for CSV file: {file_path}")

        df = pd.read_csv(file_path)

        if text_col not in df.columns:
            raise ValueError(
                f"Column '{text_col}' not found in {file_path}. "
                f"Available columns: {list(df.columns)}"
            )

        for text in df[text_col].dropna().astype(str):
            text = text.strip()
            if text:
                yield text

    elif suffix == ".xlsx":
        if text_col is None:
            raise ValueError(f"text_col must be provided for Excel file: {file_path}")

        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")

        if text_col not in df.columns:
            raise ValueError(
                f"Column '{text_col}' not found in {file_path}. "
                f"Available columns: {list(df.columns)}"
            )

        for text in df[text_col].dropna().astype(str):
            text = text.strip()
            if text:
                yield text

    elif suffix == ".xls":
        if text_col is None:
            raise ValueError(f"text_col must be provided for Excel file: {file_path}")

        df = pd.read_excel(file_path, sheet_name=sheet_name)

        if text_col not in df.columns:
            raise ValueError(
                f"Column '{text_col}' not found in {file_path}. "
                f"Available columns: {list(df.columns)}"
            )

        for text in df[text_col].dropna().astype(str):
            text = text.strip()
            if text:
                yield text

    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def collect_files(paths: List[str]) -> List[Path]:
    all_files = []
    for path in paths:
        all_files.extend(find_supported_files(path))
    return sorted(set(all_files))


# =========================================================
# Counting
# =========================================================

def build_word_counter(
    corpus: CorpusSpec,
    text_col: Optional[str],
    sheet_name=0,
    lowercase=True,
    remove_function_words=False,
    min_word_len=3,
    stopword_set=None,
) -> dict:
    files = collect_files(corpus.paths)

    counter = Counter()
    n_texts = 0
    n_tokens = 0

    print(f"\nReading corpus: {corpus.name}")
    print(f"Found {len(files)} supported file(s).")

    for file_path in files:
        print(f"Processing: {file_path}")

        for text in iter_texts_from_file(file_path, text_col=text_col, sheet_name=sheet_name):
            tokens = simple_word_tokenize(text, lowercase=lowercase)

            if remove_function_words:
                tokens = remove_stopwords_from_tokens(tokens, stopword_set, min_word_len)

            if not tokens:
                continue

            counter.update(tokens)
            n_texts += 1
            n_tokens += len(tokens)

    return {
        "name": corpus.name,
        "paths": corpus.paths,
        "counter": counter,
        "n_texts": n_texts,
        "n_tokens": n_tokens,
        "n_types": len(counter),
    }


def top_words_to_dataframe(corpus_data: dict, top_k: int) -> pd.DataFrame:
    rows = []
    total_tokens = corpus_data["n_tokens"]

    for rank, (word, count) in enumerate(corpus_data["counter"].most_common(top_k), start=1):
        rows.append({
            "corpus_name": corpus_data["name"],
            "corpus_paths": " | ".join(corpus_data["paths"]),
            "rank": rank,
            "word": word,
            "count": count,
            "relative_frequency": count / total_tokens if total_tokens > 0 else 0.0,
            "n_texts": corpus_data["n_texts"],
            "n_tokens": corpus_data["n_tokens"],
            "n_types": corpus_data["n_types"],
        })

    return pd.DataFrame(rows)


# =========================================================
# Argument parser
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract top-k words from one or more corpora."
    )

    parser.add_argument("--paths", type=str, nargs="+", default=None)
    parser.add_argument("--names", type=str, nargs="+", default=None)
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional CSV with columns name,path. Multiple rows with same name are merged.",
    )

    parser.add_argument("--text_col", type=str, default=None)
    parser.add_argument("--sheet", default=0)

    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--no_lowercase", action="store_true")
    parser.add_argument("--remove_function_words", action="store_true")
    parser.add_argument("--remove_stopwords", action="store_true")
    parser.add_argument("--stopword_library", type=str, default="nltk", choices=["nltk", "sklearn"])
    parser.add_argument("--min_word_len", type=int, default=3)

    parser.add_argument("--output_csv", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    specs = make_corpus_specs(paths=args.paths, names=args.names, manifest=args.manifest)
    remove_function_words = args.remove_function_words or args.remove_stopwords

    if remove_function_words:
        stopword_set = load_stopwords(args.stopword_library)
    else:
        stopword_set = None

    frames = []

    for spec in specs:
        corpus_data = build_word_counter(
            corpus=spec,
            text_col=args.text_col,
            sheet_name=args.sheet,
            lowercase=not args.no_lowercase,
            remove_function_words=remove_function_words,
            min_word_len=args.min_word_len,
            stopword_set=stopword_set,
        )
        frames.append(top_words_to_dataframe(corpus_data, args.top_k))

    output_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved top-word results to: {args.output_csv}")
    print("\nPreview:")
    print(output_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
