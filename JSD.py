import argparse
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
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
    Load a corpus manifest CSV.

    Required columns:
    - name: corpus/split name, e.g. EFCAMDAT_A1A2
    - path: file or directory path

    Multiple rows can share the same name. Those paths will be merged into
    one corpus. This is useful when one proficiency split is stored across
    multiple files.
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


def make_corpus_specs(
    paths: Optional[List[str]],
    names: Optional[List[str]],
    manifest: Optional[str],
    argument_name: str,
) -> List[CorpusSpec]:
    """
    Build CorpusSpec objects either from:
    1. a manifest CSV; or
    2. a list of paths, optionally with matching names.
    """
    if manifest is not None:
        return load_manifest(manifest)

    if not paths:
        raise ValueError(f"You must provide either --{argument_name}_paths or --{argument_name}_manifest.")

    if names is not None and len(names) != len(paths):
        raise ValueError(
            f"--{argument_name}_names must have the same length as --{argument_name}_paths. "
            f"Got {len(names)} names and {len(paths)} paths."
        )

    specs = []
    for i, path in enumerate(paths):
        name = names[i] if names is not None else default_corpus_name(path)
        specs.append(CorpusSpec(name=name, paths=[path]))

    return specs


# =========================================================
# Lightweight stopword removal
# =========================================================

def load_stopwords(stopword_library: str = "nltk") -> set:
    """
    Load English stopwords using a lightweight library.

    Recommended:
    - nltk: widely used and transparent
    - sklearn: no corpus download needed, useful fallback
    """
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


def remove_stopwords_from_tokens(tokens: List[str], stopword_set: set, min_word_len: int = 3) -> List[str]:
    return [
        token for token in tokens
        if token.lower() not in stopword_set and len(token) >= min_word_len
    ]


# =========================================================
# Tokenization
# =========================================================

def simple_word_tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Simple word tokenizer.

    Keeps:
    - alphabetic words
    - apostrophe forms, e.g. don't, students'

    Removes:
    - numbers
    - punctuation
    - symbols
    """
    if not isinstance(text, str):
        return []

    if lowercase:
        text = text.lower()

    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)


# =========================================================
# File loading
# =========================================================

def find_supported_files(input_path: str) -> List[Path]:
    """
    Accept a single file or a directory.

    If directory, recursively find:
    - .txt
    - .csv
    - .xlsx
    - .xls
    """
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
    """
    Yield texts from .txt, .csv, .xlsx, or .xls files.
    """
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

        # Let pandas choose the appropriate engine for older .xls files.
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

    # Keep deterministic order and remove duplicates.
    return sorted(set(all_files))


# =========================================================
# Word counter
# =========================================================

def build_word_counter(
    corpus: CorpusSpec,
    text_col: Optional[str] = None,
    sheet_name=0,
    lowercase: bool = True,
    remove_function_words: bool = False,
    min_word_len: int = 3,
    stopword_set: Optional[set] = None,
) -> dict:
    """
    Build word-frequency Counter from a corpus specification.
    """
    files = collect_files(corpus.paths)

    counter = Counter()
    n_texts = 0
    n_tokens = 0

    print(f"\nReading corpus: {corpus.name}")
    print(f"Input path(s): {corpus.paths}")
    print(f"Found {len(files)} supported file(s).")

    for file_path in files:
        print(f"Processing: {file_path}")

        for text in iter_texts_from_file(
            file_path=file_path,
            text_col=text_col,
            sheet_name=sheet_name,
        ):
            tokens = simple_word_tokenize(text, lowercase=lowercase)

            if remove_function_words:
                tokens = remove_stopwords_from_tokens(
                    tokens=tokens,
                    stopword_set=stopword_set,
                    min_word_len=min_word_len,
                )

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


# =========================================================
# Jensen-Shannon Divergence
# =========================================================

def counter_to_probability_vector(counter: Counter, vocab: List[str], smoothing: float = 1e-12) -> np.ndarray:
    counts = np.array([counter.get(word, 0) for word in vocab], dtype=np.float64)
    counts = counts + smoothing
    return counts / counts.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    KL divergence KL(p || q), using log base 2.
    """
    return float(np.sum(p * np.log2(p / q)))


def js_divergence(counter_a: Counter, counter_b: Counter, smoothing: float = 1e-12) -> dict:
    """
    Jensen-Shannon Divergence between two word-frequency distributions.

    Lower = more similar.
    Higher = more different.

    With log base 2, JSD is bounded between 0 and 1.
    """
    vocab = sorted(set(counter_a.keys()) | set(counter_b.keys()))

    if not vocab:
        raise ValueError("Cannot compute JSD because the union vocabulary is empty.")

    p = counter_to_probability_vector(counter_a, vocab, smoothing=smoothing)
    q = counter_to_probability_vector(counter_b, vocab, smoothing=smoothing)
    m = 0.5 * (p + q)

    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    return {
        "jsd_full_vocab": jsd,
        "js_distance_full_vocab": float(np.sqrt(jsd)),
        "full_vocab_size": len(vocab),
    }


# =========================================================
# Batch comparison
# =========================================================

def compute_batch_jsd(
    pretraining_specs: List[CorpusSpec],
    downstream_specs: List[CorpusSpec],
    pretraining_text_col: Optional[str],
    downstream_text_col: Optional[str],
    pretraining_sheet=0,
    downstream_sheet=0,
    lowercase: bool = True,
    remove_function_words: bool = False,
    min_word_len: int = 3,
    stopword_library: str = "nltk",
    smoothing: float = 1e-12,
) -> pd.DataFrame:
    if remove_function_words:
        stopword_set = load_stopwords(stopword_library)
    else:
        stopword_set = None

    print("\nSettings")
    print(f"Lowercase: {lowercase}")
    print(f"Stopword removal: {remove_function_words}")
    print(f"Stopword library: {stopword_library if remove_function_words else None}")
    print(f"Minimum word length: {min_word_len if remove_function_words else None}")
    print(f"Smoothing: {smoothing}")

    pretraining_data = [
        build_word_counter(
            corpus=spec,
            text_col=pretraining_text_col,
            sheet_name=pretraining_sheet,
            lowercase=lowercase,
            remove_function_words=remove_function_words,
            min_word_len=min_word_len,
            stopword_set=stopword_set,
        )
        for spec in pretraining_specs
    ]

    downstream_data = [
        build_word_counter(
            corpus=spec,
            text_col=downstream_text_col,
            sheet_name=downstream_sheet,
            lowercase=lowercase,
            remove_function_words=remove_function_words,
            min_word_len=min_word_len,
            stopword_set=stopword_set,
        )
        for spec in downstream_specs
    ]

    rows = []

    for pretrain in pretraining_data:
        for downstream in downstream_data:
            print(f"\nComputing JSD: {pretrain['name']} vs {downstream['name']}")

            jsd_result = js_divergence(
                counter_a=pretrain["counter"],
                counter_b=downstream["counter"],
                smoothing=smoothing,
            )

            rows.append({
                "pretraining_name": pretrain["name"],
                "pretraining_paths": " | ".join(pretrain["paths"]),
                "downstream_name": downstream["name"],
                "downstream_paths": " | ".join(downstream["paths"]),

                "lowercase": lowercase,
                "remove_function_words": remove_function_words,
                "stopword_library": stopword_library if remove_function_words else None,
                "min_word_len": min_word_len if remove_function_words else None,
                "smoothing": smoothing,

                "pretraining_n_texts": pretrain["n_texts"],
                "pretraining_n_tokens": pretrain["n_tokens"],
                "pretraining_n_types": pretrain["n_types"],

                "downstream_n_texts": downstream["n_texts"],
                "downstream_n_tokens": downstream["n_tokens"],
                "downstream_n_types": downstream["n_types"],

                "full_vocab_size": jsd_result["full_vocab_size"],
                "jsd_full_vocab": jsd_result["jsd_full_vocab"],
                "js_distance_full_vocab": jsd_result["js_distance_full_vocab"],
            })

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(
        by=["downstream_name", "jsd_full_vocab", "pretraining_name"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return result_df


# =========================================================
# Argument parser
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Jensen-Shannon Divergence between multiple pretraining "
            "proficiency splits and one or more downstream datasets."
        )
    )

    # Pretraining corpora
    parser.add_argument(
        "--pretraining_paths",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Pretraining split paths. Each path is treated as one corpus/split. "
            "Each path can be a file or directory."
        ),
    )
    parser.add_argument(
        "--pretraining_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional names matching --pretraining_paths, e.g. A1A2 B1B2 B2C1.",
    )
    parser.add_argument(
        "--pretraining_manifest",
        type=str,
        default=None,
        help=(
            "Optional CSV with columns name,path. Use this when one split "
            "contains multiple files."
        ),
    )

    # Downstream corpora
    parser.add_argument(
        "--downstream_paths",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Downstream dataset paths. Each path is treated as one corpus. "
            "Each path can be a file or directory."
        ),
    )
    parser.add_argument(
        "--downstream_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional names matching --downstream_paths, e.g. IELTS FCE.",
    )
    parser.add_argument(
        "--downstream_manifest",
        type=str,
        default=None,
        help="Optional CSV with columns name,path for downstream corpora.",
    )

    # Column/sheet settings
    parser.add_argument("--pretraining_text_col", type=str, default=None)
    parser.add_argument("--downstream_text_col", type=str, default=None)
    parser.add_argument("--pretraining_sheet", default=0)
    parser.add_argument("--downstream_sheet", default=0)

    # Processing settings
    parser.add_argument("--no_lowercase", action="store_true")
    parser.add_argument("--remove_function_words", action="store_true")
    parser.add_argument("--remove_stopwords", action="store_true")
    parser.add_argument("--stopword_library", type=str, default="nltk", choices=["nltk", "sklearn"])
    parser.add_argument("--min_word_len", type=int, default=3)
    parser.add_argument("--smoothing", type=float, default=1e-12)

    # Output
    parser.add_argument("--output_csv", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    pretraining_specs = make_corpus_specs(
        paths=args.pretraining_paths,
        names=args.pretraining_names,
        manifest=args.pretraining_manifest,
        argument_name="pretraining",
    )

    downstream_specs = make_corpus_specs(
        paths=args.downstream_paths,
        names=args.downstream_names,
        manifest=args.downstream_manifest,
        argument_name="downstream",
    )

    remove_function_words = args.remove_function_words or args.remove_stopwords

    result_df = compute_batch_jsd(
        pretraining_specs=pretraining_specs,
        downstream_specs=downstream_specs,
        pretraining_text_col=args.pretraining_text_col,
        downstream_text_col=args.downstream_text_col,
        pretraining_sheet=args.pretraining_sheet,
        downstream_sheet=args.downstream_sheet,
        lowercase=not args.no_lowercase,
        remove_function_words=remove_function_words,
        min_word_len=args.min_word_len,
        stopword_library=args.stopword_library,
        smoothing=args.smoothing,
    )

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    result_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved JSD results to: {args.output_csv}")

    print("\nPreview:")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
