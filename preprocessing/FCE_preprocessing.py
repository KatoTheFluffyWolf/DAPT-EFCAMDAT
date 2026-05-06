import xml.etree.ElementTree as ET
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import argparse

# Map single holistic score to 0 - 20 scale for AES
FCE_SCORE_MAP = {
    "0": 0,
    "1.1": 1,
    "1.2": 4,
    "1.3": 7,
    "2.1": 9,
    "2.2": 10,
    "2.3": 11,
    "2.3T": 11.5,
    "3.1": 12,
    "3.2": 13,
    "3.3": 14,
    "4.1": 15,
    "4.2": 16,
    "4.3": 17,
    "5.1": 18,
    "5.2": 19,
    "5.3": 20,
}

def restore_original_text(node):
    parts = []

    if node.text:
        parts.append(node.text)

    for child in node:
        tag = child.tag.split("}", 1)[-1]

        if tag == "NS":
            for sub in child:
                if sub.tag.split("}", 1)[-1] == "i":
                    parts.append(restore_original_text(sub))
                    break
        else:
            parts.append(restore_original_text(child))

        if child.tail:
            parts.append(child.tail)

    return "".join(parts)

def extract_rows_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []
    script_id = Path(xml_path).stem   # safer grouping key

    for answer_tag in ["answer1", "answer2"]:
        answer_node = root.find(f".//{answer_tag}")
        if answer_node is None:
            continue

        coded_answer = answer_node.find("coded_answer")
        answer_score = answer_node.findtext("exam_score")

        if answer_score not in FCE_SCORE_MAP:
            continue
        if coded_answer is None:
            continue

        text = restore_original_text(coded_answer)
        text = re.sub(r"\s+", " ", text).strip()

        rows.append({
            "script_id": script_id,
            "file_name": Path(xml_path).name,
            "answer_id": answer_tag,
            "text": text,
            "answer-s": answer_score,
            "mapped_score": FCE_SCORE_MAP.get(answer_score)
        })

    return rows

def folder_to_df(folder_path):
    all_rows = []

    for xml_file in Path(folder_path).rglob("*.xml"):
        try:
            rows = extract_rows_from_xml(xml_file)
            all_rows.extend(rows)
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

    return pd.DataFrame(all_rows)

def global_preprocess(df):
    # cleanup that should happen before splitting
    df = df.dropna(subset=["text", "answer-s"]).copy()
    df = df.drop_duplicates(subset=["text"]).copy()
    df = df.reset_index(drop=True)
    return df

def preprocess_split(df):
    # split-specific preprocessing goes here
    # right now just reset index
    df = df.reset_index(drop=True)
    return df

def split_by_script(df, test_size=0.15, dev_size_of_remaining=0.1765, random_state=42):
    # first split: train+dev vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_dev_idx, test_idx = next(gss1.split(df, groups=df["script_id"]))

    train_dev_df = df.iloc[train_dev_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # second split: train vs dev
    gss2 = GroupShuffleSplit(n_splits=1, test_size=dev_size_of_remaining, random_state=random_state)
    train_idx, dev_idx = next(gss2.split(train_dev_df, groups=train_dev_df["script_id"]))

    train_df = train_dev_df.iloc[train_idx].copy()
    dev_df = train_dev_df.iloc[dev_idx].copy()

    return train_df, dev_df, test_df

def save_df(df, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse FCE XML files, preprocess them, split by script_id, and save CSV outputs."
    )

    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to the folder containing the raw FCE XML files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the processed CSV files."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Test split size. Default is 0.15."
    )
    parser.add_argument(
        "--dev_size_of_remaining",
        type=float,
        default=0.1765,
        help="Dev split size applied to the remaining train+dev portion. Default is 0.1765."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for GroupShuffleSplit. Default is 42."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    folder_path = args.folder_path
    output_dir = args.output_dir

    # 1. Parse XML files
    df = folder_to_df(folder_path)

    # 2. Global preprocessing before split
    df = global_preprocess(df)

    # 3. Split by script_id
    train_df, dev_df, test_df = split_by_script(
        df,
        test_size=args.test_size,
        dev_size_of_remaining=args.dev_size_of_remaining,
        random_state=args.random_state
    )

    # 4. Preprocess each split separately
    train_df = preprocess_split(train_df)
    dev_df = preprocess_split(dev_df)
    test_df = preprocess_split(test_df)

    # 5. Print summary
    print("Full dataset:", df.shape)
    print("Train:", train_df.shape)
    print("Dev:", dev_df.shape)
    print("Test:", test_df.shape)

    print("\nSample rows:")
    print(train_df.head(10))

    # 6. Check leakage
    train_scripts = set(train_df["script_id"])
    dev_scripts = set(dev_df["script_id"])
    test_scripts = set(test_df["script_id"])

    print("\nOverlap checks:")
    print("Train ∩ Dev:", len(train_scripts & dev_scripts))
    print("Train ∩ Test:", len(train_scripts & test_scripts))
    print("Dev ∩ Test:", len(dev_scripts & test_scripts))

    # 7. Save files
    save_df(df,       f"{output_dir}/full_dataset.csv")
    save_df(train_df, f"{output_dir}/train.csv")
    save_df(dev_df,   f"{output_dir}/dev.csv")
    save_df(test_df,  f"{output_dir}/test.csv")

if __name__ == "__main__":
    main()