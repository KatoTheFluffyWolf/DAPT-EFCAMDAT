import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import argparse

def load_dataset(path):
    ds = pd.read_csv(path)
    return ds

def remove_non_english_chars(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9\s\.,!?;:'\"()\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_dataset(ds):
    # 1 Remove missing values
    ds = ds.dropna(subset=["Essay", "Overall"], how="any")

    # 2 Remove duplicates
    ds = ds.drop_duplicates(subset=["Essay"])

    # 3 Remove unused columns
    ds = ds.drop(
        columns=[
            "Question",
            "Examiner_Commen",
            "Task_Response",
            "Coherence_Cohesion",
            "Lexical_Resource",
            "Range_Accuracy",
        ],
        errors="ignore"
    )

    # 4 Convert Overall to numeric and remove invalid values
    ds["Overall"] = pd.to_numeric(ds["Overall"], errors="coerce")
    ds = ds[(ds["Overall"] >= 1.0) & (ds["Overall"] <= 9.0)]

    # 5 Remove ultra-short texts
    ds = ds[ds["Essay"].astype(str).str.len() >= 100]

    # 6 Remove foreign language letters
    ds["Essay"] = ds["Essay"].apply(remove_non_english_chars)

    # reset index
    ds = ds.reset_index(drop=True)

    return ds

def save_df(out_path, df):
    df.to_csv(out_path, index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess IELTS dataset and create train/dev/test splits for Task 2.")

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the raw IELTS CSV file."
    )
    parser.add_argument(
        "--train_out",
        type=str,
        required=True,
        help="Path to save the Task 2 training CSV."
    )
    parser.add_argument(
        "--dev_out",
        type=str,
        required=True,
        help="Path to save the Task 2 development CSV."
    )
    parser.add_argument(
        "--test_out",
        type=str,
        required=True,
        help="Path to save the Task 2 test CSV."
    )
    parser.add_argument(
        "--task_type",
        type=int,
        default=2,
        help="Task type to filter. Default is 2."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Test split size. Default is 0.15."
    )
    parser.add_argument(
        "--dev_size",
        type=float,
        default=0.1765,
        help="Development split size applied to the remaining training portion. Default is 0.1765."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for splitting. Default is 42."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    df = load_dataset(args.input_csv)
    df = preprocess_dataset(df)

    task2 = df[df["Task_Type"] == args.task_type].reset_index(drop=True)

    # Task 2
    task2_train_df, task2_test_df = train_test_split(
        task2,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True
    )

    task2_train_df, task2_dev_df = train_test_split(
        task2_train_df,
        test_size=args.dev_size,   # dev becomes about 15% of total
        random_state=args.random_state,
        shuffle=True
    )

    
    save_df(args.train_out, task2_train_df)
    save_df(args.dev_out, task2_dev_df)
    save_df(args.test_out, task2_test_df)

    print("Task 2:", len(task2))
    print("  Train:", len(task2_train_df))
    print("  Dev:", len(task2_dev_df))
    print("  Test:", len(task2_test_df))

if __name__ == "__main__":
    main()