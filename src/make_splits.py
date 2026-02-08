import os
import sys
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROCESSED_PATH = "data/processed_sms.csv"
OUT_DIR = "data/splits"

RANDOM_STATE = 42
TEST_SIZE = 0.10   # 10% test
VAL_SIZE = 0.10    # 10% validation (from remaining 90%)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(PROCESSED_PATH)

    # Split 1: train_val vs test
    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    # Split 2: train vs val (val is 10% of original by setting relative size)
    val_relative_size = VAL_SIZE / (1 - TEST_SIZE)

    train, val = train_test_split(
        train_val,
        test_size=val_relative_size,
        random_state=RANDOM_STATE,
        stratify=train_val["label"]
    )

    train_path = os.path.join(OUT_DIR, "train.csv")
    val_path = os.path.join(OUT_DIR, "val.csv")
    test_path = os.path.join(OUT_DIR, "test.csv")

    train.to_csv("data/splits/train.csv", index=False, quoting=csv.QUOTE_ALL)
    val.to_csv("data/splits/val.csv", index=False, quoting=csv.QUOTE_ALL)
    test.to_csv("data/splits/test.csv", index=False, quoting=csv.QUOTE_ALL)


    print("Splits saved:")
    print(" -", train_path, train.shape, train["label"].value_counts().to_dict())
    print(" -", val_path, val.shape, val["label"].value_counts().to_dict())
    print(" -", test_path, test.shape, test["label"].value_counts().to_dict())

if __name__ == "__main__":
    main()
