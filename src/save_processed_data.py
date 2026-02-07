import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.preprocessing import load_and_preprocess

DATA_PATH = "data/SMSSpamCollection"
OUTPUT_PATH = "data/processed_sms.csv"

def main():
    df = load_and_preprocess(DATA_PATH)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Processed dataset saved to {OUTPUT_PATH}")
    print(df.head())

if __name__ == "__main__":
    main()
