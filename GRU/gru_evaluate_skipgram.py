import os
import sys
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


MAX_LEN = 50
MODEL_PATH = "results/gru_skipgram_model.h5"
OUT_METRICS = "results/metrics_gru_skipgram.csv"


def load_splits():
    train_df = pd.read_csv("data/splits/train.csv")
    test_df = pd.read_csv("data/splits/test.csv")

    train_df["clean_text"] = train_df["clean_text"].fillna("").astype(str)
    test_df["clean_text"] = test_df["clean_text"].fillna("").astype(str)

    train_df = train_df[train_df["clean_text"].str.strip() != ""]
    test_df = test_df[test_df["clean_text"].str.strip() != ""]

    return train_df, test_df


def build_tokenizer(train_texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def main():
    train_df, test_df = load_splits()

    tokenizer = build_tokenizer(train_df["clean_text"])

    X_test = tokenizer.texts_to_sequences(test_df["clean_text"])
    X_test = pad_sequences(X_test, maxlen=MAX_LEN)

    y_test = test_df["label"].values

    model = load_model(MODEL_PATH)

    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

    metrics_df = pd.DataFrame([{
        "model": "GRU",
        "embedding": "Word2Vec Skip-gram",
        "max_len": MAX_LEN,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }])

    os.makedirs("results", exist_ok=True)
    metrics_df.to_csv(OUT_METRICS, index=False)
    print(f"\nSaved metrics to {OUT_METRICS}")


if __name__ == "__main__":
    main()
