import os
import random
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

DATA_DIR = "data/splits"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_split(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{name}.csv")

    # Read CSV with proper quoting to handle commas in text fields
    df = pd.read_csv(path, quotechar='"', skipinitialspace=True)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Normalize column names / shape
    if "label" in df.columns and ("text" in df.columns or "clean_text" in df.columns):
        text_col = "text" if "text" in df.columns else "clean_text"
        df = df[["label", text_col]].copy()
        df = df.rename(columns={text_col: "text"})
    else:
        raise ValueError(f"CSV {path} missing expected columns. Found: {list(df.columns)}")

    df = df.dropna().reset_index(drop=True)

    # label mapping: convert 'ham'/'spam' strings OR keep numeric labels
    if df["label"].dtype == object:
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["label"] = df["label"].astype(int)

    return df

def tokenize(text: str):
    # Keep it consistent with your preprocessing later; this is a safe baseline
    return str(text).lower().split()

def build_rnn(input_shape, units=64, dropout=0.3, lr=1e-3):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.SimpleRNN(units, dropout=dropout),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def eval_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm

def vectorize_texts(token_lists, w2v, max_len, embed_dim):
    X = np.zeros((len(token_lists), max_len, embed_dim), dtype="float32")
    for i, toks in enumerate(token_lists):
        vecs = []
        for t in toks:
            if t in w2v.wv:
                vecs.append(w2v.wv[t])
        if len(vecs) == 0:
            continue
        vecs = np.array(vecs, dtype="float32")[:max_len]
        X[i, :vecs.shape[0], :] = vecs
    return X

def main():
    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test")

    train_tok = [tokenize(t) for t in train_df["text"]]
    val_tok   = [tokenize(t) for t in val_df["text"]]
    test_tok  = [tokenize(t) for t in test_df["text"]]

    # Word2Vec Skip-gram
    embed_dim = 100
    w2v = Word2Vec(
        sentences=train_tok,
        vector_size=embed_dim,
        window=5,
        min_count=1,
        workers=4,
        sg=1,              # <-- skip-gram
        seed=RANDOM_SEED,
        epochs=10
    )

    max_len = 30  # SMS are short; 20–40 usually works
    X_train = vectorize_texts(train_tok, w2v, max_len, embed_dim)
    X_val   = vectorize_texts(val_tok, w2v, max_len, embed_dim)
    X_test  = vectorize_texts(test_tok, w2v, max_len, embed_dim)

    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    grid = [
        {"units": 32, "dropout": 0.2, "lr": 1e-3},
        {"units": 64, "dropout": 0.3, "lr": 1e-3},
        {"units": 64, "dropout": 0.4, "lr": 5e-4},
    ]

    best = {"val_f1": -1, "cfg": None, "model": None}
    es = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    for cfg in grid:
        model = build_rnn(input_shape=(max_len, embed_dim), **cfg)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=15,
            batch_size=64,
            callbacks=[es],
            verbose=1
        )
        val_prob = model.predict(X_val, verbose=0).ravel()
        _, _, _, val_f1, _ = eval_metrics(y_val, val_prob)

        if val_f1 > best["val_f1"]:
            best = {"val_f1": val_f1, "cfg": cfg, "model": model}

    model = best["model"]
    test_prob = model.predict(X_test, verbose=0).ravel()
    acc, prec, rec, f1, cm = eval_metrics(y_test, test_prob)

    model_path = os.path.join(RESULTS_DIR, "rnn_skipgram_model.h5")
    model.save(model_path)

    metrics_path = os.path.join(RESULTS_DIR, "metrics_rnn_skipgram.csv")
    pd.DataFrame([{
        "embedding": "word2vec_skipgram",
        "max_len": max_len,
        "embed_dim": embed_dim,
        "best_units": best["cfg"]["units"],
        "best_dropout": best["cfg"]["dropout"],
        "best_lr": best["cfg"]["lr"],
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "cm_tn": int(cm[0,0]),
        "cm_fp": int(cm[0,1]),
        "cm_fn": int(cm[1,0]),
        "cm_tp": int(cm[1,1]),
        "model_path": model_path
    }]).to_csv(metrics_path, index=False)

    print("Best config:", best["cfg"])
    print("Saved:", model_path)
    print("Saved:", metrics_path)
    print("Test F1:", f1)

if __name__ == "__main__":
    main()
