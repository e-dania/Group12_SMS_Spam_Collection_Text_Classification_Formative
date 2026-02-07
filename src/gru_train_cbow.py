import os
import sys
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embeddings.word2vec_cbow import train_word2vec_cbow, build_embedding_matrix
from models.gru_model import build_gru_model

MAX_LEN = 50
EMBED_DIM = 100
EPOCHS = 10
BATCH_SIZE = 32

def main():
    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")

    train_df["clean_text"] = train_df["clean_text"].fillna("").astype(str)
    val_df["clean_text"] = val_df["clean_text"].fillna("").astype(str)

    train_df = train_df[train_df["clean_text"].str.strip() != ""]
    val_df = val_df[val_df["clean_text"].str.strip() != ""]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df["clean_text"])

    X_train = tokenizer.texts_to_sequences(train_df["clean_text"])
    X_val = tokenizer.texts_to_sequences(val_df["clean_text"])

    X_train = pad_sequences(X_train, maxlen=MAX_LEN)
    X_val = pad_sequences(X_val, maxlen=MAX_LEN)

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    tokenized_texts = [text.split() for text in train_df["clean_text"]]
    w2v_model = train_word2vec_cbow(tokenized_texts, EMBED_DIM)

    embedding_matrix = build_embedding_matrix(
        tokenizer.word_index,
        w2v_model,
        EMBED_DIM
    )

    model = build_gru_model(
        vocab_size=len(tokenizer.word_index) + 1,
        embedding_dim=EMBED_DIM,
        embedding_matrix=embedding_matrix,
        max_len=MAX_LEN
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    es = EarlyStopping(patience=2, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[es],
        verbose=1
    )

    model.save("results/gru_cbow_model.h5")
    print("GRU CBOW model saved")

if __name__ == "__main__":
    main()
