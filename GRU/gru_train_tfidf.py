import os
import sys
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


MAX_FEATURES = 3000
EPOCHS = 10
BATCH_SIZE = 32


def main():
    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")


    train_df["clean_text"] = train_df["clean_text"].fillna("").astype(str)
    val_df["clean_text"] = val_df["clean_text"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train = vectorizer.fit_transform(train_df["clean_text"]).toarray()
    X_val = vectorizer.transform(val_df["clean_text"]).toarray()

    # reshape to (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    model = Sequential()
    model.add(GRU(64, input_shape=(1, MAX_FEATURES)))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

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

    model.save("results/gru_tfidf_model.h5")
    print("GRU TF-IDF model saved")


if __name__ == "__main__":
    main()
