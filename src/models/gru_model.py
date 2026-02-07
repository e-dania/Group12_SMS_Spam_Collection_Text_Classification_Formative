from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout

def build_gru_model(vocab_size, embedding_dim, embedding_matrix=None, max_len=50):
    model = Sequential()

    if embedding_matrix is not None:
        model.add(
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                input_length=max_len,
                trainable=False
            )
        )
    else:
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))

    model.add(GRU(128))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model
