import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from preprocessing.preprocess import load_and_preprocess

df = load_and_preprocess("../data/SMSSpamCollection")

sentences = [text.split() for text in df['clean_text']]

model_w2v = Word2Vec.load("embeddings/word2vec_cbow.model")

word_index = {word: i+1 for i, word in enumerate(model_w2v.wv.index_to_key)}

sequences = [[word_index.get(w, 0) for w in s] for s in sentences]
X = pad_sequences(sequences, maxlen=50)

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

embedding_matrix = np.zeros((len(word_index)+1, 100))

for word, i in word_index.items():
    embedding_matrix[i] = model_w2v.wv[word]

model = Sequential([
    Embedding(len(word_index)+1, 100,
              weights=[embedding_matrix],
              input_length=50,
              trainable=False),
    LSTM(64),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=32)

print(model.evaluate(X_test, y_test))
