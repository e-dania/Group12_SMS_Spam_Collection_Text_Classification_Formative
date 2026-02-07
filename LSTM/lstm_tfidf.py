from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from preprocessing.preprocess import load_and_preprocess
import numpy as np

df = load_and_preprocess("../data/SMSSpamCollection")

X = df['clean_text']
y = df['label']

vectorizer = TfidfVectorizer(max_features=300)
X_tfidf = vectorizer.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

model = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[2])),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=32)

loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)
