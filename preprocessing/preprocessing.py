import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def load_and_preprocess(path):
    # TAB-separated file
    df = pd.read_csv(path, sep='\t', header=None, names=['label', 'text'])

    df['clean_text'] = df['text'].apply(clean_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    return df
