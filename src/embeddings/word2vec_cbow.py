from gensim.models import Word2Vec
import numpy as np

def train_word2vec_cbow(tokenized_texts, vector_size=100, window=5, min_count=2):
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0,          # CBOW
        workers=4
    )
    return model

def build_embedding_matrix(word_index, w2v_model, vector_size):
    embedding_matrix = np.zeros((len(word_index) + 1, vector_size))
    for word, idx in word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
    return embedding_matrix
