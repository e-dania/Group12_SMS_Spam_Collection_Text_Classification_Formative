from gensim.models import Word2Vec
from preprocessing.preprocess import load_and_preprocess

df = load_and_preprocess("../data/sms_spam.csv")

sentences = [text.split() for text in df['clean_text']]

skipgram = Word2Vec(sentences, vector_size=100, window=5, sg=1, min_count=2)
cbow = Word2Vec(sentences, vector_size=100, window=5, sg=0, min_count=2)

skipgram.save("word2vec_skipgram.model")
cbow.save("word2vec_cbow.model")
