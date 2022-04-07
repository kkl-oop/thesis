
from numpy import size
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd

def vectorization():
  sentences = LineSentence("corpus.txt")
  w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=3, workers=4)
  vocabulary = w2v_model.wv.index_to_key
  w2v_model.save("model.model")
  w2v_dict = {}
  for word in vocabulary:
    w2v_dict.update({word: w2v_model.wv[word]})
    
  x = pd.DataFrame(w2v_dict)
  x.to_csv("vec.csv", index=0)
  
  
if __name__ == '__main__':
  vectorization()



