'''
Description: 
Author: LJJ
Date: 2022-04-13 11:14:41
LastEditTime: 2022-04-13 11:30:40
LastEditors: LJJ
'''
names = []
with open("./name.txt", mode='r', encoding='utf-8') as f:
  for line in f.readlines():
    names.append(line.replace("\n", ""))
    
# print(names)
print(len(names))

unknown_words = []
with open("./compose_books.txt", mode='r',encoding='utf-8') as f:
  content = [word for word in f.read().split(" ")]
  for name in names:
    if name not in content:
      unknown_words.append(name)
      print(f"{name} is not searched")
      
      
from gensim.models import word2vec
import gensim
from torch import embedding

sentences = word2vec.Text8Corpus("./compose_books.txt")
model = gensim.models.Word2Vec(sentences,sg=1,window=5,min_count=1,negative=3,sample=0.001,hs=1,workers=4)
model.save("book.model")
model.wv.save_word2vec_format("book.model.bin", binary=True)

embedding = {}
for name in names:
  if name in unknown_words:
    continue
  else:
    name_embeding = model.wv[name]
    embedding[name] = name_embeding
    

print(len(embedding))