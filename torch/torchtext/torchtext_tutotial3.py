'''
Description: 
Author: LJJ
Date: 2022-04-05 16:17:08
LastEditTime: 2022-04-05 16:40:01
LastEditors: LJJ
'''
from pickletools import optimize
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torchtext.data import Field, TabularDataset,BucketIterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spacy_en = spacy.load("en")

def tokenize(text):
  return [tok.text for tok in spacy_en.tokenizer(text)]

quote = Field(sequential=True,use_vocab=True,tokenize=tokenize,lower=True)
score = Field(sequential=False,use_vocab=False)

fields = {"quote" : ("q",quote), "score" : ("s", score)}

train_data,test_data = TabularDataset.splits(
  path="mydata",train="train.json",test="test.json",format="json",fields=fields
)

quote.build_vocab(train_data,max_size=10000,min_freq=1,vectors="glove.6B.100d")
train_iterator,test_iterator = BucketIterator.splits(
  (train_data,test_data),batch_size=2,device=device
)

class RNN_LSTM(nn.Module):
  def __init__(self,input_size,embed_size,hidden_size,num_layers):
    super(RNN_LSTM,self).__init__()
    self.hidden_size = hidden_size()
    self.num_layers = num_layers
    
    self.embedding = nn.Embedding(input_size,embed_size)
    self.rnn = nn.LSTM(embed_size,hidden_size,num_layers)
    self.fc_out = nn.Linear(hidden_size,1)
    
  def forward(self,x):
    h0 = torch.zeros(self.num_layers,x.size(1),self.hidden_size).to(device)
    c0 = torch.zeros(self.num_layers,x.size(1),self.hidden_size).to(device)
    embedded = self.embedding(x)
    outputs, _ = self.rnn(embedded,(h0,c0))
    prediction = self.fc_out(outputs[-1,:,:])
    return prediction
    
    
input_size = len(quote.vocab)
hidden_size = 512
num_layers = 2
embedding_size = 100
learning_rate = 0.005
num_epochs = 10

model = RNN_LSTM(input_size, embedding_size, hidden_size,num_layers).to(device)

pretrained_embeddings = quote.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)


for epoch in range(num_epochs):
  for batch_idx, batch in enumerate(train_iterator):
    data = batch.q.to(device=device)
    target = batch.s.to(device=device)
    
    scores = model(data)
    loss = criterion(scores.squeeze(1),target.type_as(scores))
    
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    