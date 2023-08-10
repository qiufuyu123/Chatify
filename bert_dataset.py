import torch
import numpy as np
from transformers import BertTokenizer
import os
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'ISTJ':[0,0,0,0],
          'ISFJ':[0,0,1,0],
          'INFJ':[0,1,1,0],
          'INFP':[0,1,1,1],
          'ESTJ':[1,0,0,0],
          'ESFJ':[1,0,1,0],
          'ENFJ':[1,1,1,0],
          'ENFP':[1,1,1,1],
          'ISTP':[0,0,0,1],
          'ISFP':[0,0,1,1],
          'INTJ':[0,1,0,0],
          'INTP':[0,1,0,1],
          'ESTP':[1,0,0,1],
          'ESFP':[1,0,1,1],
          'ENTJ':[1,1,0,0],
          'ENTP':[1,1,0,1]
          }

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label['type']] for label in df]
        self.texts = [label['code'] for label in df]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y
    
