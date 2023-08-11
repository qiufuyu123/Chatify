import torch
import numpy as np
from transformers import BertTokenizer
import os
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
cute_labels={'ISTJ':'沉稳（白尾海雕）',
          'ISFJ':'温和（贝弗伦兔）',
          'INFJ':'沉稳（白尾海雕）',
          'INFP':'温和（贝弗伦兔）',
          'ESTJ':'勇敢（阿穆尔虎）',
          'ESFJ':'温和（贝弗伦兔）',
          'ENFJ':'开朗（萨摩耶）',
          'ENFP':'开朗（萨摩耶）',
          'ISTP':'感性浪漫（白唇鹿）',
          'ISFP':'感性浪漫（白唇鹿）',
          'INTJ':'沉稳（白尾海雕）',
          'INTP':'沉稳（白尾海雕）',
          'ESTP':'勇敢（阿穆尔虎）',
          'ESFP':'开朗（萨摩耶）',
          'ENTJ':'勇敢（阿穆尔虎）',
          'ENTP':'沉稳（白尾海雕）'
          }
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
    
