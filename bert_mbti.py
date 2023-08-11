from torch import nn
from transformers import BertModel
import numpy as np
import os
from torch.optim import Adam
from tqdm import tqdm
from bert_dataset import Dataset
import torch 
from transformers import BertTokenizer
from bert_dataset import labels
tokenizer = BertTokenizer.from_pretrained('./tokenizer')
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('./tokenizer')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


class MBTINode():
    def __init__(self,model_name,type_idx):
        self.model_name = model_name
        self.type_idx = type_idx
        self.module = BertClassifier()
    
    def loadmodule(self):
        print('Load Model:',self.model_name,flush=True,end='')

        self.module.load_state_dict(torch.load(self.model_name,map_location=torch.device('cpu')))
        # quantized_model = torch.quantization.quantize_dynamic(
        # self.module, {torch.nn.Linear}, dtype=torch.qint8
        # )
        
        print('[OK]')
    def predict(self,test_input) -> (int,float,float):
        device = torch.device("cpu")
        mask = test_input['attention_mask'].to(device)
        input_id = test_input['input_ids'].squeeze(1).to(device)
        ouput = self.module(input_id,mask)
        out2 = ouput.argmax(dim=1)
        return (out2,ouput[0][0],ouput[0][1])
    
    def score(self,test_data):
        test = Dataset(test_data)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model = self.module
        if use_cuda:
            model = model.cuda()

        total_acc_test = 0
        with torch.no_grad():
            idx = 1
            for test_input, test_label in test_dataloader:
                    test_label = test_label.to(device)[0]
                    mask = test_input['attention_mask'].to(device)
                    input_id = test_input['input_ids'].squeeze(1).to(device)
                    output = model(input_id, mask)
                    #print("claim:",output.argmax(dim=1),"fact:",test_label)
                    out2=output.argmax(dim=1)
                    acc = (out2 == test_label[self.type_idx]).sum().item()
                    if acc == 0:
                        print('claim:',out2,'fact:',test_label[self.type_idx])
                    idx+=1
                    if idx % 100 == 0:
                        print('cur is:',idx)
                    total_acc_test += acc   
        print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
        
        
class MBTIMain():
  def __init__(self) -> None:
      self.ie = MBTINode('./out/ie.pt',0)
      self.sn = MBTINode('./out/sn.pt',1)
      self.tf = MBTINode('./out/tf.pt',2)
      self.jp = MBTINode('./out/jp.pt',3)
      


  def loadmodule(self):
      self.ie.loadmodule()
      self.sn.loadmodule()
      self.tf.loadmodule()
      self.jp.loadmodule()
  
  def score(self,test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    total_acc_test = 0
    with torch.no_grad():
        idx = 1
        for test_input, test_label in test_dataloader:
            test_label = test_label[0]
            claim = self.predict(test_input)
            fact = ('E' if test_label[0] else 'I')+('N' if test_label[1] else 'S')+('F' if test_label[2] else 'T')+('P' if test_label[3] else 'J')
                    
            idx+=1
            if idx % 100 ==0:
                print('cur idx:',idx)
            if claim != fact:
                print('claim:',claim,'fact:',fact)
  
  def predict(self,text):
    token = tokenizer(text,padding='max_length',truncation=True,max_length=512,return_tensors='pt')
    ie,i_v,e_v = self.ie.predict(token)
    sn,s_v,n_v = self.sn.predict(token)
    tf,t_v,f_v = self.tf.predict(token)
    jp,j_v,p_v = self.jp.predict(token)

    return (('E' if ie else 'I')+('N' if sn else 'S')+('F' if tf else 'T')+('P' if jp else 'J')),0,0,0,0

# node = MBTIMain()
# print('model init')
# node.loadmodule()
# print('loaded ok')
# print(node.predict("happy!!!!"))
# datas = np.array(torch.load(os.path.join('../out', "precoded3.bin"),map_location=torch.device('cpu')))
# d_train,d_val,d_test = np.split(datas,[int(.8*len(datas)),int(.9*len(datas))])
# print(len(d_train),len(d_val),len(d_test))
# EPOCHS = 4
# node = MBTIMain()
# node.loadmodule()
# LR = 1e-5
# #print(d_train[:2])
# # 量化的网络层为所有的nn.Linear的权重，使其成为int8
# # quantized_model = torch.quantization.quantize_dynamic(
# #     model, {torch.nn.Linear}, dtype=torch.qint8
# # )
 
# # 打印动态量化后的BERT模型
# #torch.save(quantized_model.state_dict(),'./out/model_IE_c.pt')
# #train(model, d_train, d_val, LR, EPOCHS)
# print(node.score(d_test))