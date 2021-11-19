import pandas as pd
import torch
from transformers import BertTokenizer
from model import BERT
def tokenize(X):
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    pad_size=50
    res=[]
    for text in X:
        text=text.replace('\\',' ')
        tokens=tokenizer(text,max_length=50)
        if len(tokens['input_ids'])<pad_size:
            tokens['input_ids']+=([0]*(pad_size-len(tokens['input_ids'])))
            tokens['token_type_ids']+=([0]*(pad_size-len(tokens['token_type_ids'])))
            tokens['attention_mask']+=([0]*(pad_size-len(tokens['attention_mask'])))
        res.append([tokens['input_ids'],tokens['token_type_ids'],tokens['attention_mask']])
    res=torch.tensor(res)
    return res

raw_train=pd.read_csv('data/ag_news_csv/train.csv',names=['label','title','text'])
raw_test=pd.read_csv('data/ag_news_csv/test.csv',names=['label','title','text'])
X_train,Y_train=list(raw_train['text'])[:10000],torch.tensor(list(raw_train['label'])[:10000])
X_test, Y_test = list(raw_test['text']), torch.tensor(list(raw_test['label']))
X_train=tokenize(X_train)
X_test=tokenize(X_test)
Y_train=Y_train-1
Y_test=Y_test-1

model=BERT()
