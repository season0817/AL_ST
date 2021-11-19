import pandas as pd
import numpy as np
import csv
import torch
import string
from torch.utils.data import Dataset
from transformers import BertTokenizer
PAD, CLS = '[PAD]', '[CLS]'

def get_dataset(name):
    if name=='ag_news':#4
        return get_ag_news()
    elif name=='amazon_review_full':#5
        return get_amazon_review_full()
    elif name=='dbpedia':#14
        return get_dbpedia()
    elif name=='yelp_review':#5
        return get_yelp_review_full()
    elif name=='imdb':
        return get_imdb()
    elif name=='yelp_review_polarity':
        return get_yelp_review_polarity()
    elif name=='amazon_review_polarity':
        return get_amazon_review_polarity()



def get_ag_news():
    raw_train=pd.read_csv('data/ag_news_csv/train.csv',names=['label','title','text'])
    raw_test=pd.read_csv('data/ag_news_csv/test.csv',names=['label','title','text'])
    num_classes=len(raw_train['label'].value_counts())
    X_train,Y_train = list(raw_train['text']),torch.tensor(list(raw_train['label']))
    X_test,Y_test = list(raw_test['text']), torch.tensor(list(raw_test['label']))
    X_train=tokenize(X_train,pad_size=40)
    X_test=tokenize(X_test,pad_size=40)
    Y_train=Y_train-1
    Y_test=Y_test-1
    return X_train,Y_train,X_test, Y_test,num_classes

def get_dbpedia():
    raw_train = pd.read_csv('data/dbpedia_csv/train.csv', names=['label', 'title', 'text'])
    raw_test = pd.read_csv('data/dbpedia_csv/test.csv', names=['label', 'title', 'text'])
    num_classes = len(raw_train['label'].value_counts())
    X_train, Y_train = list(raw_train['text']), torch.tensor(list(raw_train['label']))
    X_test, Y_test = list(raw_test['text']), torch.tensor(list(raw_test['label']))
    #words_num = summary_words_num(X_train)
    #print(words_num)
    X_train = tokenize(X_train,pad_size=70)#63.66
    X_test = tokenize(X_test,pad_size=70)
    Y_train = Y_train - 1
    Y_test = Y_test - 1
    return X_train, Y_train, X_test, Y_test, num_classes

def get_yelp_review_full():
    raw_train = pd.read_csv('data/yelp_review_full_csv/train.csv', names=['label', 'text'])
    raw_test = pd.read_csv('data/yelp_review_full_csv/test.csv', names=['label', 'text'])
    num_classes = len(raw_train['label'].value_counts())
    X_train, Y_train = list(raw_train['text']), torch.tensor(list(raw_train['label']))
    X_test, Y_test = list(raw_test['text']), torch.tensor(list(raw_test['label']))
    #pad_size = summary_words_num(X_train)
    #print(pad_size)#174
    X_train = tokenize(X_train,pad_size=180)
    X_test = tokenize(X_test,pad_size=180)
    Y_train = Y_train - 1
    Y_test = Y_test - 1
    return X_train, Y_train, X_test, Y_test, num_classes

def get_imdb():
    raw_train = pd.read_csv('data/imdb_csv/imdb_train.csv', names=['id','text', 'label'])
    raw_test = pd.read_csv('data/imdb_csv/imdb_test.csv', names=['id','text', 'label'])
    num_classes = len(raw_train['label'].value_counts())
    X_train, Y_train = list(raw_train['text']), torch.tensor([int(i) for i in list(raw_train['label'])])
    X_test, Y_test = list(raw_test['text']),torch.tensor([int(i) for i in list(raw_test['label'])])
    X_train = tokenize(X_train, pad_size=235)#235
    X_test = tokenize(X_test, pad_size=235)#235
    Y_train = Y_train
    Y_test = Y_test
    return X_train, Y_train, X_test, Y_test, num_classes

def get_yelp_review_polarity():
    raw_train=pd.read_csv('data/yelp_review_polarity_csv/train.csv',names=['label','text'])
    raw_test=pd.read_csv('data/yelp_review_polarity_csv/test.csv',names=['label','text'])
    num_classes=len(raw_train['label'].value_counts())
    X_train,Y_train=list(raw_train['text']),torch.tensor(list(raw_train['label']))
    X_test,Y_test=list(raw_test['text']),torch.tensor(list(raw_test['label']))
    X_train=tokenize(X_train,pad_size=180)
    X_test=tokenize(X_test,pad_size=180)
    Y_train=Y_train-1
    Y_test=Y_test-1
    return X_train,Y_train,X_test,Y_test,num_classes

def get_amazon_review_full():
    raw_train=pd.read_csv('data/amazon_review_full_csv/train.csv',names=['label','title','text'])
    raw_test=pd.read_csv('data/amazon_review_full_csv/test.csv',names=['label','title','text'])
    num_classes=len(raw_train['label'].value_counts())
    X_train,Y_train=list(raw_train['text']),torch.tensor(list(raw_train['label']))
    X_test, Y_test = list(raw_test['text']), torch.tensor(list(raw_test['label']))
    X_train=tokenize(X_train,pad_size=100)
    X_test=tokenize(X_test,pad_size=100)
    Y_train=Y_train-1
    Y_test=Y_test-1
    return X_train,Y_train,X_test, Y_test,num_classes

def get_amazon_review_polarity():
    raw_train=pd.read_csv('data/amazon_review_polarity_csv/train.csv',names=['label','title','text'])
    raw_test=pd.read_csv('data/amazon_review_polarity_csv/test.csv',names=['label','title','text'])
    num_classes=len(raw_train['label'].value_counts())
    X_train,Y_train=list(raw_train['text']),torch.tensor(list(raw_train['label']))
    X_test,Y_test=list(raw_test['text']),torch.tensor(list(raw_test['label']))
    X_train=tokenize(X_train,pad_size=100)
    X_test=tokenize(X_test,pad_size=100)
    Y_train=Y_train-1
    Y_test=Y_test-1
    return X_train,Y_train,X_test,Y_test,num_classes

def get_handler():
    return DataHandler

def summary_words_num(X):
    tokenizer=BertTokenizer.from_pretrained('./bert-base-uncased')
    lens=[]
    for text in X:
        tokens=tokenizer(text)
        lens.append(len(tokens['input_ids']))
    return np.mean(lens)


def tokenize(X,pad_size):
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    pad_size=pad_size
    res=[]
    remove=str.maketrans('','',string.punctuation)
    for text in X:
        text=text.replace('\\',' ')
        #text=str(text).translate(remove)
        tokens=tokenizer(text,max_length=pad_size)
        if len(tokens['input_ids'])<pad_size:
            tokens['input_ids']+=([0]*(pad_size-len(tokens['input_ids'])))
            tokens['token_type_ids']+=([0]*(pad_size-len(tokens['token_type_ids'])))
            tokens['attention_mask']+=([0]*(pad_size-len(tokens['attention_mask'])))
        res.append([tokens['input_ids'],tokens['token_type_ids'],tokens['attention_mask']])
    res=torch.tensor(res)
    return res

class DataHandler(Dataset):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y


    def __getitem__(self, index):
        x=self.X[index]
        y=self.Y[index]
        return x,y,index

    def __len__(self):
        return len(self.X)