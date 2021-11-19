import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
def get_net(name):
    if name=='BERT':
        return BERT
    elif name=='TextCNN':
        return CNN

class BERT(nn.Module):
    def __init__(self,num_classes):
        super(BERT,self).__init__()
        dropout=0.5
        self.bert=BertModel.from_pretrained('./bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad=True
        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(768, num_classes)

    def forward(self,x):
        encoded =self.bert(input_ids=x[:,0],attention_mask=x[:,2])
        pooled=encoded.pooler_output
        pooled=self.dropout(pooled)
        outputs=self.fc(pooled)
        return outputs,pooled


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        num_filters=100
        filter_sizes=(4, 5, 6)
        dropout=0.5
        self.embedding = nn.Embedding(50000, 128, padding_idx=50000 - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, 128)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 4)


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        mid = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(mid)
        out = self.fc(out)
        return out,mid
