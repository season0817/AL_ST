import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers.optimization import AdamW,Adafactor
from transformers import get_scheduler
import os
from losses import SupConLoss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
class Strategy:
    def __init__(self, X, Y,X_te, Y_te,num_classes,idxs_lb, net, handler, args):
        self.X = X
        self.Y = Y
        self.X_te=X_te
        self.Y_te=Y_te
        self.num_classes=num_classes
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:1" if use_cuda else "cpu")
        self.clf = self.net(self.num_classes).to(self.device)###
        print("device:",self.device)

    def query_with_label_num(self, label_num_dict):
        #random
        max_num = np.array(list(label_num_dict.values())).max()
        sum_num=np.array(list(label_num_dict.values())).sum()
        tot_num=max_num*self.num_classes-sum_num
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        probs_res = probs.max(1)[1]
        #idxs_filtered=np.where(probs.max(1)[0]>0.85)#
        #idxs_unlabeled_filtered=idxs_unlabeled[idxs_filtered]#
        #robs_res_filtered=probs_res[idxs_filtered]#
        idxs_res = []
        label_res=[]
        for label in range(self.num_classes):
            ins=np.random.choice(np.where(probs_res == label)[0],int((max_num-label_num_dict[label])),replace=False)#
            idxs_cur = idxs_unlabeled[ins]#
            idxs_res.extend(list(idxs_cur))
            label_res.extend([label for i in range(int((max_num-label_num_dict[label])))])
        return idxs_res,torch.tensor(label_res)

    def query_most_confident_with_label_num(self,label_num_dict):
        max_num = np.array(list(label_num_dict.values())).max()
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        U,probs_res= probs.max(1)
        ins = U.sort(descending=True)[1]
        idxs_re,probs_re=idxs_unlabeled[ins],probs_res[ins]
        idxs_res=[]
        label_res=[]
        for label in range(self.num_classes):
            tmp_ins=np.random.choice(np.where(probs_re == label)[0],int((max_num-label_num_dict[label])+6),replace=False)#
            idxs_cur = idxs_re[tmp_ins]#
            idxs_res.extend(list(idxs_cur))
            label_res.extend([label for i in range(int((max_num-label_num_dict[label])+6))])
        return idxs_res,torch.tensor(label_res)

    def query_randomly_with_num(self,num):
        ins = np.random.choice(np.where(self.idxs_lb == 0)[0], num,replace=False)
        probs = self.predict_prob(self.X[ins], self.Y[ins])
        return ins, probs.max(1)[1]

    def query_most_confident_with_num(self,num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        U = probs.max(1)[0]
        ins = U.sort(descending=True)[1][:num]
        #acc = 1.0 * (self.Y[idxs_unlabeled] == probs.max(1)[1]).sum().item() / len(self.Y[idxs_unlabeled])
        return idxs_unlabeled[ins], probs[ins].max(1)[1]

    def query_most_ent(self,num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        log_probs = torch.log(probs)
        U = (probs * log_probs).sum(1)
        ins = U.sort()[1][-num:]
        return idxs_unlabeled[ins],probs[ins].max(1)[1]

    def query_most_ent_balanced(self,label_num_dict):
        max_num = np.array(list(label_num_dict.values())).max()
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        _, probs_res = probs.max(1)
        log_probs = torch.log(probs)
        U = (probs * log_probs).sum(1)
        ins = U.sort()[1]
        idxs_re, probs_re = idxs_unlabeled[ins], probs_res[ins]
        idxs_res = []
        label_res = []
        for label in range(self.num_classes):
            if len(np.where(probs_re == label)[0])<((max_num - label_num_dict[label]) + 1):
                idxs_len=len(np.where(probs_re == label)[0])
            else:
                idxs_len=((max_num - label_num_dict[label]) + 1)
            tmp_ins = np.where(probs_re == label)[0][-idxs_len:]
            idxs_cur = idxs_re[tmp_ins]
            idxs_res.extend(list(idxs_cur))
            label_res.extend([label for i in range(idxs_len)])
        return idxs_res, torch.tensor(label_res)

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def save_oracle_samples(self,P_idxs):
        idxs_train = self.get_idxs_train()
        self.sample_Y=self.Y[idxs_train].to(self.device)
        self.sample_projected=self.get_projected(self.X[idxs_train],self.Y[idxs_train]).to(self.device)

    def save_pseudo_samples(self,P_idxs):
        probs = self.predict_prob(self.X[P_idxs], self.Y[P_idxs])
        self.pseudo_Y=probs.max(1)[1]
        self.pseudo_X=self.get_projected(self.X[P_idxs], self.Y[P_idxs]).to(self.device)

    def ST_query(self,n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        U = probs.max(1)[0]
        ins = U.sort(descending=True)[1][:n]
        acc = 1.0 * (self.Y[idxs_unlabeled] == probs.max(1)[1]).sum().item() / len(self.Y[idxs_unlabeled])
        print(acc)
        return idxs_unlabeled[ins], probs[ins].max(1)[1]

    def update_D_(self,idxs_lb,P_idxs,label):
        self.pseudo_idxs=P_idxs
        self.idxs_lb=idxs_lb
        acc = 1.0 * (self.Y[P_idxs] == label).sum().item() / len(self.Y[P_idxs])
        print(acc)
        #self.Y[np.arange(self.n_pool)[P_idxs]]=torch.LongTensor(label.numpy())
        self.tmp_Y=self.Y[P_idxs]
        self.Y[P_idxs] = torch.LongTensor(label.numpy())

    def reset(self,idxs_lb,P_idxs):
        self.idxs_lb=idxs_lb
        self.Y[P_idxs]=self.tmp_Y

    def normalize(self,*xs):
        return [None if x is None else F.normalize(x,dim=-1) for x in xs]

    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)
        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        loss = (p_loss + q_loss) / 2
        return loss

    def info_nce(self,query, positive_key, negative_keys=None, temperature=0.1, reduction='mean'):
        # Inputs all have 2 dimensions.
        if query.dim() != 2 or positive_key.dim() != 2 or (negative_keys is not None and negative_keys.dim() != 2):
            raise ValueError('query, positive_key and negative_keys should all have 2 dimensions.')

        # Each query sample is paired with exactly one positive key sample.
        if len(query) != len(positive_key):
            raise ValueError('query and positive_key must have the same number of samples.')

        # Embedding vectors should have same number of components.
        if query.shape[1] != positive_key.shape[1] != (
        positive_key.shape[1] if negative_keys is None else negative_keys.shape[1]):
            raise ValueError('query, positive_key and negative_keys should have the same number of components.')

        # Normalize to unit vectors
        query, positive_key, negative_keys = self.normalize(query, positive_key, negative_keys)

        if negative_keys is not None:
            # Explicit negative keys
            #print("CL loss")
            # Cosine between positive pairs
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

            # Cosine between all query-negative combinations
            negative_logits = query @ negative_keys.transpose(-2,-1)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        else:
            # Negative keys are implicitly off-diagonal positive keys.
            #print("No negative samples")
            # Cosine between all combinations
            logits = query @ positive_key.transpose(-2,-1)

            # Positive keys are the entries on the diagonal
            labels = torch.arange(len(query), device=query.device)

        return F.cross_entropy(logits / temperature, labels, reduction=reduction)

    def _train_with_r_drop(self,epoch,loader_tr,optimizer):
        print("func:_train")
        self.clf.train()
        criterion=SupConLoss()
        for batch_idx,(x,y,idxs) in enumerate(loader_tr):
            x,y=x.to(self.device),y.to(self.device)
            optimizer.zero_grad()
            out1,e1=self.clf(x)
            out2,e2=self.clf(x)
            #CL
            """
            ce_loss = 0.5 * (F.cross_entropy(out1, y) + F.cross_entropy(out2, y))
            feats=torch.cat([F.normalize(e1.unsqueeze(1),dim=-1),F.normalize(e2.unsqueeze(1),dim=-1)],dim=1)
            cl_loss=criterion(self.device,feats,y)
            lamb=0.5
            loss=(1-lamb)*ce_loss+lamb*cl_loss
            """
            #R_drop
            #ce_loss = 0.5 * (F.cross_entropy(out1, y) + F.cross_entropy(out2, y))
            kl_loss = self.compute_kl_loss(out1, out2)
            a=0.2
            #loss =ce_loss+a * kl_loss
            loss=kl_loss
            pred = out1.max(1)[1]
            acc = 1.0 * (y == pred).sum().item() / len(pred)
            #print(loss)
            #print(acc)
            loss.backward()
            optimizer.step()

    def _train(self, epoch, loader_tr, optimizer):
        #print("func:_train")
        self.clf.train()
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)
            pred = out.max(1)[1]
            acc = 1.0 * (y == pred).sum().item() / len(pred)
            #print(loss)
            #print(acc)
            loss.backward()
            optimizer.step()

    def _train_add_cl(self,epoch,loader_tr,optimizer):

        self.save_pseudo_samples(self.pseudo_idxs)

        optimizer.zero_grad()
        pos_keys, neg_keys = None, None
        for y in self.pseudo_Y:
            idxs_pos = np.where(self.sample_Y.cpu().numpy() == int(y))
            np.random.shuffle(idxs_pos)
            pos = self.sample_projected[idxs_pos][0].unsqueeze(0)
            # print(pos)
            if pos_keys == None:
                pos_keys = pos
            else:
                pos_keys = torch.cat([pos_keys, pos], dim=0)
            idxs_neg = np.where(self.sample_Y.cpu().numpy() != int(y))
            np.random.shuffle(idxs_neg)
            neg = self.sample_projected[idxs_neg][:4].unsqueeze(0)
            # neg = self.sample_projected[idxs_neg][0].unsqueeze(0)
            # print(neg)
            if neg_keys == None:
                neg_keys = neg
            else:
                neg_keys = torch.cat([neg_keys, neg], dim=0)
        neg_keys = neg_keys.reshape(neg_keys.shape[0] * neg_keys.shape[1], neg_keys.shape[2])
        # print(pos_keys)
        # print(neg_keys)
        query = self.pseudo_X
        # print("loss",loss)
        loss_infoNCE = self.info_nce(query, pos_keys, neg_keys)
        loss_infoNCE = loss_infoNCE.requires_grad_()
        print("loss infoNCE", loss_infoNCE)

        self.clf.train()
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)
            loss=loss+loss_infoNCE
            pred = out.max(1)[1]
            acc = 1.0 * (y == pred).sum().item() / len(pred)
            print(loss)
            # print(acc)
            loss.backward()
            optimizer.step()



    def get_idxs_train(self):
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        return idxs_train

    def get_new_labeled_distri(self):
        idxs_train = self.get_idxs_train()
        trainset = self.Y[idxs_train].numpy()
        label_num_dict={}
        for i in range(self.num_classes):
            num=len(trainset[np.where(trainset == i)])
            print("label {} :{}".format(i, num))
            label_num_dict[i]=num
        return label_num_dict

    def evaluate(self):
        loader_te = DataLoader(self.handler(self.X_te, self.Y_te),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        P = torch.zeros(len(self.Y_te), dtype=self.Y_te.dtype)
        feats = torch.zeros([len(self.Y_te), self.num_classes], dtype=torch.float)
        labels = torch.zeros(len(self.Y_te), dtype=self.Y_te.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.cpu()
                feats[idxs] = out.cpu()
                labels[idxs] = y.cpu()
        acc = 1.0 * (self.Y_te == P).sum().item() / len(self.Y_te)
        return acc

    def train(self,lr,n_epoch):
        #n_epoch = self.args['n_epoch']
        #self.clf = self.net(self.num_classes).to(self.device)
        idxs_train=self.get_idxs_train()
        self.get_new_labeled_distri()

        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train]),
                               shuffle=True, **self.args['loader_tr_args'])
        num_training_steps=n_epoch*len(loader_tr)

        self.clf.train()
        #optimizer = optim.Adam(self.clf.parameters(), **self.args['optimizer_args'])
        param_optimizer = list(self.clf.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer=AdamW(optimizer_grouped_parameters,lr=lr,weight_decay=0.01)
        best_acc=0
        early_stopping=self.args['early_stopping']
        for epoch in range(1, n_epoch+1):
            self._train(epoch, loader_tr, optimizer)
            #self._train_with_r_drop(epoch,loader_tr,optimizer)
            if epoch%2==0 and early_stopping==True:
                acc=self.evaluate()
                print("epoch", epoch)
                print("evaluate acc:",acc)
                if acc>best_acc:
                    best_acc=acc
                    torch.save(self.clf,self.args['save_path'])
                else:
                    self.clf=torch.load(self.args['save_path'])
                    self.clf=self.clf.to(self.device)
                    print("break")
                    break


    def selftrain(self,lr,n_epoch,origin_acc):
        #self.clf = self.net(self.num_classes).to(self.device)
        idxs_train = self.get_idxs_train()
        #idxs_train = np.arange(self.n_pool)[~self.idxs_lb]
        #self.get_new_labeled_distri()

        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train]),
                              shuffle=True, **self.args['loader_tr_args'])
        num_training_steps = n_epoch * len(loader_tr)

        self.clf.train()
        # optimizer = optim.Adam(self.clf.parameters(), **self.args['optimizer_args'])
        param_optimizer = list(self.clf.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=0.01)
        best_acc=0
        origin_acc=origin_acc
        origin_clf=torch.load(self.args['save_path'])
        for epoch in range(1, n_epoch + 1):
            self._train(epoch, loader_tr, optimizer)
            #self._train_with_r_drop(epoch, loader_tr, optimizer)
            if epoch%2==0:
                acc=self.evaluate()
                print("epoch", epoch)
                print("evaluate acc:",acc)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(self.clf, self.args['save_path'])
                else:
                    self.clf = torch.load(self.args['save_path'])
                    print("break")
                    break
        if origin_acc>best_acc:
            self.clf = origin_clf
            print("al better")


    def init(self,lr,n_epoch):
        #n_epoch = self.args['n_epoch']
        #self.clf = self.net(self.num_classes).to(self.device)
        idxs_train = self.get_idxs_train()
        self.get_new_labeled_distri()

        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train]),
                               shuffle=True, **self.args['loader_tr_args'])
        num_training_steps = n_epoch * len(loader_tr)

        self.clf.train()
        # optimizer = optim.Adam(self.clf.parameters(), **self.args['optimizer_args'])

        param_optimizer = list(self.clf.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=0.01)
        best_acc=0
        for epoch in range(1, n_epoch + 1):
            self._train(epoch, loader_tr, optimizer)
            #self._train_with_r_drop(epoch, loader_tr, optimizer)
            if epoch%2==0:
                acc=self.evaluate()
                print("epoch",epoch)
                print("evaluate acc:",acc)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(self.clf, self.args['save_path'])
                else:
                    self.clf = torch.load(self.args['save_path'])
                    print("break")
                    break

    def train_with_cl(self):
        n_epoch = self.args['n_epoch']
        idxs_train = self.get_idxs_train()
        self.get_new_labeled_distri()
        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train]),
                               shuffle=True, **self.args['loader_tr_args'])
        self.clf.train()
        param_optimizer = list(self.clf.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args['optimizer_args']['lr'])
        for epoch in range(1, n_epoch+1):
            self._train_add_cl(epoch, loader_tr, optimizer)

    def predict(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype)
        feats=torch.zeros([len(Y),self.num_classes],dtype=torch.float)
        labels=torch.zeros(len(Y),dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x,y = x.to(self.device),y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.cpu()
                feats[idxs]=out.cpu()
                labels[idxs]=y.cpu()
        return P,feats.numpy(),labels.numpy()

    def plot_result(self,feats,labels,rd,title="SL"):
        #label_names=['World','Sports','Business','Sci and Tech']
        label_names=['Company','EducationalInstitution','Artist','Athlete','OfficeHolder','MeanOfTransportation','Building','NaturalPlace','Village','Animal','Plant','Album','Film','WrittenWork']
        tsne = TSNE(n_components=2)
        data = tsne.fit_transform(feats)
        color = ['#00FFFF', '#2E8B57', '#000080', '#8B0000', '#ADFF2F', '#BA55D3',
                 '#B8860B', 'deeppink', 'gold', 'slategray',
                 '#8A2BE2', '#0000FF', '#000000', '#D2691E']

        for i in range(self.num_classes):
            idxs = np.where(labels == i)
            plt.scatter(data[idxs, 0], data[idxs, 1],c=color[i],label=label_names[i])
        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        plt.savefig("./pics_new/"+title+str(rd)+".png",bbox_inches='tight')
        plt.clf()
        #plt.show()

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        #probs = torch.zeros([len(Y), len(np.unique(Y))])
        probs = torch.zeros([len(Y), self.num_classes])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x,y = x.to(self.device),y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        
        return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x,y = x.to(self.device),y.to(self.device)
                out, e1 = self.clf(x)
                embedding[idxs] = e1.cpu()
        
        return embedding

    def get_projected(self,X,Y):
        loader_te = DataLoader(self.handler(X, Y),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        embedding = torch.zeros([len(Y), self.num_classes])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embedding[idxs] = out.cpu()
        return embedding

