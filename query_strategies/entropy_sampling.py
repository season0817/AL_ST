import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
	def __init__(self, X, Y,X_te, Y_te,num_classes, idxs_lb, net, handler, args):
		super(EntropySampling, self).__init__(X, Y,X_te, Y_te,num_classes, idxs_lb, net, handler, args)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)
		ins=U.sort()[1][:n]
		#probs[U.sort()[1]  概率
		return idxs_unlabeled[ins]

	def query_with_class_dependent(self,n):
		idxs_unlabeled=np.arange(self.n_pool)[~self.idxs_lb]
		probs=self.predict_prob(self.X[idxs_unlabeled],self.Y[idxs_unlabeled])
		log_probs=torch.log(probs)
		U=(probs*log_probs).sum(1)
		ins=U.sort()[1]
		samples_per_class = n // self.num_classes
		idxs_res=[]
		probs_res=probs[ins].max(1)[1]
		for label in range(self.num_classes):
			idxs_cur=ins[np.where(probs_res==label)[0][:samples_per_class]]
			idxs_res.extend(list(idxs_cur))
		#当accuracy很低出现某一类没有预测结果时,可能需要以下步骤，添加顺序的其他index
		if len(idxs_res)<n:
			idxs_add=[]
			for idx in list(ins):
				if idx not in idxs_res:
					idxs_add.append(idx)
				if len(idxs_add)==n-len(idxs_res):
					break
			idxs_res.extend(idxs_add)
		idxs_res=np.array(idxs_res)
		return idxs_unlabeled[idxs_res]


