import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
	def __init__(self, X, Y,num_classes, idxs_lb, net, handler, args):
		super(RandomSampling, self).__init__(X, Y,num_classes, idxs_lb, net, handler, args)

	def query(self, n):
		return np.random.choice(np.where(self.idxs_lb==0)[0], n)

	def query_D_(self,n):
		ins=np.random.choice(np.where(self.idxs_lb==0)[0], n)
		probs = self.predict_prob(self.X[ins], self.Y[ins])
		return ins,probs.max(1)[1]