import numpy as np
from .strategy import Strategy

class MaxConfidence(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(MaxConfidence, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        U = probs.max(1)[0]
        ins = U.sort(descending=True)[1][:n]#降序
        acc = 1.0 * (self.Y[idxs_unlabeled] == probs.max(1)[1]).sum().item() / len(self.Y[idxs_unlabeled])
        print(acc)
        return idxs_unlabeled[ins], probs[ins].max(1)[1]
