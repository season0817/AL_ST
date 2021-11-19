import pandas as pd
import numpy as np
import torch
from model import get_net
from dataset import get_dataset,get_handler
from query_strategies import LeastConfidence,MaxConfidence
# parameters
SEED = 1
NUM_INIT_LB = 120
NUM_QUERY = 1000
ST_NUM_QUERY= 12
NUM_ROUND = 10
DATA_NAME = 'ag_news'
MODEL_NAME='BERT'

args={
    'n_epoch':50,
    'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
    'loader_te_args':{'batch_size': 64, 'num_workers': 1},
    'optimizer_args':{'lr': 0.000005}
}
if __name__ == '__main__':
    print("MODEL_NAME",MODEL_NAME)
    print("NUM_INIT_LB",NUM_INIT_LB)
    print("NUM_QUERY",NUM_QUERY)
    print("ST_NUM_QUERY", ST_NUM_QUERY)
    print("NUM_ROUND", NUM_ROUND)

    #set seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.enabled = False

    # load dataset
    X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)

    # start experiment
    n_pool = len(Y_tr)
    n_test = len(Y_te)
    print('number of labeled pool: {}'.format(NUM_INIT_LB))#1000
    print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))#
    print('number of testing pool: {}'.format(n_test))

    # generate initial labeled pool
    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_tmp = np.arange(n_pool)
    np.random.shuffle(idxs_tmp)
    idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

    # load network
    net = get_net(MODEL_NAME)
    handler = get_handler()

    #strategy = LeastConfidence.LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
    strategy=MaxConfidence.MaxConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
    print(DATA_NAME)
    print('SEED {}'.format(SEED))
    print(type(strategy).__name__)

    # round 0 accuracy
    strategy.train()
    P = strategy.predict(X_te, Y_te)
    acc = np.zeros(NUM_ROUND + 1)
    acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    print('Round 0\ntesting accuracy {}'.format(acc[0]))
    for rd in range(1, NUM_ROUND+1):
        print('Round {}'.format(rd))

        # query
        P_idxs, p_labels = strategy.query(ST_NUM_QUERY)
        #q_idxs = strategy.query(NUM_QUERY)
        idxs_lb[P_idxs] = True
        cur_n_pool=len(idxs_lb[np.where(idxs_lb==True)])#当前标注池中的样本数量
        print("now, the number of labeled pool:",cur_n_pool)



        # update
        #strategy.update(idxs_lb)
        strategy.update_D_(idxs_lb,P_idxs,p_labels)

        strategy.train()

        # round accuracy
        P = strategy.predict(X_te, Y_te)
        acc[rd] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
        print('testing accuracy {}'.format(acc[rd]))

    # print results
    print('SEED {}'.format(SEED))
    print(type(strategy).__name__)
    print(acc)