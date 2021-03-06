import pandas as pd
import numpy as np
import torch
from model import get_net
from dataset import get_dataset,get_handler
from query_strategies import random_sampling,LeastConfidence,MaxConfidence,entropy_sampling
import time
# parameters
SEED = 1
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
    print(time.asctime(time.localtime(time.time())))
    print(args)
    print("MODEL_NAME", MODEL_NAME)
    print("NUM_ROUND", NUM_ROUND)
    print("DATA_NAME", DATA_NAME)
    print('SEED {}'.format(SEED))
    # load dataset
    X_tr, Y_tr, X_te, Y_te,num_classes = get_dataset(DATA_NAME)

    n_pool = len(Y_tr)
    n_test = len(Y_te)
    num_init_sizes = [60]
    for NUM_INIT_LB in num_init_sizes:
        #ST_NUM_QUERY = int(NUM_INIT_LB / 10)
        ST_NUM_QUERY=12
        print("NUM_INIT_LB", NUM_INIT_LB)
        print("ST_NUM_QUERY", ST_NUM_QUERY)
        #set seed
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.backends.cudnn.enabled = False
        # start experiment
        print('number of labeled pool: {}'.format(NUM_INIT_LB))#1000
        print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))#
        print('number of testing pool: {}'.format(n_test))

        # generate initial labeled pool
        idxs_lb = np.zeros(n_pool, dtype=bool)
        idxs_tmp = np.arange(n_pool)
        np.random.shuffle(idxs_tmp)
        #idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

        # generate initial labeled pool without class dependent
        samples_per_class = NUM_INIT_LB // num_classes
        idxs_init = []
        for label in range(num_classes):
            idxs_candi = np.where(Y_tr == label)[0]
            np.random.shuffle(idxs_candi)
            idxs_lb[idxs_candi[:samples_per_class]] = True
            idxs_init.extend(list(idxs_candi[:samples_per_class]))

        # load network
        net = get_net(MODEL_NAME)
        handler = get_handler()


        #strategy = LeastConfidence.LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
        #strategy=MaxConfidence.MaxConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
        strategy = random_sampling.RandomSampling(X_tr, Y_tr, num_classes,idxs_lb, net, handler, args)
        print(type(strategy).__name__)

        # round 0 accuracy
        strategy.train(args['optimizer_args']['lr'])
        P,feats, labels = strategy.predict(X_te, Y_te)
        acc = np.zeros(NUM_ROUND + 1)
        acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
        print('\tRound 0\ntesting accuracy {}'.format(acc[0]))
        strategy.plot_result(feats, labels, 0,"AG_NEWS_ST")
        for rd in range(1, NUM_ROUND+1):
            print('\tRound {}'.format(rd))

            # query
            P_idxs, p_labels = strategy.query_D_(ST_NUM_QUERY)
            #q_idxs = strategy.query(NUM_QUERY)
            idxs_lb[P_idxs] = True
            cur_n_pool=len(idxs_lb[np.where(idxs_lb==True)])#?????????????????????????????????
            print("\tnow, the number of labeled pool:",cur_n_pool)

            # update
            #strategy.update(idxs_lb)
            strategy.update_D_(idxs_lb,P_idxs,p_labels)

            strategy.train(args['optimizer_args']['lr'])

            # round accuracy
            P,feats, labels = strategy.predict(X_te, Y_te)
            acc[rd] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
            print('testing accuracy {}'.format(acc[rd]))
            strategy.plot_result(feats, labels, rd,"AG_NEWS_ST")

        # print results
        print(acc)