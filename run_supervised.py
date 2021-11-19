import pandas as pd
import numpy as np
import torch
from model import get_net
from dataset import get_dataset,get_handler
from query_strategies import LeastConfidence,entropy_sampling
# parameters
SEED = 1
DATA_NAME = 'dbpedia'
MODEL_NAME='BERT'
args={
    'save_path':'./dict/db_se_tiny1.ckpt',
    'early_stopping':False,
    'n_epoch':10,
    'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
    'loader_te_args':{'batch_size': 64, 'num_workers': 1},
    'optimizer_args':{'lr': 0.00003}
}
if __name__ == '__main__':
    print(args)
    print("MODEL_NAME",MODEL_NAME)

    #set seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.enabled = False

    # load dataset
    X_tr, Y_tr, X_te, Y_te,num_classes = get_dataset(DATA_NAME)

    n_pool = len(Y_tr)
    n_test = len(Y_te)

    # start experiment
    train_pool_size = [210+42*i for i in range(11)]
    acc = np.zeros(len(train_pool_size))
    rd=0
    for NUM_INIT_LB in train_pool_size:
        print("round {}".format(rd))
        # generate initial labeled pool
        idxs_lb = np.zeros(n_pool, dtype=bool)
        idxs_tmp = np.arange(n_pool)
        np.random.shuffle(idxs_tmp)
        """
        samples_per_class = NUM_INIT_LB // num_classes
        idxs_init = []
        for label in range(num_classes):
            idxs_candi = np.where(Y_tr == label)[0]
            np.random.shuffle(idxs_candi)
            idxs_lb[idxs_candi[:samples_per_class]] = True
            idxs_init.extend(list(idxs_candi[:samples_per_class]))
        """
        idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True
        print('number of labeled pool: {}'.format(NUM_INIT_LB))
        print('number of testing pool: {}'.format(n_test))

        # load network
        net = get_net(MODEL_NAME)
        handler = get_handler()

        strategy=entropy_sampling.EntropySampling(X_tr,Y_tr,X_te, Y_te,num_classes,idxs_lb,net,handler,args)

        strategy.train(args['optimizer_args']['lr'],args['n_epoch'])
        P,feats, labels  = strategy.predict(X_te, Y_te)

        acc[rd]=1.0 * (Y_te == P).sum().item() / len(Y_te)
        print('testing accuracy {}'.format(acc[rd]))
        #strategy.plot_result(feats, labels, 0, "ARP_SL_84")
        rd+=1


    # print results
    print('SEED {}'.format(SEED))
    print(type(strategy).__name__)
    print(acc)