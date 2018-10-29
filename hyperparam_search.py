import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
import pdb
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from itertools import islice
import utils as ut
import pickle
import time
import datetime
import visdom
import models 
import os

timestamp = str(datetime.datetime.now()).replace(' ','')


parser = argparse.ArgumentParser(description='Source separation experiments')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--optimizer', type=str, default='Adam', metavar='optim', help='Optimizer')

# dataset sizes
parser.add_argument('--ntrs', type=int, default=100)
parser.add_argument('--ntsts', type=int, default=20)
parser.add_argument('--nval', type=int, default=10)


# model describers 
parser.add_argument('--nn', type=str, default='mlp', help='mlp, rnn')
parser.add_argument('--att', type=int, default=1, help='0 1')
parser.add_argument('--share', type=int, default=1, help='0 1')
parser.add_argument('--side', type=int, default=1, help='enable use of side data')
parser.add_argument('--gated', type=int, default=0, help='0 1')
parser.add_argument('--num_layers', type=int, default=2, help='1 2 3')
parser.add_argument('--act', type=str, default='relu', help='relu, sigmoid')

# hyper parameters to search over 
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--K', type=int, default=150)
parser.add_argument('--Kdis', type=int, default=250)
parser.add_argument('--ntemp', type=int, default=100)
parser.add_argument('--ds', type=int, default=1, help='downsampling the side data')
parser.add_argument('--side_model', type=str, default='mlp', help="mlp rnn")

parser.add_argument('--linear', type=int, default=0)

# others 
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--clip_norm', type=float, default=0.25)
parser.add_argument('--plot_interval', type=int, default=100)
parser.add_argument('--plot_training', type=int, default=0)
parser.add_argument('--save_files', type=int, default=1)
parser.add_argument('--EP_train', type=int, default=2000)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--Kdisc', type=int, default=90)
parser.add_argument('--notes', type=str, default='')
parser.add_argument('--val_intervals', type=int, default=100)

parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--side_max', type=int, default=500, help='max size of side data')

arguments = parser.parse_args()

arguments.cuda = torch.cuda.is_available()

torch.manual_seed(arguments.seed)
if arguments.cuda:
    torch.cuda.manual_seed(arguments.seed)
np.random.seed(arguments.seed)

if arguments.plot_training:
    vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev',
                        use_incoming_socket=False)
    assert vis.check_connection()
else:
    vis = None


loader, tr_directories, tst_directories, val_directories = ut.timit_prepare_data(arguments, 
                                                                                 folder='TRAIN', 
                                                                                 ntrs=arguments.ntrs, 
                                                                                 ntsts=arguments.ntsts)
arguments.tr_directories = tr_directories
arguments.tst_directories = tst_directories
arguments.val_directories = val_directories

save_path = 'model_files'
if not os.path.exists(save_path):
    os.mkdir(save_path)

results_path = 'paramsearch_results'
if not os.path.exists(results_path):
    os.mkdir(results_path)

arguments.model = 'arc_{}_att_{}_share_{}_side_{}_sm_{}_{}'.format(arguments.nn, arguments.att, arguments.share, arguments.side, arguments.side_model, timestamp)
print(arguments.model)

# sample the configurations 
hyperparam_configs = ut.sample_hyperparam_configs(arguments, Nconfigs=100)

all_results = []
for cnum, config in enumerate(hyperparam_configs):
    print('Config {}, {}'.format(cnum, config))
    # set the hyperparams
    arguments.lr = float(config[0])
    arguments.K = int(config[1])
    arguments.Kdis = int(config[2])
    arguments.num_layers = int(config[3])
    arguments.dropout = float(config[4])
    arguments.gated = int(config[5])
    arguments.act = str(config[6])
    arguments.ntemp = int(config[7])
    arguments.ds = int(config[8])

    # set the model
    if arguments.nn=='mlp':
        if arguments.att:
            if arguments.share:
                    if arguments.side: 
                        snet = models.mlp_att_share_side(arguments, arguments.K, arguments.Kdis, 513)
                    else:
                        snet = models.mlp_att_share(arguments, arguments.K, arguments.Kdis, 513)
            else:
                snet = models.mlp_att(arguments, arguments.K, arguments.Kdis, 513)
        else:
            if arguments.share:
                snet = models.mlp_share(arguments, arguments.K, arguments.Kdis, 513)
            else:
                snet = models.mlp(arguments, arguments.K, arguments.Kdis, 513)
    if arguments.nn=='rnn':
        if arguments.att:
            if arguments.share:
                if arguments.side: 
                    snet = models.lstm_att_share_side(arguments, arguments.K, arguments.Kdis, 513)
                else:
                    snet = models.lstm_att_share(arguments, arguments.K, arguments.Kdis, 513)
            else:
                snet = models.lstm_att(arguments, arguments.K, arguments.Kdis, 513)
        else:
            if arguments.share:
                snet = models.lstm_share(arguments, arguments.K, arguments.Kdis, 513)
            else:
                snet = models.lstm(arguments, arguments.K, arguments.Kdis, 513)

    if arguments.cuda:
        snet = snet.cuda()

    if arguments.optimizer=='sgd':
        opt = torch.optim.SGD(snet.parameters(), lr=arguments.lr)
    else:
        opt = torch.optim.Adam(snet.parameters(), lr=arguments.lr)

    snet.trainer(loader, opt, vis)

    snet.eval()

    bss_evals_test = ut.timit_test_data(arguments, snet, directories=tst_directories)
    mean_test = ut.compute_meansdr(arguments, bss_evals_test)

    bss_evals_val = ut.timit_test_data(arguments, snet, directories=val_directories)
    mean_val = ut.compute_meansdr(arguments, bss_evals_val)

    results_dict = {'config': config,
                    'bss_evals_test': bss_evals_test,
                    'mean_test': mean_test,
                    'bss_evals_val': bss_evals_val,
                    'mean_val': mean_val,
                    'arguments': arguments,
                    'model_name': arguments.model}
    all_results.append(results_dict)
    
    torch.save(all_results, results_path + '/' + arguments.model + '.results')




