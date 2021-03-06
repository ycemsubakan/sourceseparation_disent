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
import visdom
import models 
import os

vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev',
                    use_incoming_socket=False)
assert vis.check_connection()

parser = argparse.ArgumentParser(description='Source separation experiments')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--optimizer', type=str, default='Adam', metavar='optim', help='Optimizer')

# dataset sizes
parser.add_argument('--ntrs', type=int, default=100)
parser.add_argument('--ntsts', type=int, default=20)
parser.add_argument('--nval', type=int, default=10)


# model describers 
parser.add_argument('--nn', type=str, default='mlp', help='mlp, rnn')
parser.add_argument('--att', type=int, default=0, help='0 1')
parser.add_argument('--share', type=int, default=0, help='0 1')
parser.add_argument('--gated', type=int, default=0, help='0 1')
parser.add_argument('--num_layers', type=int, default=2, help='1 2 3')
parser.add_argument('--act', type=str, default='relu', help='relu, sigmoid')

# hyper parameters to search over 
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--K', type=int, default=150)
parser.add_argument('--Kdis', type=int, default=250)
parser.add_argument('--ntemp', type=int, default=100)


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
parser.add_argument('--val_intervals', type=int, default=50)

parser.add_argument('--dropout', type=float, default=0.2)

arguments = parser.parse_args()

arguments.cuda = torch.cuda.is_available()

torch.manual_seed(arguments.seed)
if arguments.cuda:
    torch.cuda.manual_seed(arguments.seed)
np.random.seed(arguments.seed)
timestamp = round(time.time())
arguments.timestamp = timestamp

loader, tr_directories, tst_directories, val_directories = ut.timit_prepare_data(arguments, 
                                                                                 folder='TRAIN', 
                                                                                 ntrs=arguments.ntrs, 
                                                                                 ntsts=arguments.ntsts)

pdb.set_trace()
arguments.tr_directories = tr_directories
arguments.tst_directories = tst_directories
arguments.val_directories = val_directories

save_path = 'model_files'
if not os.path.exists(save_path):
    os.mkdir(save_path)

arguments.model = '{}{}_att{}_share{}_gated{}_{}'.format(arguments.nn, arguments.num_layers, arguments.att, arguments.share, arguments.gated, arguments.act)
  
# model definition:  
if arguments.nn=='mlp':
    if arguments.att:
        if arguments.share:
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
            pass
        else:
            pass
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

#for par in snet.parameters():
#    c = 0.01
#    nn.init.uniform(par, -c, c)
snet.trainer(loader, opt, vis)
#torch.save(snet.state_dict(), save_path + '/' + arguments.model + '.t')

snet.eval()

results_path = 'results'
if not os.path.exists(results_path):
    os.mkdir(results_path)

bss_evals = ut.timit_test_data(arguments, snet, directories=tst_directories)
ut.compute_meansdr(arguments, bss_evals)

torch.save(bss_evals, results_path + '/' + arguments.model + '.bsseval')
torch.save(arguments, results_path + '/' + arguments.model + '.args')

#vis.heatmap(dt[0][0].squeeze().t().sqrt())
#vis.heatmap(dt[0][2].squeeze().t().sqrt())
#vis.heatmap(dt[0][3].squeeze().t().sqrt())




