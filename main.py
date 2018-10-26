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

#from deep_sep_expr_shared import sep_run

parser = argparse.ArgumentParser(description='Source separation experiments with GANs/Autoencoders')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--optimizer', type=str, default='RMSprop', metavar='optim',
                    help='Optimizer')

parser.add_argument('--ntrs', type=int, default=100)
parser.add_argument('--ntsts', type=int, default=20)

parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--plot_interval', type=int, default=100)
parser.add_argument('--plot_training', type=int, default=1)
parser.add_argument('--save_files', type=int, default=1)
parser.add_argument('--EP_train', type=int, default=100)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--K', type=int, default=150)
parser.add_argument('--Kdis', type=int, default=100)
parser.add_argument('--Kdisc', type=int, default=90)
parser.add_argument('--notes', type=str, default='')
parser.add_argument('--model', type=str, default='standard_ff', help='standard_ff, standard_rnn, dis_ff, dis_rnn')


arguments = parser.parse_args()

arguments.cuda = torch.cuda.is_available()

torch.manual_seed(arguments.seed)
if arguments.cuda:
    torch.cuda.manual_seed(arguments.seed)
np.random.seed(arguments.seed)
timestamp = round(time.time())
arguments.timestamp = timestamp

loader, tr_directories, tst_directories, side_directories, side_sources = ut.timit_prepare_data(arguments, folder='TRAIN', ntrs=arguments.ntrs, ntsts=arguments.ntsts)
arguments.tr_directories = tr_directories
arguments.tst_directories = tst_directories
arguments.side_directories = side_directories

save_path = 'model_files'
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
if arguments.model == 'st_ff':
    snet = models.sourcesep_net_st_ff(arguments, arguments.K, arguments.Kdis, 513)
elif arguments.model == 'st_rnn':
    snet = models.sourcesep_net_st_rnn(arguments, arguments.K, arguments.Kdis, 513)

elif arguments.model == 'distemplate_ff_dis_ff':
    snet = models.sourcesep_net_distemplate_ff_dis_ff(arguments, arguments.K, arguments.Kdis, 513)
elif arguments.model == 'dis_ff_dis_rnn':
    snet = models.sourcesep_net_dis_ff_dis_rnn(arguments, arguments.K, arguments.Kdis, 513)
elif arguments.model == 'dis_ff_dis_ff':
    snet = models.sourcesep_net_dis_ff_dis_ff(arguments, arguments.K, arguments.Kdis, 513)
elif arguments.model == 'disside_ff_dis_ff':
    snet = models.sourcesep_net_disside_ff_dis_ff(arguments, arguments.K, arguments.Kdis, 513, side_sources)
elif arguments.model == 'disatt_ff_dis_ff':
    snet = models.sourcesep_net_disatt_ff_dis_ff(arguments, arguments.K, arguments.Kdis, 513, side_sources)

if arguments.cuda:
    snet = snet.cuda()

opt = torch.optim.Adam(snet.parameters(), lr=arguments.lr)
#for par in snet.parameters():
#    c = 0.01
#    nn.init.uniform(par, -c, c)
snet.trainer(loader, opt, vis)
torch.save(snet.state_dict(), save_path + '/' + arguments.model + '.t')


results_path = 'results'
if not os.path.exists(results_path):
    os.mkdir(results_path)

bss_evals = ut.timit_test_data(arguments, snet, directories=tst_directories)

all_sdrs = []
for bss_eval in bss_evals:
    all_sdrs.append(bss_eval[0].mean())
print('mean SDR {} model {}'.format(np.mean(all_sdrs), arguments.model))

torch.save(bss_evals, results_path + '/' + arguments.model + '.bsseval')
torch.save(arguments, results_path + '/' + arguments.model + '.args')

#vis.heatmap(dt[0][0].squeeze().t().sqrt())
#vis.heatmap(dt[0][2].squeeze().t().sqrt())
#vis.heatmap(dt[0][3].squeeze().t().sqrt())




