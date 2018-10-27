import numpy as np
import torch
import pdb

fl = 'arc_mlp_att_1_share_0_1540657028.results'
path = 'paramsearch_results/'

results = torch.load(path + fl)

results_lst = []
for rslt in results:
    arg = (rslt['mean_test'], rslt['mean_val'], list(rslt['config']))
    results_lst.append(arg)
    print(arg)

pdb.set_trace()


