import numpy as np
import torch
import pdb
import os 


path = 'paramsearch_results/'
files = os.listdir(path)

for fl in files:
    results = torch.load(path + fl)
    results_lst = []
    all_tst_sdrs = []
    all_val_sdrs = []
    for rslt in results:
        arg = (rslt['mean_test'], rslt['mean_val'], list(rslt['config']))
        results_lst.append(arg)
        #print(arg)
        all_tst_sdrs.append(rslt['mean_test'])
        all_val_sdrs.append(rslt['mean_val'])
    #max_ind_tst = np.amax(all_tst_sdrs)
    max_ind_val = int(np.argmax(all_val_sdrs))
    print('Model name: {}, best config: {}, num. of completed configs: {}, \n best test_sdr: {}, best val_sdr: {}'.format(results[0]['model_name'], results[max_ind_val]['config'], len(results),
                                                                                                             all_tst_sdrs[max_ind_val], all_val_sdrs[max_ind_val]))



