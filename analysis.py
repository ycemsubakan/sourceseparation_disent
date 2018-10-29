
# coding: utf-8

# In[ ]:

import numpy as np
import torch
import pdb
import os 
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")

''' 
TODO:
    1) all results in one pandas dataframe
    2) box plot given some feature

'''


path = 'paramsearch_results/'
files = os.listdir(path)
df = pd.DataFrame()

for fl in files:
    results = torch.load(path + fl)
    results_lst = []
    all_tst_sdrs = []
    all_val_sdrs = []
    for rslt in results:
        arg = (rslt['mean_test'], rslt['mean_val'], list(rslt['config']))
        results_lst.append(arg)
        #print(arg)
        to_append = {}
        for k in rslt['arguments'].__dict__:
            if 'directories' in k:
                continue
            to_append[k] = rslt['arguments'].__dict__[k]
        to_append['mean_test'] = rslt['mean_test']
        to_append['mean_val']  = rslt['mean_val']
        
        to_append = pd.DataFrame(to_append, index=[len(df)])
        df = df.append(to_append)
        
        all_tst_sdrs.append(rslt['mean_test'])
        all_val_sdrs.append(rslt['mean_val'])


    #max_ind_tst = np.amax(all_tst_sdrs)
    max_ind_val = int(np.argmax(all_val_sdrs))

def clean_name(x):
    try:
        x = x.split('_2018')[0]
    except:
        None
    try:
        x = x.split('-2018')[0]
    except:
        None   
    return x
            
df['model_name'] = df['model'].map(clean_name)
df_good_runs = df[df.mean_test>1]


df_no_side = df_good_runs[df_good_runs.side.map(lambda x: np.isnan(x))]
len(df_no_side)



df_no_side.boxplot(column=['mean_val'], by=['att'])

pdb.set_trace()
tips = sns.load_dataset("tips")
# tips['total_bill']
ax = sns.violinplot(x=tips['total_bill'])


# In[ ]:

ax = sns.violinplot(x=df_no_side['mean_val'].reset_index()['mean_val'])


# In[ ]:




# In[1]:

df_no_side['mean_val'].reset_index()['mean_val']


# In[ ]:




# In[ ]:

len(df)
df_good_runs.sort_values('mean_val', ascending=False)[['model_name','mean_test','ntemp']]


# In[ ]:




# In[ ]:




# In[ ]:



