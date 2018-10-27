import numpy as np
import pdb
import os
import time
import sys


NN    = ['mlp', 'rnn']
NN    = ['mlp']
ATT   = [1]
SHARE = [1]

for nn in NN:
    for att in ATT:
        for share in SHARE:

            command="hyperparam_search.py --nn {} --att {} --share {}".format(nn, att, share)
            print(command)
            
            command = "{} cc_launch.sh {}".format(sys.argv[1], command) 

            os.system(command)
            time.sleep(2)












