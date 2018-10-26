import numpy as np
import torch 
import pdb
import matplotlib.pyplot as plt

folder = 'results/'
bss_st_ff = torch.load(folder + 'standard_ff.bsseval')
bss_dis_ff_dis_rnn = torch.load(folder + 'dis_ff_dis_rnn.bsseval')
bss_dis_ff_dis_ff = torch.load(folder + 'dis_ff_dis_ff.bsseval')
bss_disside_ff_dis_ff = torch.load(folder + 'disatt_ff_dis_ff.bsseval')


args_dis_ff_dis_rnn = torch.load(folder + 'dis_ff_dis_ff.args')

st_ff_sdrs = []
dis_ff_dis_rnn_sdrs = []
dis_ff_dis_ff_sdrs = []
disside_ff_dis_ff_sdrs = []

for res1, res2, res3, res4 in zip(bss_st_ff, bss_dis_ff_dis_rnn, bss_dis_ff_dis_ff, bss_disside_ff_dis_ff):
    st_ff_sdrs.append(res1[0].mean())
    dis_ff_dis_rnn_sdrs.append(res2[0].mean())
    dis_ff_dis_ff_sdrs.append(res3[0].mean())
    disside_ff_dis_ff_sdrs.append(res4[0].mean())

plt.plot(st_ff_sdrs, label='st_ff')
#plt.plot(dis_ff_dis_rnn_sdrs, label='dis_ff')
#plt.plot(dis_ff_dis_ff_sdrs, label='dis_ff_dis_ff')
plt.plot(disside_ff_dis_ff_sdrs, label='disside_ff_dis_ff')

plt.legend()

print('st ff {}'.format(np.mean(st_ff_sdrs)))
print('dis ff dis rnn {}'.format(np.mean(dis_ff_dis_rnn_sdrs)))
print('dis_ff_dis_ff {}'.format(np.mean(dis_ff_dis_ff_sdrs)))
print('disside_ff_dis_ff {}'.format(np.mean(disside_ff_dis_ff_sdrs)))


pdb.set_trace()
