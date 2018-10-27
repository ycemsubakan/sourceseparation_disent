import torch
import numpy
import visdom
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from nn import GatedDense

class base_model(nn.Module):
    def __init__(self, arguments, K, Kdis, Linput):
        super(base_model, self).__init__()
        self.Kdis = Kdis
        self.Linput = Linput
        self.arguments = arguments
        self.pper = arguments.plot_interval

    def trainer(self, loader, opt, vis):
        
        EP = self.arguments.EP_train
        errs = []
        for ep in range(EP):
            for i, dt in enumerate(loader):
                self.zero_grad()
                xhat1, xhat2 = self.forward(dt)
                
                err = (dt[2] - xhat1).pow(2).mean() + (dt[3] - xhat2).pow(2).mean()
                err.backward()

                # grad clipping: 
                params = opt.param_groups[0]['params']
                torch.nn.utils.clip_grad_norm_(params, self.arguments.clip_norm, norm_type=2)
                
                opt.step()

                print('Error {}, batch [{}/{}], epoch [{}/{}]'.format(err.item(),
                                                        i+1, len(loader), ep+1, EP))
            errs.append(err.item())
            if self.arguments.plot_training and ((ep % self.pper) == 0):
                vis.line(errs, win='err')

                opts = {'title' : 'source 1'}
                vis.heatmap(dt[2][0].t().sqrt(), opts=opts, win='source 1')

                opts = {'title' : 'source 2'}
                vis.heatmap(dt[3][0].t().sqrt(), opts=opts, win='source 2')

                opts = {'title' : 'sourcehat 1'}
                vis.heatmap(xhat1[0].t().sqrt(), opts=opts, win='sourcehat 1')

                opts = {'title' : 'sourcehat 2'}
                vis.heatmap(xhat2[0].t().sqrt(), opts=opts, win='sourcehat 2')

                opts = {'title' : 'mixture'}
                vis.heatmap(dt[0][0].t().sqrt(), opts=opts, win='mixture')

    def cat_disvar(self, dt, f):
        lst =  [dt, f.unsqueeze(1).expand(dt.size(0), 
                                             dt.size(1), 
                                             f.size(1))]
        cat_dt = torch.cat(lst, dim=2)
        return cat_dt

''' MLP '''
class mlp(base_model): 
    def __init__(self, arguments, K, Kdis, Linput):
        super(mlp, self).__init__(arguments, K, Kdis, Linput)
        dropout = arguments.dropout

        if arguments.num_layers==1:
            self.sep_1 = GatedDense(Linput, Linput, dropout=dropout)
            self.sep_2 = GatedDense(Linput, Linput, dropout=dropout)
        elif arguments.num_layers==2:
            self.sep_1 = nn.Sequential(GatedDense(Linput, K, dropout=dropout), 
                                       GatedDense(K, Linput, dropout=dropout))
            self.sep_2 = nn.Sequential(GatedDense(Linput, K, dropout=dropout),
                                       GatedDense(K, Linput, dropout=dropout))
        elif arguments.num_layers==3:
            self.sep_1 = nn.Sequential(GatedDense(Linput, K, dropout=dropout), 
                                       GatedDense(K, K, dropout=dropout),
                                       GatedDense(K, Linput, dropout=dropout))
            self.sep_2 = nn.Sequential(GatedDense(Linput, K, dropout=dropout), 
                                       GatedDense(K, K, dropout=dropout),
                                       GatedDense(K, Linput, dropout=dropout))
    def forward(self, dt):

        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        xhat1 = F.softplus(self.sep_1(dt[0]))
        xhat2 = F.softplus(self.sep_2(dt[0]))

        return xhat1, xhat2


''' MLP with Shared Architecture '''
class mlp_share(base_model): 
    def __init__(self, arguments, K, Kdis, Linput):
        super(mlp_share, self).__init__(arguments, K, Kdis, Linput)
        dropout = arguments.dropout
        self.Linput = Linput

        if arguments.num_layers==1:
            self.sep = GatedDense(Linput, 2*Linput, dropout=dropout)
        elif arguments.num_layers==2:
            self.sep = nn.Sequential(GatedDense(Linput, 2*K, dropout=dropout),
                                     GatedDense(2*K, 2*Linput, dropout=dropout))
        elif arguments.num_layers==3:
            self.sep = nn.Sequential(GatedDense(Linput, 2*K, dropout=dropout),
                                     GatedDense(2*K, 2*K, dropout=dropout),
                                     GatedDense(2*K, 2*Linput, dropout=dropout))
    def forward(self, dt):

        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        h = self.sep(dt[0])

        xhat1 = F.softplus(h[:,:,:self.Linput])
        xhat2 = F.softplus(h[:,:,self.Linput:])

        return xhat1, xhat2

''' LSTM '''
class lstm(base_model): 
    def __init__(self, arguments, K, Kdis, Linput):
        super(lstm, self).__init__(arguments, K, Kdis, Linput)
        dropout=arguments.dropout

        self.sep_rnn1 = nn.LSTM(input_size = Linput,
                               hidden_size = K,
                               num_layers=arguments.num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)

        self.sep_rnn2 = nn.LSTM(input_size = Linput,
                               hidden_size = K,
                               num_layers=arguments.num_layers, 
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)

        if arguments.num_layers==1:
            self.sep_out1 = GatedDense(2*K, Linput, dropout=dropout)
            self.sep_out2 = GatedDense(2*K, Linput, dropout=dropout)
        elif arguments.num_layers==2:
            self.sep_out1 = nn.Sequential(GatedDense(2*K, K, dropout=dropout), 
                                          GatedDense(K, Linput, dropout=dropout))
            self.sep_out2 = nn.Sequential(GatedDense(2*K, K, dropout=dropout),
                                          GatedDense(K, Linput, dropout=dropout))

    def forward(self, dt):

        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        hhat1, _ = (self.sep_rnn1(dt[0]))
        hhat2, _ = (self.sep_rnn2(dt[0]))

        # get the network outputs
        xhat1 = F.softplus(self.sep_out1(hhat1))
        xhat2 = F.softplus(self.sep_out2(hhat2))
        
        return xhat1, xhat2

''' LSTM with Shared Architecture '''
class lstm_share(base_model): 
    def __init__(self, arguments, K, Kdis, Linput):
        super(lstm_share, self).__init__(arguments, K, Kdis, Linput)
        dropout=arguments.dropout

        self.sep_rnn = nn.LSTM(input_size = Linput,
                               hidden_size = 2*K,
                               num_layers=arguments.num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)

        self.sep_out = GatedDense(2*K, 2*Linput, dropout=dropout)
        if arguments.num_layers==1:
            self.sep_out = GatedDense(4*K, 2*Linput, dropout=dropout)
        elif arguments.num_layers==2:
            self.sep_out = nn.Sequential(GatedDense(4*K, 2*K, dropout=dropout), 
                                       GatedDense(2*K, 2*Linput, dropout=dropout))

    def forward(self, dt):

        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        h, _ = (self.sep_rnn(dt[0]))
        
        h = self.sep_out(h)
 
        xhat1 = F.softplus(h[:,:,:self.Linput])
        xhat2 = F.softplus(h[:,:,self.Linput:])

        return xhat1, xhat2


''' MLP with Attention '''
class mlp_att(base_model): 
    def __init__(self, arguments, K, Kdis, Linput):
        super(mlp_att, self).__init__(arguments, K, Kdis, Linput)
       
        dropout=self.arguments.dropout
        self.ntemp = self.arguments.ntemp

        self.dim_red = nn.Linear(Linput, K)

        self.templates1 = nn.Linear(self.ntemp, Kdis, bias=False)
        self.templates2 = nn.Linear(self.ntemp, Kdis, bias=False)
        
        self.sel1 = nn.Linear(K, self.ntemp)
        self.sel2 = nn.Linear(K, self.ntemp)

        if arguments.num_layers==1:
            self.sep_1 = GatedDense(Kdis + K, Linput, dropout=dropout)
            self.sep_2 = GatedDense(Kdis + K, Linput, dropout=dropout)
        elif arguments.num_layers==2:
            self.sep_1 = nn.Sequential(GatedDense(Kdis + K, Kdis + K, dropout=dropout), 
                                       GatedDense(Kdis + K, Linput, dropout=dropout))
            self.sep_2 = nn.Sequential(GatedDense(Kdis + K, Kdis + K, dropout=dropout), 
                                       GatedDense(Kdis + K, Linput, dropout=dropout))
        elif arguments.num_layers==3:
            self.sep_1 = nn.Sequential(GatedDense(Kdis + K, Kdis + K, dropout=dropout), 
                                       GatedDense(Kdis + K, Kdis + K, dropout=dropout),
                                       GatedDense(Kdis + K, Linput, dropout=dropout))
            self.sep_2 = nn.Sequential(GatedDense(Kdis + K, Kdis + K, dropout=dropout), 
                                       GatedDense(Kdis + K, Kdis + K, dropout=dropout),
                                       GatedDense(Kdis + K, Linput, dropout=dropout))
    def forward(self, dt):
        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        #pdb.set_trace()
        eye = torch.eye(self.ntemp).cuda()
        temps1 = self.templates1(eye).unsqueeze(1).unsqueeze(1)
        temps2 = self.templates2(eye).unsqueeze(1).unsqueeze(1)

        mix = (self.dim_red(dt[0]))

        ws1 = F.softmax(self.sel1(mix).permute(2, 0, 1), dim=0)
        ws2 = F.softmax(self.sel2(mix).permute(2, 0, 1), dim=0)
        
        f1 = (ws1.unsqueeze(-1)*temps1).sum(0)
        f2 = (ws2.unsqueeze(-1)*temps2).sum(0)

        cat_s1 = torch.cat([mix, f1], dim=2)
        cat_s2 = torch.cat([mix, f2], dim=2)

        # get the hhats
        hhat1 = (self.sep_1(cat_s1))
        hhat2 = (self.sep_2(cat_s2))

        xhat1 = F.softplus(hhat1)
        xhat2 = F.softplus(hhat2)

        return xhat1, xhat2

''' MLP with Attention and Shared Architecture'''
class mlp_att_share(base_model): 
    def __init__(self, arguments, K, Kdis, Linput):
        super(mlp_att_share, self).__init__(arguments, K, Kdis, Linput)
       
        dropout=self.arguments.dropout
        self.ntemp = self.arguments.ntemp

        self.dim_red = nn.Linear(Linput, K)

        self.templates1 = nn.Linear(self.ntemp, Kdis, bias=False)
        self.templates2 = nn.Linear(self.ntemp, Kdis, bias=False)
        
        self.sel1 = nn.Linear(K, self.ntemp)
        self.sel2 = nn.Linear(K, self.ntemp)

        if arguments.num_layers==1:
            self.sep_1 = GatedDense(Kdis + K, Linput, dropout=dropout)
            self.sep_2 = GatedDense(Kdis + K, Linput, dropout=dropout)
        elif arguments.num_layers==2:
            self.sep_1 = nn.Sequential(GatedDense(Kdis + K, Kdis + K, dropout=dropout), 
                                       GatedDense(Kdis + K, Linput, dropout=dropout))
            self.sep_2 = nn.Sequential(GatedDense(Kdis + K, Kdis + K, dropout=dropout), 
                                       GatedDense(Kdis + K, Linput, dropout=dropout))
        elif arguments.num_layers==3:
            self.sep_1 = nn.Sequential(GatedDense(Kdis + K, Kdis + K, dropout=dropout), 
                                       GatedDense(Kdis + K, Kdis + K, dropout=dropout),
                                       GatedDense(Kdis + K, Linput, dropout=dropout))
            self.sep_2 = nn.Sequential(GatedDense(Kdis + K, Kdis + K, dropout=dropout), 
                                       GatedDense(Kdis + K, Kdis + K, dropout=dropout),
                                       GatedDense(Kdis + K, Linput, dropout=dropout))
    def forward(self, dt):
        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        #pdb.set_trace()
        eye = torch.eye(self.ntemp).cuda()
        temps1 = self.templates1(eye).unsqueeze(1).unsqueeze(1)
        temps2 = self.templates2(eye).unsqueeze(1).unsqueeze(1)

        mix = (self.dim_red(dt[0]))

        ws1 = F.softmax(self.sel1(mix).permute(2, 0, 1), dim=0)
        ws2 = F.softmax(self.sel2(mix).permute(2, 0, 1), dim=0)
        
        f1 = (ws1.unsqueeze(-1)*temps1).sum(0)
        f2 = (ws2.unsqueeze(-1)*temps2).sum(0)

        cat_s1 = torch.cat([mix, f1], dim=2)
        cat_s2 = torch.cat([mix, f2], dim=2)

        # get the hhats
        hhat1 = (self.sep_1(cat_s1))
        hhat2 = (self.sep_2(cat_s2))

        xhat1 = F.softplus(hhat1)
        xhat2 = F.softplus(hhat2)

        return xhat1, xhat2

