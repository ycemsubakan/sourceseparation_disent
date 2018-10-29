import torch
import numpy
import visdom
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
import utils as ut
from nn import GatedDense, Dense, Linear

class base_model(nn.Module):
    def __init__(self, arguments, K, Kdis, Linput):
        super(base_model, self).__init__()
        self.Kdis = Kdis
        self.Linput = Linput
        self.arguments = arguments
        self.pper = arguments.plot_interval

        if arguments.gated:
            self.Dense = GatedDense
        elif not arguments.gated:
            self.Dense = Dense

        if arguments.act == 'relu':
            self.activation = nn.ReLU
        elif arguments.act == 'sigmoid':
            self.activation = nn.Sigmoid

        # sorry for hacky code:
        if arguments.linear:
            self.Dense = Linear

    def trainer(self, loader, opt, vis):
        
        EP = self.arguments.EP_train
        errs = []
        all_vals = []
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

            if self.arguments.verbose and ((ep % self.pper) == 0):
                print('Error {}, batch [{}/{}], epoch [{}/{}]'.format(err.item(),
                                                        i+1, len(loader), ep+1, EP))
            if (ep > 500) and ((ep % self.arguments.val_intervals) == 0):
                print('Validation computations...')
                self.eval()
                val_bss_evals = ut.timit_test_data(self.arguments, self, directories=self.arguments.val_directories)
                self.train()
                all_vals.append(ut.compute_meansdr(self.arguments, val_bss_evals))
                print(all_vals)

                # stop the run if the first val is bad
                if all_vals[-1] < 2.0:
                    print('breaking the run..')
                    break

                if (len(all_vals) > 1) and (all_vals[-1] < all_vals[-2]): 
                    print('breaking the run..')
                    break


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
            self.sep_1 = self.Dense(Linput, Linput, dropout=dropout, activation=self.activation)
            self.sep_2 = self.Dense(Linput, Linput, dropout=dropout, activation=self.activation)
        elif arguments.num_layers==2:
            self.sep_1 = nn.Sequential(self.Dense(Linput, K, dropout=dropout, activation=self.activation), 
                                       self.Dense(K, Linput, dropout=dropout, activation=self.activation))
            self.sep_2 = nn.Sequential(self.Dense(Linput, K, dropout=dropout, activation=self.activation),
                                       self.Dense(K, Linput, dropout=dropout, activation=self.activation))
        elif arguments.num_layers==3:
            self.sep_1 = nn.Sequential(self.Dense(Linput, K, dropout=dropout, activation=self.activation), 
                                       self.Dense(K, K, dropout=dropout, activation=self.activation),
                                       self.Dense(K, Linput, dropout=dropout, activation=self.activation))
            self.sep_2 = nn.Sequential(self.Dense(Linput, K, dropout=dropout, activation=self.activation), 
                                       self.Dense(K, K, dropout=dropout, activation=self.activation),
                                       self.Dense(K, Linput, dropout=dropout, activation=self.activation))
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
            self.sep = self.Dense(Linput, 2*Linput, dropout=dropout)
        elif arguments.num_layers==2:
            self.sep = nn.Sequential(self.Dense(Linput, 2*K, dropout=dropout, activation=self.activation),
                                     self.Dense(2*K, 2*Linput, dropout=dropout, activation=self.activation))
        elif arguments.num_layers==3:
            self.sep = nn.Sequential(self.Dense(Linput, 2*K, dropout=dropout, activation=self.activation),
                                     self.Dense(2*K, 2*K, dropout=dropout, activation=self.activation),
                                     self.Dense(2*K, 2*Linput, dropout=dropout, activation=self.activation))
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
        
        assert arguments.num_layers < 3

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
            self.sep_out1 = self.Dense(2*K, Linput, dropout=dropout, activation=self.activation)
            self.sep_out2 = self.Dense(2*K, Linput, dropout=dropout, activation=self.activation)
        elif arguments.num_layers==2:
            self.sep_out1 = nn.Sequential(self.Dense(2*K, K, dropout=dropout, activation=self.activation), 
                                          self.Dense(K, Linput, dropout=dropout, activation=self.activation))
            self.sep_out2 = nn.Sequential(self.Dense(2*K, K, dropout=dropout, activation=self.activation),
                                          self.Dense(K, Linput, dropout=dropout, activation=self.activation))

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
        assert arguments.num_layers < 3

        self.sep_rnn = nn.LSTM(input_size = Linput,
                               hidden_size = 2*K,
                               num_layers=arguments.num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)

        if arguments.num_layers==1:
            self.sep_out = self.Dense(4*K, 2*Linput, dropout=dropout, activation=self.activation)
        elif arguments.num_layers==2:
            self.sep_out = nn.Sequential(self.Dense(4*K, 2*K, dropout=dropout, activation=self.activation), 
                                         self.Dense(2*K, 2*Linput, dropout=dropout, activation=self.activation))

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

        self.dim_red1 = nn.Linear(Linput, K)
        self.dim_red2 = nn.Linear(Linput, K)

        self.templates1 = nn.Linear(self.ntemp, Kdis, bias=False)
        self.templates2 = nn.Linear(self.ntemp, Kdis, bias=False)
        
        self.sel1 = nn.Linear(K, self.ntemp)
        self.sel2 = nn.Linear(K, self.ntemp)

        if arguments.num_layers==1:
            self.sep_1 = self.Dense(Kdis + K, Linput, dropout=dropout, activation=self.activation)
            self.sep_2 = self.Dense(Kdis + K, Linput, dropout=dropout, activation=self.activation)
        elif arguments.num_layers==2:
            self.sep_1 = nn.Sequential(self.Dense(Kdis + K, Kdis + K, dropout=dropout, activation=self.activation), 
                                       self.Dense(Kdis + K, Linput, dropout=dropout, activation=self.activation))
            self.sep_2 = nn.Sequential(self.Dense(Kdis + K, Kdis + K, dropout=dropout, activation=self.activation), 
                                       self.Dense(Kdis + K, Linput, dropout=dropout, activation=self.activation))
        elif arguments.num_layers==3:
            self.sep_1 = nn.Sequential(self.Dense(Kdis + K, Kdis + K, dropout=dropout, activation=self.activation), 
                                       self.Dense(Kdis + K, Kdis + K, dropout=dropout, activation=self.activation),
                                       self.Dense(Kdis + K, Linput, dropout=dropout, activation=self.activation))
            self.sep_2 = nn.Sequential(self.Dense(Kdis + K, Kdis + K, dropout=dropout, activation=self.activation), 
                                       self.Dense(Kdis + K, Kdis + K, dropout=dropout, activation=self.activation),
                                       self.Dense(Kdis + K, Linput, dropout=dropout, activation=self.activation))
    def forward(self, dt):
        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        eye = torch.eye(self.ntemp).cuda()
        temps1 = self.templates1(eye).unsqueeze(1).unsqueeze(1)
        temps2 = self.templates2(eye).unsqueeze(1).unsqueeze(1)

        mix1 = (self.dim_red1(dt[0]))
        mix2 = (self.dim_red2(dt[0]))

        
        ws1 = F.softmax(self.sel1(mix1).permute(2, 0, 1), dim=0)
        ws2 = F.softmax(self.sel2(mix2).permute(2, 0, 1), dim=0)
        
        f1 = (ws1.unsqueeze(-1)*temps1).sum(0)
        f2 = (ws2.unsqueeze(-1)*temps2).sum(0)

        cat_s1 = torch.cat([mix1, f1], dim=2)
        cat_s2 = torch.cat([mix2, f2], dim=2)

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
       
        self.Linput = Linput
        dropout=self.arguments.dropout
        self.ntemp = self.arguments.ntemp

        self.dim_red = nn.Linear(Linput, 2*K)

        self.templates = nn.Linear(2*self.ntemp, Kdis, bias=False)
        
        self.sel1 = nn.Linear(2*K, 2*self.ntemp)
        self.sel2 = nn.Linear(2*K, 2*self.ntemp)

        hidden_size = 2*(Kdis+K)
        if arguments.num_layers==1:
            self.sep = self.Dense( hidden_size, 2*Linput, dropout=dropout, activation=self.activation)
        elif arguments.num_layers==2:
            self.sep = nn.Sequential(self.Dense(hidden_size, hidden_size, dropout=dropout, activation=self.activation), 
                                     self.Dense(hidden_size, 2*Linput, dropout=dropout, activation=self.activation))
        elif arguments.num_layers==3:
            self.sep = nn.Sequential(self.Dense(hidden_size, hidden_size, dropout=dropout, activation=self.activation), 
                                       self.Dense(hidden_size, hidden_size, dropout=dropout, activation=self.activation),
                                       self.Dense(hidden_size, 2*Linput, dropout=dropout, activation=self.activation))
    def forward(self, dt):
        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        eye = torch.eye(2*self.ntemp).cuda()
        temps = self.templates(eye).unsqueeze(1).unsqueeze(1)

        mix = (self.dim_red(dt[0]))

        ws1 = F.softmax(self.sel1(mix).permute(2, 0, 1), dim=0)
        ws2 = F.softmax(self.sel2(mix).permute(2, 0, 1), dim=0)
        
        f1 = (ws1.unsqueeze(-1)*temps).sum(0)
        f2 = (ws2.unsqueeze(-1)*temps).sum(0)

        cat_s = torch.cat([mix, f1, f2], dim=2)

        # get the hhats
        h = (self.sep(cat_s))

        xhat1 = F.softplus(h[:,:,:self.Linput])
        xhat2 = F.softplus(h[:,:,self.Linput:])

        return xhat1, xhat2

''' LSTM with Attention '''
class lstm_att(base_model): 
    def __init__(self, arguments, K, Kdis, Linput):
        super(lstm_att, self).__init__(arguments, K, Kdis, Linput)
        
        assert arguments.num_layers < 3

        dropout=self.arguments.dropout
        self.ntemp = self.arguments.ntemp

        self.templates1 = nn.Linear(self.ntemp, Kdis, bias=False)
        self.templates2 = nn.Linear(self.ntemp, Kdis, bias=False)
        
        self.sel1 = nn.Linear(2*K, self.ntemp)
        self.sel2 = nn.Linear(2*K, self.ntemp)

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
            self.sep_out1 = self.Dense(Kdis + 2*K, Linput, dropout=dropout, activation=self.activation)
            self.sep_out2 = self.Dense(Kdis + 2*K, Linput, dropout=dropout, activation=self.activation)
        elif arguments.num_layers==2:
            self.sep_out1 = nn.Sequential(self.Dense(Kdis + 2*K, Kdis + 2*K, dropout=dropout, activation=self.activation), 
                                       self.Dense(Kdis + 2*K, Linput, dropout=dropout, activation=self.activation))
            self.sep_out2 = nn.Sequential(self.Dense(Kdis + 2*K, Kdis + 2*K, dropout=dropout, activation=self.activation), 
                                       self.Dense(Kdis + 2*K, Linput, dropout=dropout, activation=self.activation))
    
            
    def forward(self, dt):
        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        #pdb.set_trace()
        eye = torch.eye(self.ntemp).cuda()
        temps1 = self.templates1(eye).unsqueeze(1).unsqueeze(1)
        temps2 = self.templates2(eye).unsqueeze(1).unsqueeze(1)

        mix1, _ = (self.sep_rnn1(dt[0]))
        mix2, _ = (self.sep_rnn2(dt[0]))
        
        ws1 = F.softmax(self.sel1(mix1).permute(2, 0, 1), dim=0)
        ws2 = F.softmax(self.sel2(mix2).permute(2, 0, 1), dim=0)
        
        f1 = (ws1.unsqueeze(-1)*temps1).sum(0)
        f2 = (ws2.unsqueeze(-1)*temps2).sum(0)

        cat_s1 = torch.cat([mix1, f1], dim=2)
        cat_s2 = torch.cat([mix2, f2], dim=2)

        # get the hhats
        hhat1 = (self.sep_out1(cat_s1))
        hhat2 = (self.sep_out2(cat_s2))

        xhat1 = F.softplus(hhat1)
        xhat2 = F.softplus(hhat2)

        return xhat1, xhat2



''' LSTM with Attention and Shared Architecture'''
class lstm_att_share(base_model): 
    def __init__(self, arguments, K, Kdis, Linput):
        super(lstm_att_share, self).__init__(arguments, K, Kdis, Linput)
        

        dropout=self.arguments.dropout
        self.ntemp = self.arguments.ntemp

        self.templates = nn.Linear(2*self.ntemp, Kdis, bias=False)
        
        self.sel1 = nn.Linear(4*K, 2*self.ntemp)
        self.sel2 = nn.Linear(4*K, 2*self.ntemp)

        self.sep_rnn = nn.LSTM(input_size = Linput,
                               hidden_size = 2*K,
                               num_layers=arguments.num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)

        hidden_size = 2*Kdis + 4*K 
        if arguments.num_layers==1:
            self.sep_out = self.Dense(hidden_size, 2*Linput, dropout=dropout, activation=self.activation)
        elif arguments.num_layers==2:
            self.sep_out = nn.Sequential(self.Dense(hidden_size, hidden_size, dropout=dropout, activation=self.activation), 
                                       self.Dense(hidden_size, 2*Linput, dropout=dropout, activation=self.activation))
    
            
    def forward(self, dt):
        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        #pdb.set_trace()
        eye = torch.eye(2*self.ntemp).cuda()
        temps = self.templates(eye).unsqueeze(1).unsqueeze(1)

        mix, _ = (self.sep_rnn(dt[0]))
        
        ws1 = F.softmax(self.sel1(mix).permute(2, 0, 1), dim=0)
        ws2 = F.softmax(self.sel2(mix).permute(2, 0, 1), dim=0)
        
        f1 = (ws1.unsqueeze(-1)*temps).sum(0)
        f2 = (ws2.unsqueeze(-1)*temps).sum(0)

        cat_s = torch.cat([mix, f1, f2], dim=2)

        # get the hhats
        h = (self.sep_out(cat_s))

        xhat1 = F.softplus(h[:,:,:self.Linput])
        xhat2 = F.softplus(h[:,:,self.Linput:])
        
        return xhat1, xhat2


''' MLP with Attention and Shared Architecture + SIDE'''
class mlp_att_share_side(base_model): 
    def __init__(self, arguments, K, Kdis, Linput):
        super(mlp_att_share_side, self).__init__(arguments, K, Kdis, Linput)
       
        self.Linput = Linput
        self.side_max = self.arguments.side_max
        dropout=self.arguments.dropout
        self.ntemp = self.arguments.ntemp
        self.ds = self.arguments.ds
        self.side_model = self.arguments.side_model
       
        ntemp = len(range(0,self.side_max,self.ds)) 

        self.dim_red = nn.Linear(Linput, 2*K)

        if self.side_model=='rnn':
            self.net1 = nn.LSTM(input_size = Linput,
                               hidden_size = K,
                               num_layers=arguments.num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)
            self.net2 = nn.LSTM(input_size = Linput,
                               hidden_size = K,
                               num_layers=arguments.num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)
        elif self.side_model == 'mlp':
            self.net1 = GatedDense(Linput, 2*K, dropout=dropout, 
                                   activation=self.activation)
            self.net2 = GatedDense(Linput, 2*K, dropout=dropout, 
                                   activation=self.activation)



        hidden_size = 2*(K+2*K)
        if arguments.num_layers==1:
            self.sep = self.Dense( hidden_size, 2*Linput, dropout=dropout, activation=self.activation)
        elif arguments.num_layers==2:
            self.sep = nn.Sequential(self.Dense(hidden_size, hidden_size, dropout=dropout, activation=self.activation), 
                                     self.Dense(hidden_size, 2*Linput,    dropout=dropout, activation=self.activation))
        elif arguments.num_layers==3:
            self.sep = nn.Sequential(self.Dense(hidden_size, hidden_size, dropout=dropout, activation=self.activation), 
                                     self.Dense(hidden_size, hidden_size, dropout=dropout, activation=self.activation),
                                     self.Dense(hidden_size, 2*Linput,    dropout=dropout, activation=self.activation))
    def forward(self, dt):
        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()
        
        mix = (self.dim_red(dt[0]))

        side1 = dt[6][:,:self.side_max,:]
        side2 = dt[7][:,:self.side_max,:]

        if self.side_model=='rnn':
            side1_red, _ = self.net1(side1)
            side2_red, _ = self.net2(side2)
            
            temps1 = side1_red[:,range(0,self.side_max,self.ds),:]
            temps2 = side2_red[:,range(0,self.side_max,self.ds),:]

            ws1 = F.softmax((temps1.unsqueeze(1) * mix.unsqueeze(2)).sum(-1), dim=2)
            ws2 = F.softmax((temps2.unsqueeze(1) * mix.unsqueeze(2)).sum(-1), dim=2)

            f1 = ( ws1.unsqueeze(-1) * temps1.unsqueeze(1) ).sum(2)
            f2 = ( ws2.unsqueeze(-1) * temps2.unsqueeze(1) ).sum(2)
        
        elif self.side_model=='mlp':
            side1_red = self.net1(side1)
            side2_red = self.net2(side2)

            ws1 = F.softmax((side1_red.unsqueeze(1) * mix.unsqueeze(2)).sum(-1), dim=2)
            ws2 = F.softmax((side2_red.unsqueeze(1) * mix.unsqueeze(2)).sum(-1), dim=2)

            f1 = ( ws1.unsqueeze(-1) * side1_red.unsqueeze(1) ).sum(2)
            f2 = ( ws2.unsqueeze(-1) * side2_red.unsqueeze(1) ).sum(2)

        
        cat_s = torch.cat([mix, f1, f2], dim=2)

        # get the hhats
        h = (self.sep(cat_s))

        xhat1 = F.softplus(h[:,:,:self.Linput])
        xhat2 = F.softplus(h[:,:,self.Linput:])

        return xhat1, xhat2


''' LSTM with Attention and Shared Architecture + SIDE'''
class lstm_att_share_side(base_model): 
    def __init__(self, arguments, K, Kdis, Linput):
        super(lstm_att_share_side, self).__init__(arguments, K, Kdis, Linput)
        

        self.Linput = Linput
        self.side_max = self.arguments.side_max
        dropout=self.arguments.dropout
        self.ntemp = self.arguments.ntemp
        self.ds = self.arguments.ds
        self.side_model = self.arguments.side_model
       
        ntemp = len(range(0,self.side_max,self.ds)) 

        self.sep_rnn = nn.LSTM(input_size = Linput,
                               hidden_size = 2*K,
                               num_layers=arguments.num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)

        if self.side_model=='rnn':
            self.net1 = nn.LSTM(input_size = Linput,
                               hidden_size = 2*K,
                               num_layers=arguments.num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)
            self.net2 = nn.LSTM(input_size = Linput,
                               hidden_size = 2*K,
                               num_layers=arguments.num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)
        
        elif self.side_model == 'mlp':
            self.net1 = GatedDense(Linput, 4*K, dropout=dropout, 
                                   activation=self.activation)
            self.net2 = GatedDense(Linput, 4*K, dropout=dropout, 
                                   activation=self.activation)
        
        hidden_size = 2*3*2*K
        if arguments.num_layers==1:
            self.sep_out = self.Dense(hidden_size, 2*Linput, dropout=dropout, activation=self.activation)
        elif arguments.num_layers==2:
            self.sep_out = nn.Sequential(self.Dense(hidden_size, hidden_size, dropout=dropout, activation=self.activation), 
                                         self.Dense(hidden_size, 2*Linput,    dropout=dropout, activation=self.activation))
    
            
    def forward(self, dt):
        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        mix, _ = (self.sep_rnn(dt[0]))
        
        side1 = dt[6][:,:self.side_max,:]
        side2 = dt[7][:,:self.side_max,:]

        if self.side_model=='rnn':
            side1_red, _ = self.net1(side1)
            side2_red, _ = self.net2(side2)
            
            temps1 = side1_red[:,range(0,self.side_max,self.ds),:]
            temps2 = side2_red[:,range(0,self.side_max,self.ds),:]

            ws1 = F.softmax((temps1.unsqueeze(1) * mix.unsqueeze(2)).sum(-1), dim=2)
            ws2 = F.softmax((temps2.unsqueeze(1) * mix.unsqueeze(2)).sum(-1), dim=2)

            f1 = ( ws1.unsqueeze(-1) * temps1.unsqueeze(1) ).sum(2)
            f2 = ( ws2.unsqueeze(-1) * temps2.unsqueeze(1) ).sum(2)
        
        elif self.side_model=='mlp':
            side1_red = self.net1(side1)
            side2_red = self.net2(side2)

            ws1 = F.softmax((side1_red.unsqueeze(1) * mix.unsqueeze(2)).sum(-1), dim=2)
            ws2 = F.softmax((side2_red.unsqueeze(1) * mix.unsqueeze(2)).sum(-1), dim=2)

            f1 = ( ws1.unsqueeze(-1) * side1_red.unsqueeze(1) ).sum(2)
            f2 = ( ws2.unsqueeze(-1) * side2_red.unsqueeze(1) ).sum(2)
        
        cat_s = torch.cat([mix, f1, f2], dim=2)
        
        # get the hhats
        h = (self.sep_out(cat_s))

        xhat1 = F.softplus(h[:,:,:self.Linput])
        xhat2 = F.softplus(h[:,:,self.Linput:])
        
        return xhat1, xhat2

