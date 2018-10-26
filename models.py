import torch
import numpy
import visdom
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

class sourcesep_net_dis_ff_dis_rnn(nn.Module):
    def __init__(self, arguments, Krnn, Kdis, Linput):
        super(sourcesep_net_dis_ff_dis_rnn, self).__init__()
        self.Kdis = Kdis
        self.Linput = Linput
        self.arguments = arguments
        self.pper = arguments.plot_interval

        self.dis_rnn = nn.LSTM(input_size=Linput,
                               hidden_size=2*Kdis,
                               num_layers=1, 
                               batch_first=True,
                               bidirectional=True)
        
        #self.sep_rnn = nn.LSTM(input_size=Linput + 2*Kdis, 
        #                       hidden_size=Krnn,
        #                       num_layers=2,
        #                       batch_first=True,
        #                       bidirectional=True)
        self.sep_rnn1 = nn.Linear(Linput + 2*Kdis, 2*Krnn)
        self.sep_rnn2 = nn.Linear(Linput + 2*Kdis, 2*Krnn)

        self.sep_out1 = nn.Linear(2*Krnn + 2*Kdis, Linput)
        self.sep_out2 = nn.Linear(2*Krnn + 2*Kdis, Linput)


    def forward(self, dt):
        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        
        Kdis = self.Kdis
        h1,_ = self.dis_rnn(dt[0]) 
        h1resize = h1.view(h1.size(0), h1.size(1), 2, h1.size(2)//2)
        ffor = h1resize[:, -1, 0, :] 
        f1for = ffor[:, :Kdis]
        f2for = ffor[:, Kdis:]

        fba = h1resize[:, 0, 1, :] 
        f1ba = fba[:, :Kdis]
        f2ba = fba[:, Kdis:]

        #combine fs for forward and backward 
        f1 = torch.cat([f1for, f1ba], dim=1)
        f2 = torch.cat([f2for, f2ba], dim=1)

        cat_s1 = self.cat_disvar(dt[0], f1)
        cat_s2 = self.cat_disvar(dt[0], f2)
        
        # get the hhats
        hhat1 = self.sep_rnn1(cat_s1)
        hhat2 = self.sep_rnn2(cat_s2)

        hhat1_f = self.cat_disvar(hhat1, f1)
        hhat2_f = self.cat_disvar(hhat2, f2)

        # get the network outputs
        xhat1 = F.softplus(self.sep_out1(hhat1_f))
        xhat2 = F.softplus(self.sep_out2(hhat2_f))

        return xhat1, xhat2

    def trainer(self, loader, opt, vis):
        
        EP = self.arguments.EP_train
        errs = []
        for ep in range(EP):
            for i, dt in enumerate(loader):
                xhat1, xhat2 = self.forward(dt)
                
                err = (dt[2] - xhat1).pow(2).mean() + (dt[3] - xhat2).pow(2).mean()
                err.backward()
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


class sourcesep_net_dis_ff_dis_ff(sourcesep_net_dis_ff_dis_rnn): 
    def __init__(self, arguments, Krnn, Kdis, Linput):
        super(sourcesep_net_dis_ff_dis_ff, self).__init__(arguments, Krnn, Kdis, Linput)

        self.dis_rnn = nn.Linear(Linput, 4*Kdis)

    def forward(self, dt):
        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        Kdis = self.Kdis
        h1 = torch.tanh(self.dis_rnn(dt[0].mean(1)))

        #combine fs for forward and backward 
        f1 = h1[:, :2*Kdis]
        f2 = h1[:, 2*Kdis:]

        cat_s1 = self.cat_disvar(dt[0], f1)
        cat_s2 = self.cat_disvar(dt[0], f2)
        
        # get the hhats
        hhat1 = self.sep_rnn1(cat_s1)
        hhat2 = self.sep_rnn2(cat_s2)

        hhat1_f = self.cat_disvar(hhat1, f1)
        hhat2_f = self.cat_disvar(hhat2, f2)

        # get the network outputs
        xhat1 = F.softplus(self.sep_out1(hhat1_f))
        xhat2 = F.softplus(self.sep_out2(hhat2_f))

        return xhat1, xhat2

class sourcesep_net_disatt_ff_dis_ff(sourcesep_net_dis_ff_dis_rnn): 
    def __init__(self, arguments, Krnn, Kdis, Linput, sidedata):
        super(sourcesep_net_disatt_ff_dis_ff, self).__init__(arguments, Krnn, Kdis, Linput)

        self.dis_rnn = None
        self.dim_red = nn.Linear(Linput, Kdis)
        self.sidedata = sidedata
        self.sidedata[0] = sidedata[0].cuda()
        self.sidedata[1] = sidedata[1].cuda()

        #nchoose = 200
        #inds = np.random.choice(self.nside, nchoose)
        self.sidedata[0] = sidedata[0][7:9]  #.reshape(-1, Linput)[inds]
        #inds = np.random.choice(self.nside, nchoose)
        self.sidedata[1] = sidedata[1][7:9]  #.reshape(-1, Linput)[inds]
        #self.nside = nchoose
        self.nside = sidedata[0].size(0) * sidedata[0].size(1)
        
        self.sel1 = None #nn.Linear(Kdis, 1)
        self.sel2 = None #nn.Linear(Kdis, 1)

        self.sep_rnn1 = nn.Linear(Kdis + Linput, Linput)
        self.sep_rnn2 = nn.Linear(Kdis + Linput, Linput)
        self.sep_out1 = None
        self.sep_out2 = None
     
    def forward(self, dt):
        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        #ws1 = self.sel1(dt[0].mean(1)).t().unsqueeze(-1)
        #ws2 = self.sel2(dt[0].mean(1)).t().unsqueeze(-1)
        side1 = self.dim_red(self.sidedata[0][:3]).reshape(-1, self.Kdis).unsqueeze(1).unsqueeze(1)
        side2 = self.dim_red(self.sidedata[1][:3]).reshape(-1, self.Kdis).unsqueeze(1).unsqueeze(1)

        mix = self.dim_red(dt[0]).unsqueeze(0)

        ws1 = F.softmax((side1*mix), dim=0)
        ws2 = F.softmax((side2*mix), dim=0)
        #ws1 = F.softmax(self.sel1(dt[0]).permute(2, 0, 1), dim=0).unsqueeze(-1) 
        #ws2 = F.softmax(self.sel2(dt[0]).permute(2, 0, 1), dim=0).unsqueeze(-1)
        
        f1 = (ws1*side1).sum(0)
        f2 = (ws2*side2).sum(0)

        #f1 = torch.tanh(self.dis_rnn(wsh1))
        #f2 = torch.tanh(self.dis_rnn2(wsh2))

        #combine fs for forward and backward 
        #f1 = h1[:, :2*Kdis]
        #f2 = h1[:, 2*Kdis:]

        cat_s1 = torch.cat([dt[0], f1], dim=2)
        cat_s2 = torch.cat([dt[0], f2], dim=2)

        #cat_s1 = dt[0]*dt[2]
        #cat_s2 = dt[0]*dt[3]

        #cat_s1 = self.cat_disvar(dt[0], dt[2])
        #cat_s2 = self.cat_disvar(dt[0], dt[3])
        
        # get the hhats
        xhat1 = F.softplus(self.sep_rnn1(cat_s1))
        xhat2 = F.softplus(self.sep_rnn2(cat_s2))

        #hhat1_f = self.cat_disvar(hhat1, f1)
        #hhat2_f = self.cat_disvar(hhat2, f2)
        #hhat1_f = torch.cat([hhat1, f1], dim=2)
        #hhat2_f = torch.cat([hhat2, f2], dim=2)

        # get the network outputs
        #xhat1 = F.softplus(self.sep_out1(hhat1_f))
        #xhat2 = F.softplus(self.sep_out2(hhat2_f))

        return xhat1, xhat2

class sourcesep_net_disside_ff_dis_ff(sourcesep_net_dis_ff_dis_rnn): 
    def __init__(self, arguments, Krnn, Kdis, Linput, sidedata):
        super(sourcesep_net_disside_ff_dis_ff, self).__init__(arguments, Krnn, Kdis, Linput)

        #self.dis_rnn = nn.Linear(Linput, 2*Kdis)
        #self.dis_rnn2 = nn.Linear(Linput, 2*Kdis)
        self.sidedata = sidedata
        self.sidedata[0] = sidedata[0].cuda()
        self.sidedata[1] = sidedata[1].cuda()

        #nchoose = 200
        #inds = np.random.choice(self.nside, nchoose)
        self.sidedata[0] = sidedata[0][7:9]  #.reshape(-1, Linput)[inds]
        #inds = np.random.choice(self.nside, nchoose)
        self.sidedata[1] = sidedata[1][7:9]  #.reshape(-1, Linput)[inds]
        #self.nside = nchoose
        self.nside = sidedata[0].size(0) * sidedata[0].size(1)
        
        self.sel1 = nn.Linear(Linput, self.nside)
        self.sel2 = nn.Linear(Linput, self.nside)

        self.sep_rnn1 = nn.Linear(2*Linput, Linput)
        self.sep_rnn2 = nn.Linear(2*Linput, Linput)
        
        #self.sep_out1 = nn.Linear(2*Krnn + 2*Kdis, Linput)
        #self.sep_out2 = nn.Linear(2*Krnn + 2*Kdis, Linput)

    def forward(self, dt):
        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        Kdis = self.Kdis
        #ws1 = self.sel1(dt[0].mean(1)).t().unsqueeze(-1)
        #ws2 = self.sel2(dt[0].mean(1)).t().unsqueeze(-1)
        ws1 = F.softmax(self.sel1(dt[0]).permute(2, 0, 1), dim=0).unsqueeze(-1) 
        ws2 = F.softmax(self.sel2(dt[0]).permute(2, 0, 1), dim=0).unsqueeze(-1)
        
        h1 = self.sidedata[0].reshape(-1, self.Linput).unsqueeze(1).unsqueeze(1)
        h2 = self.sidedata[1].reshape(-1, self.Linput).unsqueeze(1).unsqueeze(1)

        #h1 = self.sidedata[0].unsqueeze(1)
        #h2 = self.sidedata[1].unsqueeze(1)

        f1 = (ws1*h1).sum(0)
        f2 = (ws2*h2).sum(0)

        #f1 = torch.tanh(self.dis_rnn(wsh1))
        #f2 = torch.tanh(self.dis_rnn2(wsh2))

        #combine fs for forward and backward 
        #f1 = h1[:, :2*Kdis]
        #f2 = h1[:, 2*Kdis:]

        cat_s1 = torch.cat([dt[0], f1], dim=2)
        cat_s2 = torch.cat([dt[0], f2], dim=2)

        #cat_s1 = self.cat_disvar(dt[0], f1)
        #cat_s2 = self.cat_disvar(dt[0], f2)
        
        # get the hhats
        xhat1 = F.softplus(self.sep_rnn1(cat_s1))
        xhat2 = F.softplus(self.sep_rnn2(cat_s2))

        #hhat1_f = self.cat_disvar(hhat1, f1)
        #hhat2_f = self.cat_disvar(hhat2, f2)
        #hhat1_f = torch.cat([hhat1, f1], dim=2)
        #hhat2_f = torch.cat([hhat2, f2], dim=2)

        # get the network outputs
        #xhat1 = F.softplus(self.sep_out1(hhat1_f))
        #xhat2 = F.softplus(self.sep_out2(hhat2_f))

        return xhat1, xhat2


class sourcesep_net_st_ff(sourcesep_net_dis_ff_dis_rnn): 
    def __init__(self, arguments, Krnn, Kdis, Linput):
        super(sourcesep_net_st_ff, self).__init__(arguments, Krnn, Kdis, Linput)
        self.dis_rnn = None
        self.pper = arguments.plot_interval

        self.sep_rnn1 = nn.Linear(Linput, Linput)
        self.sep_rnn2 = nn.Linear(Linput, Linput)

        #self.sep_out1 = nn.Linear(2*Krnn, Linput)
        #self.sep_out2 = nn.Linear(2*Krnn, Linput)

    def forward(self, dt):

        if self.arguments.cuda:
            for i, d in enumerate(dt):
                dt[i] = d.cuda()

        xhat1 = F.softplus(self.sep_rnn1(dt[0]))
        xhat2 = F.softplus(self.sep_rnn2(dt[0]))

        # get the network outputs
        #xhat1 = F.softplus(self.sep_out1(hhat1))
        #xhat2 = F.softplus(self.sep_out2(hhat2))
        return xhat1, xhat2


class sourcesep_net_st_rnn(sourcesep_net_dis_ff_dis_rnn): 
    def __init__(self, arguments, Krnn, Kdis, Linput):
        super(sourcesep_net_st_rnn, self).__init__(arguments, Krnn, Kdis, Linput)
        self.dis_rnn = None
        self.pper = arguments.plot_interval

        self.sep_rnn1 = nn.LSTM(input_size = Linput,
                               hidden_size = Krnn,
                               num_layers=1, 
                               batch_first=True,
                               bidirectional=True)
        self.sep_rnn2 = nn.LSTM(input_size = Linput,
                               hidden_size = Krnn,
                               num_layers=1, 
                               batch_first=True,
                               bidirectional=True)

        self.sep_out1 = nn.Linear(2*Krnn, Linput)
        self.sep_out2 = nn.Linear(2*Krnn, Linput)

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


