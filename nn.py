import pdb
from pydoc import locate

import torch
import torch.nn as nn
import torch.nn.functional as F

# Variational Dropout
# ----------------------------------------------------------------------------------

""" Variational Dropout. Taken from https://github.com/salesforce/awd-lstm-lm """
class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        
        mask = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)
        mask.requires_grad = False
        mask = mask.expand_as(x)
        return mask * x



# Gated Dense 
# ----------------------------------------------------------------------------------

''' Neat way of doing  ResNet while changing the dimension of the representation'''
class GatedDense(nn.Module):
    def __init__(self, input_size, output_size, dropout=0, activation='relu'):
        super(GatedDense, self).__init__()

        self.activation = nn.ReLU() if activation=='relu' else nn.Sigmoid()

        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        h = self.h(x)

        g = self.activation(self.g(x))

        out = h * g 
        out = self.drop(out)

        return out

class Dense(nn.Module):
    def __init__(self, input_size, output_size, dropout=0, activation='relu'):
        super(Dense, self).__init__()

        self.activation = nn.ReLU() if activation=='relu' else nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        h = self.activation(self.h(x))

        out = h 
        out = self.drop(out)

        return out

class Linear(nn.Module):
    def __init__(self, input_size, output_size, dropout=0, activation=None):
        super(Linear, self).__init__()

        self.h = nn.Linear(input_size, output_size)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        h = self.h(x)

        out = h 
        out = self.drop(out)

        return out


# RNN Cell
# ----------------------------------------------------------------------------------

""" RNN Cell that supports Variational Dropout """
class RNNCell(nn.Module):
    def __init__(self, rnn, input_size, hidden_size):
        super(RNNCell, self).__init__()
        rnn  = locate('torch.nn.%sCell' % rnn)
        self.cell = rnn(input_size, hidden_size)

    def forward(self, x, hidden, step, dropout_p=0.5):
        # x      : (batch_size, input_size)
        # hidden : (batch_size, hidden_size)

        should_mask = self.training and dropout_p > 0.
        new_mask    = should_mask and step == 0

        if new_mask: 
            self.mask = x.data.new(*x.size()).bernoulli_(1 - dropout_p)
        if should_mask: 
            x = self.mask * x

        return self.cell(x, hidden)


# -----------------------------------------------------------------------------------------
# Old Models used for Oracle LM
# -----------------------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, num_layers, hidden_dim, args):
        super(Model, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(args.vocab_size, hidden_dim)
        rnn       = locate('torch.nn.{}.'.format(args.rnn))
        self.rnns = [ rnn(hidden_dim, hidden_dim, num_layers=1, batch_first=True) 
                        for _ in range(num_layers) ]

        self.rnns = nn.ModuleList(self.rnns)
        self.mask = None

    def step(self, x, hidden_state, step, var_drop_p=0.5):
        assert x.size(1)  == 1, 'this method is for single timestep use only'
        
        if step == 0 and self.training and var_drop_p > 0.: 
            # new sequence --> create mask
            self.mask = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - var_drop_p)
            self.mask = Variable(self.mask, requires_grad=False) / (1 - var_drop_p)

        output = x * self.mask if self.training and var_drop_p > 0. else x

        for l, rnn in enumerate(self.rnns):
            output, hidden_state = rnn(output, hidden_state)
            if self.training and var_drop_p > 0: output = output * self.mask

        return output, hidden_state 


class Generator(Model):
    def __init__(self, args, is_oracle=False):
        super(Generator, self).__init__(args.num_layers_gen, args.hidden_dim_gen, args)
        
        in_size = args.hidden_dim_gen

        self.output_layer = nn.Linear(in_size, args.vocab_size)
        self.is_oracle = is_oracle

    def forward(self, x, hidden_state=None):
        assert len(x.size()) == 2 # bs x seq_len
        ''' note that x[:, 0] is always SOS token'''

        # if only one word is given, use it as starting token, than sample from your distribution 
        teacher_force  = x.size(1) != 1
        seq_len        = x.size(1) if teacher_force else self.args.max_seq_len
        input_idx      = x[:, [0]]
        outputs, words = [], []

        for t in range(seq_len):
            # choose first token, or overwrite sampled one
            if teacher_force or t == 0: 
                input_idx = x[:, [t]]

            input = self.embedding(input_idx)
            output, hidden_state = self.step(input, hidden_state, t, \
                    var_drop_p=self.args.var_dropout_p_gen)
    
            dist = self.output_layer(output)
            alpha = self.args.alpha_train if self.training  else self.args.alpha_test
            if not self.is_oracle: 
                dist = dist * alpha
   
            if not teacher_force:
                if self.training or self.is_oracle or True: 
                    input_idx = Categorical(logits=dist.squeeze(1)).sample().unsqueeze(1)
                else: 
                    input_idx = dist.squeeze(1).max(dim=1)[1].unsqueeze(1)
                words += [input_idx]

            # note : these are 1-off with input, or aligned with target
            outputs += [dist] 
        
        if not teacher_force : 
            words = torch.cat(words, dim=1)
        
        logits = torch.cat(outputs, dim=1)
        return logits, words


