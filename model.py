from __future__ import division


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict


import os, time
import numpy as np
import pandas as pd

import pickle as pkl
import math
import random
import tools
import copy

# from statistics import geometric_mean
from scipy.stats import gmean
from tqdm import tqdm
from task import generate_trials
from decimal import Decimal

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device")



class Model(nn.Module):
    def __init__(self, hp: dict):
        super(Model, self).__init__()
        self.hp = hp
        self.alpha = hp['alpha'] # 1.0 * hp['dt'] / hp['tau']
        self.rng = np.random.RandomState(hp['seed'])
        self.device = hp['device']
        self.w_rec_init = hp['w_rec_init']
        self.activation = hp['activation']

        self.in_size_model = hp['in_size_model']
        self.output_size = hp['out_size']
        
        self.hidden_size_ctx = hp['hid_size_ctx']
        self.hidden_size_hpc = hp['hid_size_hpc']

        if 'hpc_loss' in hp and hp['hpc_loss']=='recon': self.hpc_reconstruct = True
        else: self.hpc_reconstruct=False

        if self.activation == 'softplus':
            self._activation = F.softplus
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif self.activation == 'tanh':
            self._activation = torch.tanh
            self._w_in_start = 1.0
            self._w_rec_start = 1.0
        elif self.activation == 'relu':
            self._activation = torch.relu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif self.activation == 'leakyrelu':
            self._activation = nn.LeakyReLU
            self._w_in_start = 1.0
            self._w_rec_start = 0.5   
        # elif self.activation == 'power':
        #     self._activation = lambda x: tf.square(tf.nn.relu(x))
        #     self._w_in_start = 1.0
        #     self._w_rec_start = 0.01
        # elif self.activation == 'retanh':
        #     self._activation = lambda x: tf.tanh(tf.nn.relu(x))
        #     self._w_in_start = 1.0
        #     self._w_rec_start = 0.5
        else:
            raise ValueError('Unknown activation')


        self.build_model()
        self.init_weights()


    def build_model(self):
 
        self.i2h_ctx = nn.Linear(self.in_size_model, self.hidden_size_ctx)
        self.i2h_hpc = nn.Linear(self.in_size_model, self.hidden_size_hpc)

        self.h2h_ctx = nn.Linear(self.hidden_size_ctx, self.hidden_size_ctx)
        self.h2h_hpc = nn.Linear(self.hidden_size_hpc, self.hidden_size_hpc)

        self.ctx2hpc = nn.Linear(self.hidden_size_ctx, self.hidden_size_hpc)
        self.hpc2ctx = nn.Linear(self.hidden_size_hpc, self.hidden_size_ctx)

        self.ctx2out = nn.Linear(self.hidden_size_ctx, self.output_size)
        self.hpc2out = nn.Linear(self.hidden_size_hpc, self.output_size)

        self.params = nn.ModuleDict({ 
                        'cortical': nn.ModuleList([self.i2h_ctx, self.h2h_ctx, self.ctx2out]),
                        'hippocampal': nn.ModuleList([self.i2h_hpc, self.h2h_hpc, self.hpc2out]),
                        'ctx2hpc': nn.ModuleList([self.ctx2hpc]),
                        'hpc2ctx': nn.ModuleList([self.hpc2ctx]),
                    })
        
        if self.hpc_reconstruct: 
            self.hpc2reconstruct = nn.Linear(self.hidden_size_hpc, self.in_size_model)
            self.params['hpc2recon'] = nn.ModuleList([self.hpc2reconstruct])



    def forward(self, x, ctx_hid, hpc_hid, mask=None):

        if mask!=None: region, layer, lesion_mask = mask

		## Save previous hiddens for leaky control
        ctx_hid_prev = ctx_hid.clone()
        hpc_hid_prev = hpc_hid.clone()
 
		## Input 2 hidden
        x_ctx = self.i2h_ctx(x)
        x_hpc = self.i2h_hpc(x)

		## Hidden cross connections
        ctx2hpc_hid = self.ctx2hpc(ctx_hid)
        hpc2ctx_hid = self.hpc2ctx(hpc_hid)

		## Hidden to hidden
        ctx_hid = self.h2h_ctx(ctx_hid)
        hpc_hid = self.h2h_hpc(hpc_hid)

		## Layer activations (subject to mask for lesions)
        if mask!=None and layer=='h2h':
            if region=='ctx': 
                 ctx_hid = self._activation(x_ctx + ctx_hid + hpc2ctx_hid) * lesion_mask
                 hpc_hid = self._activation(x_hpc + hpc_hid + ctx2hpc_hid)
            else: 
                 ctx_hid = self._activation(x_ctx + ctx_hid + hpc2ctx_hid)
                 hpc_hid = self._activation(x_hpc + hpc_hid + ctx2hpc_hid) * lesion_mask
        else:
            ctx_hid = self._activation(x_ctx + ctx_hid + hpc2ctx_hid)
            hpc_hid = self._activation(x_hpc + hpc_hid + ctx2hpc_hid)

        ## Leaky RNN control
        ctx_hid = (1-self.alpha) * ctx_hid_prev + (self.alpha * ctx_hid)
        hpc_hid = (1-self.alpha) * hpc_hid_prev + (self.alpha * hpc_hid)

        ## Output computation
        if mask!=None and layer=='output':
            if region=='ctx': 
                ctx_out = self.ctx2out(ctx_hid) * lesion_mask
                hpc_out = self.hpc2out(hpc_hid)
            else:
                ctx_out = self.ctx2out(ctx_hid) 
                hpc_out = self.hpc2out(hpc_hid) * lesion_mask
        else:
            ctx_out = self.ctx2out(ctx_hid) 
            hpc_out = self.hpc2out(hpc_hid)

        output = ctx_out + hpc_out
        # output = torch.relu(output)

        if self.hp['loss_type'] == 'lsq': # least squares loss
            output = torch.sigmoid(output)
        else: # cross entropy
            output = F.softmax(output) # , dim=2)

        if self.hpc_reconstruct:
            hpc_reconstruction = self.hpc2reconstruct(hpc_hid)
            return (output, hpc_reconstruction), ctx_hid, hpc_hid
        return output, ctx_hid, hpc_hid


    def set_optimizer(self, opt=None):
        if opt==None:
            if 'optimizer' not in self.hp or self.hp['optimizer'] == 'adam':
                self.optimizer = optim.Adam(self.parameters(), lr=self.hp['learning_rate'], 
                                            weight_decay=self.hp['weight_decay'])
            elif self.hp['optimizer'] == 'sgd':
                self.optimizer = optim.SGD(self.parameters(), lr=self.hp['learning_rate'])
        else:
            self.optimizer = opt

    def init_weights(self):
        self.apply(self._init_weights)

        # if self.w_rec_init == 'diag':
        #     ctx_hid_weights = self._w_rec_start * np.eye(self.hidden_size_ctx)
        #     hpc_hid_weights = self._w_rec_start * np.eye(self.hidden_size_hpc)
        # elif self.w_rec_init == 'randortho':
        #     ctx_hid_weights = self._w_rec_start * gen_ortho_matrix(self.hidden_size_ctx, rng=self.rng)
        #     hpc_hid_weights = self._w_rec_start * gen_ortho_matrix(self.hidden_size_hpc, rng=self.rng)
        # elif self.w_rec_init == 'randgauss':
        #     ctx_hid_weights = (self._w_rec_start * \
        #                     self.rng.randn(self.hidden_size_ctx, self.hidden_size_ctx) / \
        #                     np.sqrt(self.hidden_size_ctx))
        #     hpc_hid_weights = (self._w_rec_start * \
        #                     self.rng.randn(self.hidden_size_hpc, self.hidden_size_hpc) / \
        #                     np.sqrt(self.hidden_size_hpc))
        # else:
        #     print("default init")
        #     self.initialize(self)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
                
	## Loss computation
    def compute_loss(self, y, y_hat, mask):
        y = y.reshape(-1, self.hp['n_output'])
        y_hat = y_hat.reshape(-1, self.hp['n_output'])

        if self.hp['loss_type'] == 'lsq':
            # criterion = nn.MSELoss() # loss = criterion(y, y_hat)
            loss = torch.square( (y-y_hat) * mask)    
        else:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(y, y_hat) * mask
            # return criterion(torch.sub(y_hat,y_loc)*mask)
        return torch.mean(loss)
    
	## Reconstruction loss (unused for report)
    def reconstruction_loss(self, x, y_hat):
        x = x.reshape(-1, self.hp['n_input'])
        y_hat = y_hat.reshape(-1, self.hp['n_input'])

        criterion = nn.MSELoss()
        loss = criterion(x, y_hat)
        return loss
        # return torch.mean(loss)

