#!/usr/bin/env python
from IPython import embed
import pandas as pd
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from libs.loss import *
from libs.network import *
from libs.trainner import *
from libs.siamese_trainner import *
from libs.tools import dist_from_geo, cdf_plot, dataHandler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsRegressor as KNR
import argparse


suffix = 'Pairs_middle_big'

hiddens=[512, 256, 128, 64, 32, 16]

start = 512
embed_dim = 2
# hiddens = [int(start / (2 ** i)) for i in range(100) if start / (2 ** i) >= embed_dim]

batch_size = 256
margin = 2
opendata_pairs = np.load('/home/ids/yukliu/OpenTest/data/opendata_pairs.npy')


fake_y = np.arange(len(opendata_pairs))[:, None]
dim_feature = opendata_pairs.shape[-1]


train_pair, test_pair, y_train, y_test = train_test_split(opendata_pairs, fake_y, train_size=.75, random_state=24)

train_dataset = Measurement(train_pair, y_train)
pair_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

test_dataset = Measurement(test_pair, y_test)
pair_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

embedding_net = ANN(dim_feature, embed_dim, hiddens=hiddens)
pair_net = PairNet(embedding_net)
lr = 1e-2
optimizer = optim.Adam(pair_net.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 10
log_interval = 500
loss_fn = PairLoss()

tr_ls, te_ls = pair_training(pair_train_loader, pair_test_loader, pair_net, loss_fn, optimizer, scheduler, n_epochs, log_interval)

torch.save(pair_net.state_dict(), f'./checkpoints/Pairnet_emb_{embed_dim}_bs_{batch_size}_{suffix}.sav')