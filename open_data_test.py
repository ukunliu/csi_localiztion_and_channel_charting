#!/usr/bin/env python
from IPython import embed
import pandas as pd
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from libs.loss import *
from libs.network import *
from libs.triplet_sampling import *
from libs.trainner import *
from libs.tools import dist_from_geo, cdf_plot, dataHandler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsRegressor as KNR
import argparse

used_data = pd.read_csv('/home/ids/yukliu/OpenTest/data/dichasus.csv')

train_data, test_data = train_test_split(used_data, train_size=.75, random_state=24)
target_cols = ['x', 'y']
train_cols = [col for col in train_data.columns if col not in [target_cols, 'time']]
test_label = test_data[target_cols].values
train_label = train_data[target_cols].values
dim_feature = len(train_cols)
# embedding_net = EmbeddingNet()


#################### Load data #####################
batch_size = 1280
train_dataset = Measurement(train_data[train_cols], train_data[target_cols])
triplet_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

test_dataset = Measurement(test_data[train_cols], test_data[target_cols])
triplet_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


#################### Model setup ####################
model_type = 'MLP'
embed_dim = 2
dist_nm = 'eu'
selector = 'SN'
suffix = 'OpenDataTest'

if model_type == 'MLP':
    model = ANN(dim_feature, embed_dim, hiddens=[128, 64, 32, 16])
elif model_type == 'CNN':
    model = Conv1d(1, embed_dim) # input channel is 1
# model = TripletNet(embedding_net)
cuda = torch.cuda.is_available()

# cuda = False # Don't use GPU
if cuda:
    model.cuda()

margin = 2
dist_dict = {'eu': pdist,
            'ef': edist}
dist_fn = dist_dict[dist_nm]
selector_nm = selector
selector_dict = {'HN': HardestNegativeTripletSelector,
                'SN': SemihardNegativeTripletSelector,
                'RN': RandomNegativeTripletSelector
                }
selector = selector_dict[selector_nm](margin, dist_fn, not cuda)
loss_fn = OnlineTripletLoss(margin, selector)

lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 10
log_interval = 50

################ GPU USAGE TEST #########################################
print(f'Model training started. The GPU availability is {cuda}')
start_time = time.time()
train_ls, test_ls = training(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
end_time = time.time()
print('--'*20)
print('**' * 7 + f'Training time {(end_time - start_time)//60} mins' + '**' * 7)
torch.save(model.state_dict(), f'./checkpoints/model_emb_{embed_dim}_m_{margin}_bs_{batch_size}_sl_{selector_nm}_{dist_fn}_{model_type}_{suffix}.sav')


################################ Evaluation ######################################
tensor_test = torch.FloatTensor(test_data[train_cols].values).cuda() if cuda else torch.FloatTensor(test_data[train_cols].values)
tensor_train = torch.FloatTensor(train_data[train_cols].values).cuda() if cuda else torch.FloatTensor(train_data[train_cols].values)


train_embedding = model(tensor_train).cpu().detach().numpy() if cuda else model(tensor_train).detach().numpy()
test_embedding = model(tensor_test).cpu().detach().numpy() if cuda else model(tensor_test).cpu().detach().numpy()

np.save(f'./data/test_embedding_{embed_dim}_m_{margin}_bs_{batch_size}_{selector_nm}_{model_type}_{dist_nm}_{suffix}', test_embedding)
np.save(f'./data/train_embedding_{embed_dim}_m_{margin}_bs_{batch_size}_{selector_nm}_{model_type}_{dist_nm}_{suffix}', train_embedding)
np.save(f'./data/test_labels_{embed_dim}_m_{margin}_bs_{batch_size}_{selector_nm}_{model_type}_{dist_nm}_{suffix}', test_label)
np.save(f'./data/train_labels_{embed_dim}_m_{margin}_bs_{batch_size}_{selector_nm}_{model_type}_{dist_nm}_{suffix}', train_label)

np.save(f'./data/train_loss_{embed_dim}_m_{margin}_bs_{batch_size}_{selector_nm}_{model_type}_{dist_nm}_{suffix}', train_ls)
np.save(f'./data/test_loss_{embed_dim}_m_{margin}_bs_{batch_size}_{selector_nm}_{model_type}_{dist_nm}_{suffix}', test_ls)