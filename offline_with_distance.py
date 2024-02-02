#!/usr/bin/env python
from collections import defaultdict
from IPython import embed
from mim import train
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
from offline_triplet_sampling import generate_triplets, generate_positive_sample_lookup

used_data = pd.read_csv('/home/ids/yukliu/OpenTest/data/dichasus.csv')

target_cols = ['x', 'y']
train_cols = [col for col in used_data.columns if col not in ['Unnamed: 0', 'time', 'x', 'y']]
dim_feature = len(train_cols)


locations = used_data[target_cols].values
csi_data = used_data[train_cols].values

location_index_map = defaultdict()
for i, loc in enumerate(locations):
    location_index_map[i] = loc

lookup = generate_positive_sample_lookup(location_index_map)
generate_triplets(csi_data, location_index_map, d=.5)

# test_label = test_data[target_cols].values
# train_label = train_data[target_cols].values
# embed_dim = 2
# batch_size = 256
# margin = 2

# train_dataset = OfflineTriplet(train_data[train_cols], train_data[target_cols])
# triplet_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

# test_dataset = OfflineTriplet(test_data[train_cols], test_data[target_cols])
# triplet_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


dim_feature = 1024
embed_dim = 2
batch_size = 256
margin = 2

opendata_triplets = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets.npy')
opendata_triplets = np.swapaxes(opendata_triplets, 0, 1)
fake_y = np.arange(len(opendata_triplets))[:, None]

train_triplet, test_triplet, y_train, y_test = train_test_split(opendata_triplets, fake_y, train_size=.75, random_state=24)

train_dataset = Measurement(train_triplet, y_train)
triplet_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

test_dataset = Measurement(test_triplet, y_test)
triplet_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


embedding_net = ANN(dim_feature, embed_dim, hiddens=[512, 256, 128, 64])
triplet_net = TripletNet(embedding_net)

loss_fn = TripletLoss(margin=margin)
lr = 1e-3
optimizer = optim.Adam(triplet_net.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 500

cuda = torch.cuda.is_available()
suffix = 'OpenDataTest_OfflineMining'

print(f'Model training started. The GPU availability is {cuda}')
start_time = time.time()
train_ls, test_ls = training(triplet_train_loader, triplet_test_loader, triplet_net, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

end_time = time.time()
print('--'*20)
print('**' * 7 + f'Training time {(end_time - start_time)//60} mins' + '**' * 7)
torch.save(triplet_net.state_dict(), f'./checkpoints/Triplet_emb_{embed_dim}_m_{margin}_bs_{batch_size}_{suffix}.sav')


tensor_test = torch.FloatTensor(test_triplet).cuda() if cuda else torch.FloatTensor(test_triplet)
tensor_train = torch.FloatTensor(train_triplet).cuda() if cuda else torch.FloatTensor(train_triplet)


train_embedding = triplet_net.get_embedding(tensor_train[:, 0]).cpu().detach().numpy() #if cuda else triplet_net.get_embedding(tensor_train).detach().numpy()
test_embedding = triplet_net.get_embedding(tensor_test[:, 0]).cpu().detach().numpy() #if cuda else triplet_net.get_embedding(tensor_test).cpu().detach().numpy()


np.save(f'./data/test_embedding_{embed_dim}_m_{margin}_bs_{batch_size}_{suffix}', test_embedding)
np.save(f'./data/train_embedding_{embed_dim}_m_{margin}_bs_{batch_size}_{suffix}', train_embedding)
np.save(f'./data/test_labels_{embed_dim}_m_{margin}_bs_{batch_size}_{suffix}', y_test)
np.save(f'./data/train_labels_{embed_dim}_m_{margin}_bs_{batch_size}_{suffix}', y_train)

np.save(f'./data/train_loss_{embed_dim}_m_{margin}_bs_{batch_size}_{suffix}', train_ls)
np.save(f'./data/test_loss_{embed_dim}_m_{margin}_bs_{batch_size}_{suffix}', test_ls)