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


parser = argparse.ArgumentParser()
parser.add_argument("-em", "--embed_dim", type=int, default=16)
args = parser.parse_args()

suffix = 'OpenDataTest_OfflineMining_MetricLearning'

# embed_dim = 2
embed_dim = args.embed_dim
embed_dim = 8
batch_size = 256
margin = 2
# opendata_triplets = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets_3.npy')
# opendata_triplets = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets_2.npy')
# opendata_triplets = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets_by_distance_05.npy')
# opendata_triplets = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets_42000.npy')
opendata_triplets = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets_new.npy')

# opendata_triplets = np.load('/home/ids/yukliu/Test/data/triplets_fabricated.npy')
if np.shape(opendata_triplets)[1] != 3:
    opendata_triplets = np.swapaxes(opendata_triplets, 0, 1)
    print(f'axis 0 1 swaped, shape {opendata_triplets.shape}')

fake_y = np.arange(len(opendata_triplets))[:, None]
dim_feature = opendata_triplets.shape[-1]

train_triplet, test_triplet, y_train, y_test = train_test_split(opendata_triplets, fake_y, train_size=.75, random_state=24)

train_dataset = Measurement(train_triplet, y_train)
triplet_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

test_dataset = Measurement(test_triplet, y_test)
triplet_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# hiddens=[512, 256, 128, 64, 32]
hiddens=[512, 256, 64, 32]

start = 512
end = embed_dim
# hiddens = [int(start / (2 ** i)) for i in range(100) if start / (2 ** i) >= end]
# embedding_net = ANN(dim_feature, embed_dim, hiddens=hiddens)
print(hiddens, suffix)
triplet_net = MetricNet(dim_feature, out_dim=1, hiddens=hiddens)

loss_fn = TripletMetricLoss(margin=margin)
lr = 1e-6
optimizer = optim.Adam(triplet_net.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 500

cuda = torch.cuda.is_available()

# print(f'Model training started. The GPU availability is {cuda}')
start_time = time.time()
train_ls, test_ls = training(triplet_train_loader, triplet_test_loader, triplet_net, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

end_time = time.time()
print('--'*20)
print('**' * 7 + f'Training time {(end_time - start_time)//60} mins {(end_time - start_time)%60} s' + '**' * 7)
torch.save(triplet_net.state_dict(), f'./checkpoints/Triplet_emb_{embed_dim}_m_{margin}_bs_{batch_size}_{suffix}.sav')


# tensor_test = torch.FloatTensor(test_triplet).cuda() if cuda else torch.FloatTensor(test_triplet)
# tensor_train = torch.FloatTensor(train_triplet).cuda() if cuda else torch.FloatTensor(train_triplet)


# train_embedding = triplet_net.get_embedding(tensor_train[:, 0]).cpu().detach().numpy() #if cuda else triplet_net.get_embedding(tensor_train).detach().numpy()
# test_embedding = triplet_net.get_embedding(tensor_test[:, 0]).cpu().detach().numpy() #if cuda else triplet_net.get_embedding(tensor_test).cpu().detach().numpy()


# np.save(f'./data/test_embedding_{embed_dim}_m_{margin}_bs_{batch_size}_{suffix}', test_embedding)
# np.save(f'./data/train_embedding_{embed_dim}_m_{margin}_bs_{batch_size}_{suffix}', train_embedding)
# np.save(f'./data/test_labels_{embed_dim}_m_{margin}_bs_{batch_size}_{suffix}', y_test)
# np.save(f'./data/train_labels_{embed_dim}_m_{margin}_bs_{batch_size}_{suffix}', y_train)

# np.save(f'./data/train_loss_{embed_dim}_m_{margin}_bs_{batch_size}_{suffix}', train_ls)
# np.save(f'./data/test_loss_{embed_dim}_m_{margin}_bs_{batch_size}_{suffix}', test_ls)