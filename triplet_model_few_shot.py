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

# used_data = pd.read_csv('/home/ids/yukliu/OpenTest/data/dichasus.csv')

# train_data, test_data = train_test_split(used_data, train_size=.75, random_state=24)
# target_cols = ['x', 'y']
# train_cols = [col for col in train_data.columns if col not in ['Unnamed: 0', 'time', 'x', 'y']]
# test_label = test_data[target_cols].values
# train_label = train_data[target_cols].values
# dim_feature = len(train_cols)
# embed_dim = 2
# batch_size = 256
# margin = 2

# train_dataset = OfflineTriplet(train_data[train_cols], train_data[target_cols])
# triplet_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

# test_dataset = OfflineTriplet(test_data[train_cols], test_data[target_cols])
# triplet_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

parser = argparse.ArgumentParser()
parser.add_argument("-em", "--embed_dim", type=int, default=2)
args = parser.parse_args()

alpha = 2
suffix = f'Triplet_few_shot_{alpha}'
# suffix = 'OpenDataTest_OfflineMining'
# suffix = 'OpenDataTest_OfflineMining_new'

embed_dim = args.embed_dim
embed_dim = 2
hiddens=[512, 256, 128, 64]

start = 512
end = embed_dim
hiddens = [int(start / (2 ** i)) for i in range(100) if start / (2 ** i) >= end]

batch_size = 256
margin = 2
# opendata_triplets = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets_3.npy')
# opendata_triplets = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets_2.npy')
# opendata_triplets = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets_by_distance_03.npy')
# opendata_triplets = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets_by_distance_HN.npy')
# opendata_triplets = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets_by_distance_HN_15.npy')

opendata_triplets = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets.npy')
triplets_idx = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets_idx.npy')

# opendata_triplets = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets_new.npy')
# opendata_triplets = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets_42000.npy')

# opendata_triplets = np.load('/home/ids/yukliu/OpenTest/data/opendata_triplets_big.npy')

# opendata_triplets = np.load('/home/ids/yukliu/Test/data/triplets_fabricated.npy')
if np.shape(opendata_triplets)[1] != 3:
    opendata_triplets = np.swapaxes(opendata_triplets, 0, 1)
    print(f'axis 0 1 swaped, shape {opendata_triplets.shape}')


opendata_pairs = np.load('/home/ids/yukliu/OpenTest/data/opendata_pairs_with_idx.npy')
meta_data = pd.read_csv('/home/ids/yukliu/OpenTest/data/dichasus.csv')
locations = meta_data[['x', 'y']].values

# Few shot data setting
num_dataset = len(locations)
size_data = num_dataset // 2
data_idx = np.random.choice(num_dataset, size_data)

# fake_y = np.arange(len(opendata_triplets))[:, None]
dim_feature = opendata_triplets.shape[-1]


train_triplet, test_triplet, y_train, y_test = train_test_split(opendata_triplets, triplets_idx, train_size=.75, random_state=24)

train_dataset = Measurement(train_triplet, y_train)
triplet_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

test_dataset = Measurement(test_triplet, y_test)
triplet_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


print(hiddens, suffix)
embedding_net = ANN(dim_feature, embed_dim, hiddens=hiddens)


triplet_net = TripletNet(embedding_net)

# loss_fn = TripletLoss(margin=margin)
loss_fn = TripletFewShotLoss(margin, data_idx, locations, alpha)

lr = 1e-2
optimizer = optim.Adam(triplet_net.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 500

cuda = torch.cuda.is_available()

print(f'Model training started. The GPU availability is {cuda}')
start_time = time.time()
train_ls, test_ls = training(triplet_train_loader, triplet_test_loader, triplet_net, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

end_time = time.time()
print('--'*20)
print('**' * 7 + f'Training time {(end_time - start_time)//60} mins {(end_time - start_time)%60} s' + '**' * 7)
torch.save(triplet_net.state_dict(), f'./checkpoints/Triplet_emb_{embed_dim}_m_{margin}_bs_{batch_size}_{suffix}.sav')