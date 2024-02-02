import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from libs.tools import *
from libs.network import *
import torch
from geopy.distance import geodesic
from scipy import spatial
import copy

bs = 256
embed_dim = 8
suffix = 'OpenDataTest_OfflineMining_MetricLearning'

dim_1 = 1024
hiddens=[512, 256, 64]

triplet_net_tmp = MetricNet(dim_1, out_dim=1, hiddens=hiddens)
triplet_net_tmp.load_state_dict(torch.load(f'/home/ids/yukliu/checkpoints/Triplet_emb_{embed_dim}_m_2_bs_{bs}_{suffix}.sav'))

meta_data = pd.read_csv('./data/dichasus.csv')
# meta_data = pd.read_csv('./data/Opendata_3.csv')
csi_cols = [col for col in meta_data.columns if col not in ['Unnamed: 0', 'time', 'x', 'y']]
csi_data = meta_data[csi_cols].values
locations = meta_data[['x', 'y']].values

csi_data = torch.tensor(csi_data, dtype=th.float32)
locations = th.tensor(locations, dtype=th.float32)

from sklearn.neighbors import KNeighborsRegressor


cc_train, cc_test, pos_train, pos_test = train_test_split(csi_data, locations, random_state=24)
knr = KNeighborsRegressor(n_neighbors=5, metric=triplet_net_tmp.get_embedding)
knr.fit(cc_train, pos_train)
rd_idx = np.random.choice(len(cc_test), 100)
cc_idx = cc_test[rd_idx]
d_pred = knr.predict(cc_idx)

np.save('./knn_metric_learning', d_pred)