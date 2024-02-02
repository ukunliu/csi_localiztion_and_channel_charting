from sklearn.manifold import MDS
import pandas as pd
import numpy as np

meta_data = pd.read_csv('/home/ids/yukliu/OpenTest/data/dichasus.csv').sample(10000)
csi_cols = [col for col in meta_data.columns if col not in ['Unnamed: 0', 'time', 'x', 'y']]
csi_data = meta_data[csi_cols].values
locations = meta_data[['x', 'y']].values
model = MDS(n_components=2, random_state=2)

all_data = pd.read_csv('/home/ids/yukliu/OpenTest/data/dichasus.csv')[csi_cols].values
out = model.fit_transform(all_data)

np.save('./data/mds_out_2', out)