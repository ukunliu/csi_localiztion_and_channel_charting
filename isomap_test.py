from sklearn.manifold import Isomap
import pandas as pd
import numpy as np

meta_data = pd.read_csv('./data/dichasus.csv')
csi_cols = [col for col in meta_data.columns if col not in ['Unnamed: 0', 'time', 'x', 'y']]
csi_data = meta_data[csi_cols].values
# locations = meta_data[['x', 'y']].values

isomap = Isomap(n_components=2)
iso_embedding = isomap.fit_transform(csi_data)
np.save('./data/iso_embedding_opendata', iso_embedding)
