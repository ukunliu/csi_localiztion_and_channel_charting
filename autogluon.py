import pandas as pd
from libs.tools import dist_from_geo, cdf_plot
from libs.autogluon_multilabel import MultilabelPredictor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import numpy as np
import subprocess as sp


# dire = '/home/ids/yukliu/Triplet_metric_learning/'
# meta_data = pd.read_csv(dire+'cir_rssi_df.csv')
# city = 'newyork'
# used_data = meta_data[meta_data.city==city]
# suffix = 'OpenData_Autogluon_3'
suffix = 'OpenData_Autogluon_15000'
meta_data = pd.read_csv('/home/ids/yukliu/OpenTest/data/dichasus.csv')
# meta_data = pd.read_csv('/home/ids/yukliu/OpenTest/data/Opendata_3.csv')
csi_cols = [col for col in meta_data.columns if col not in ['Unnamed: 0', 'time', 'x', 'y']]
target_cols = ['x', 'y']
used_cols = [col for col in meta_data.columns if col not in ['Unnamed: 0']]

locations = meta_data[['x', 'y']].values
used_data = meta_data[used_cols].sample(15000)
train_data, test_data = train_test_split(used_data, train_size=.75, random_state=24)
# target_cols = ['Lat', 'Lon']
# train_cols = [col for col in train_data.columns if col not in [target_cols, 'city']]


# CIR_FEATURES = [col for col in used_data.columns if col.startswith('CIR')] + target_cols
# RSSI_FEATURES = [col for col in used_data.columns if col.startswith('RSSI')] + target_cols


# train_cir, train_rssi, test_cir, test_rssi = train_data[CIR_FEATURES], train_data[RSSI_FEATURES], \
#                                                 test_data[CIR_FEATURES], test_data[RSSI_FEATURES]

dim_feature = len(used_cols) - len(target_cols)


########################## Model Training ################################# 
hyperparameters = {
    'GBM': [
        {'ag_args_fit': {'num_gpus': 1}}   # Train with GPU
    ],
    'XGB': [
        {'ag_args_fit': {'num_gpus': 1}}   # Train with GPU
    ],
    'RF': [
        {'ag_args_fit': {'num_gpus': 1}}   # Train with GPU
    ]
}

num_gpus = 0

predictor = MultilabelPredictor(labels=target_cols, problem_types = ['regression', 'regression'], path='./checkpoints/')

start_time = time.time()
# sp.run('nvidia-smi')
predictor.fit(train_data=train_data, time_limit=200, num_gpus=num_gpus)


test_prediction = predictor.predict(test_data)


end_time = time.time()

print('**' * 7 + f'Running time {end_time - start_time}' + '**' * 7)
################### Evaluation ########################

d_fn = np.linalg.norm
dist_err = d_fn(test_prediction-test_data[target_cols].values, axis=1)
# dist_err = list(map(lambda a, b : d_fn(a - b), test_prediction, test_data[target_cols].values))
np.save(f'./data/dist_err_autogluon_{suffix}', dist_err)
# np.save(f'./data/dist_err_cir_autogluon_{city}', dist_errs_cir)
# np.save(f'./data/dist_err_rssi_autogluon_{city}', dist_errs_rssi)