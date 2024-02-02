from collections import defaultdict
import numpy as np
from geopy.distance import geodesic
from scipy import spatial
from sklearn.neighbors import KNeighborsRegressor
from libs.tools import dist_from_geo
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def ap_indices_fabrication(pos_data, d=.3):
    # d = .3
    ap_indices = dict()
    meta_tree = spatial.cKDTree(pos_data)
    for i in range(len(pos_data)):
        proximal_index = meta_tree.query_ball_point(pos_data[i], d)
        if len(proximal_index) > 10:
            ap_indices[i] = np.random.choice(proximal_index, 10)
        else:
            ap_indices = proximal_index

    return ap_indices


def apn_indices_fabrication(pos_data, da=.3, dn=10):
    ap_indices = dict()
    an_indices = dict()
    meta_tree = spatial.cKDTree(pos_data)
    n = len(pos_data)
    all_index = np.arange(n)
    for i in range(n):
        proximal_index = meta_tree.query_ball_point(pos_data[i], da)
        
        mask = np.ones(n, dtype=bool)
        distant_anti_index = meta_tree.query_ball_point(pos_data[i], dn)
        mask[distant_anti_index] = False
        distant_index = all_index[mask]
        if len(proximal_index) > 10:
            ap_indices[i] = np.random.choice(proximal_index, 10)
        else:
            ap_indices[i] = proximal_index

        if len(distant_index) > 10:
            an_indices[i] = np.random.choice(distant_index, 10)
        else:
            an_indices[i] = distant_index

    return ap_indices, an_indices


def triplet_fabrication(csi_data, pos_data, d, num_triplets=15000):
    ap_indices = ap_indices_fabrication(pos_data, d)

    triplets_by_distance = []
    anchor_indices = list(ap_indices.keys())
    n_sample = len(anchor_indices)

    while len(triplets_by_distance) < num_triplets:
        anchor_indice = np.random.choice(anchor_indices)
        pos_indices = ap_indices[anchor_indice]
        if len(pos_indices) < 1:
            continue
        pos_indice = np.random.choice(pos_indices)
        neg_indice = np.random.randint(n_sample)

        triplets_by_distance.append((csi_data[anchor_indice], csi_data[pos_indice], csi_data[neg_indice]))
        
    return np.array(triplets_by_distance)


def triplet_fabrication_with_index(csi_data, pos_data, d=.3, num_triplets=15000):
    ap_indices = ap_indices_fabrication(pos_data, d)

    triplets_by_distance = []
    anchor_indices = list(ap_indices.keys())
    n_sample = len(anchor_indices)
    apn_idx = []

    while len(triplets_by_distance) < num_triplets:
        anchor_indice = np.random.choice(anchor_indices)
        pos_indices = ap_indices[anchor_indice]
        if len(pos_indices) < 1:
            continue

        pos_indice = np.random.choice(pos_indices)
        neg_indice = np.random.randint(n_sample)

        triplets_by_distance.append((csi_data[anchor_indice], csi_data[pos_indice], csi_data[neg_indice]))
        apn_idx.append((anchor_indice, pos_indice, neg_indice))
        
    return np.array(triplets_by_distance), np.array(apn_idx)


def triplet_hn_fabrication(csi_data, pos_data, d, dn=10, num_triplets=15000):
    ap_indices, an_indices = apn_indices_fabrication(pos_data, d, dn)

    triplets_by_distance = []
    anchor_indices = list(ap_indices.keys())
    n_sample = len(anchor_indices)

    while len(triplets_by_distance) < num_triplets:
        anchor_indice = np.random.choice(anchor_indices)
        pos_indices = ap_indices[anchor_indice]
        neg_indices = an_indices[anchor_indice]
        if len(pos_indices) < 1:
            continue
        pos_indice = np.random.choice(pos_indices)
        
        if len(neg_indices) < 1:
            continue
        neg_indice = np.random.choice(neg_indices)

        triplets_by_distance.append((csi_data[anchor_indice], csi_data[pos_indice], csi_data[neg_indice]))
        
    return np.array(triplets_by_distance)
    

def localisation_cc(cc, pos, n_neigh=5, geo=False, d_fn=np.linalg.norm):
    cc_train, cc_test, pos_train, pos_test = train_test_split(cc, pos, random_state=24)
    knr = KNeighborsRegressor(n_neighbors=n_neigh)
    knr.fit(cc_train, pos_train)

    pos_pred = knr.predict(cc_test)
    if geo:
        dist_err = dist_from_geo(pos_pred, pos_test)
    else:
        dist_err = d_fn(pos_pred - pos_test, axis=1)

    return dist_err


def distance_sampling(csi_data, locations, metric_fn, num=20000):
    # num = 20000
    d_ls = []
    c_ls = []
    for i in range(num):
        ids = np.random.choice(len(csi_data), 2)
        c1, c2 = csi_data[ids[0]], csi_data[ids[1]]
        metric_d = metric_fn(c1, c2).detach().numpy()[0]
        d = np.linalg.norm(locations[ids[0]]-locations[ids[1]])
        d_ls.append(d)
        c_ls.append(metric_d)
    
    return np.array(d_ls), np.array(c_ls)


def distance_ci_visualization(distances_flat, dissimilarities_flat):
    max_distance = np.max(distances_flat)
    bins = np.linspace(0, max_distance, 200)
    bin_indices = np.digitize(distances_flat, bins)

    bin_medians = np.zeros(len(bins) - 1)
    bin_25_perc = np.zeros(len(bins) - 1)
    bin_75_perc = np.zeros(len(bins) - 1)
    for i in range(1, len(bins)):
        try:
            bin_values = dissimilarities_flat[bin_indices == i]
            bin_25_perc[i - 1], bin_medians[i - 1], bin_75_perc[i - 1] = np.percentile(bin_values, [25, 50, 75])
        except:
            break

    return bins[:i-1], bin_medians[: i-1], bin_25_perc[:i-1], bin_75_perc[:i-1]


def draw_ci_plot(d, c, ax=plt, label=None):
    bins, medians, perc_25, perc_75 = distance_ci_visualization(d, c)
    ax.plot(bins, medians, label=label)
    ax.fill_between(bins, perc_25, perc_75, alpha=0.5)
    ax.xlabel('Euclidean distance')
    ax.ylabel('Metric distance')
    ax.grid()
