import numpy as np
import matplotlib.pyplot as plt


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


def embedding_distance(cc_arr, locations, num=20000):
    d_ls = []
    c_ls = []
    for i in range(num):
        ids = np.random.choice(len(cc_arr), 2)
        c1, c2 = cc_arr[ids[0]], cc_arr[ids[1]]
        metric_d = np.linalg.norm(c1-c2)
        d = np.linalg.norm(locations[ids[0]]-locations[ids[1]])
        d_ls.append(d)
        c_ls.append(metric_d)
    return np.array(d_ls), np.array(c_ls)


def draw_ci_plot(d, c, ax=plt, label=None):
    bins, medians, perc_25, perc_75 = distance_ci_visualization(d, c)
    ax.plot(bins, medians, label=label)
    ax.fill_between(bins, perc_25, perc_75, alpha=0.5)
    ax.xlabel('Euclidean distance')
    ax.ylabel('Metric distance')
    ax.grid()