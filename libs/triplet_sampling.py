from itertools import combinations
from geopy.distance import geodesic
import numpy as np
import torch
import matplotlib
# matplotlib.use('Agg')

def geo_loss(y_true, y_pred):
    """
    Calculate the Haversine distance loss between two sets of latitude and longitude coordinates.
    Assumes y_true and y_pred have shape (batch_size, 2), where each row represents (latitude, longitude).
    """
    # Convert latitude and longitude from degrees to radians
    y_true_r = torch.deg2rad(y_true)
    y_pred_r = torch.deg2rad(y_pred)

    # Haversine formula
    dlon = y_pred_r[1] - y_true_r[1]
    dlat = y_pred_r[0] - y_true_r[0]

    r = 6371000  # Radius of Earth in kilometers
    a = torch.square(torch.sin(dlat/2)) + torch.cos(y_true[1]) * torch.cos(y_pred[1]) * torch.square(torch.sin(dlon/2))
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    # Calculate Haversine distance
    haversine_distance = r * c
    return haversine_distance


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return -distance_matrix


def edist(vectors):
    norm_v = torch.norm(vectors, dim=-1)[:, None]
    distance_matrix = torch.sqrt(vectors.mm(torch.t(vectors)) / (norm_v.mm(torch.t(norm_v))))
    return distance_matrix


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector():
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, dist_fn, pos_sample_bar=5, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        self.dist_fn = dist_fn
        self.pos_sample_bar = pos_sample_bar # samples within certain range are considered positive samples

    def get_triplets(self, embeddings, labels):
        # if self.cpu:
        #     embeddings = embeddings.cpu()
        # else:
        #     embeddings = embeddings.cuda()
        distance_matrix = self.dist_fn(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []
        for label in labels:
            # geo_labels = np.array(list(map(lambda tx: torch.norm(label-tx), labels)))
            geo_labels = np.linalg.norm(label - labels, axis=1)
            geo_mask = ((geo_labels <= self.pos_sample_bar) & (geo_labels > 0))
            label_indices = np.where(geo_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(geo_mask))[0]
            anchor_positives = np.array(list(combinations(label_indices, 2)))  # All anchor-positive pairs

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            loss_values = ap_distances.unsqueeze(dim=1) - distance_matrix[anchor_positives[:,0][:,None], negative_indices]
            loss_values = loss_values.data.cpu().numpy()
            for i, loss_val in enumerate(loss_values):
                hard_negative = self.negative_selection_fn(loss_val)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positives[i][0], anchor_positives[i][1], hard_negative])

        if len(triplets) <= 1:
            triplets.append([0, 0, 0])
            print('No triplet mined, pend triplets with 0.')

        triplets = np.array(triplets)
        return torch.LongTensor(triplets) if self.cpu else torch.cuda.LongTensor(triplets)

def HardestNegativeTripletSelector(margin, dist_fn, cpu=False): 
    return FunctionNegativeTripletSelector(margin=margin,
                                            negative_selection_fn=hardest_negative,
                                            dist_fn=dist_fn,
                                            cpu=cpu)


def RandomNegativeTripletSelector(margin, dist_fn, cpu=False): 
    return FunctionNegativeTripletSelector(margin=margin,
                                            negative_selection_fn=random_hard_negative,
                                            dist_fn=dist_fn,
                                            cpu=cpu)


def SemihardNegativeTripletSelector(margin, dist_fn, cpu=False): 
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=lambda x: semihard_negative(x, margin),
                                            dist_fn=dist_fn, 
                                            cpu=cpu)