from matplotlib.style import available
import torch.nn as nn
import torch.nn.functional as F
import torch
#import tensorflow as tf
#import math

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative, target, size_average = True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        
        if size_average:
            losses = losses.mean()
        else:
            losses = losses.sum()
        
        return losses, len(anchor)
    

class PairLoss(nn.Module):
    def __init__(self):
        super(PairLoss, self).__init__()
    
    def forward(self, x, y, f_x, f_y, size_average = True):
        distance_feature = (x - y).pow(2).sum(1).sqrt()
        distance_embedding = (f_x - f_y).pow(2).sum(1).sqrt()
        mask = (distance_feature > 0) & (distance_feature < .8)
        losses = (distance_embedding[mask] - distance_feature[mask]).pow(2) / distance_feature[mask]      
        if size_average:
            losses = losses.mean()
        else:
            losses = losses.sum()
        
        return losses
    

class TripletFewShotLoss(nn.Module):
    def __init__(self, margin, data_idx, locations, alpha=1):
        super(TripletFewShotLoss, self).__init__()
        self.margin = margin
        self.data_idx = data_idx
        self.positions = torch.tensor(locations)
        self.alpha = alpha
    
    def forward(self, anchor, positive, negative, target, size_average = True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)

        anchor_mask = []
        pos_mask = []
        neg_mask = []
        target = target.detach().numpy()
        for i, j, k in target:
            if i in self.data_idx:
                anchor_mask.append(1)
            else:
                anchor_mask.append(0)

            if j in self.data_idx:
                pos_mask.append(1)
            else:
                pos_mask.append(0)

            if k in self.data_idx:
                neg_mask.append(1)
            else:
                neg_mask.append(0)

        available_data = sum(anchor_mask) + sum(pos_mask) + sum(neg_mask) 
        if available_data == 0:
            penalty = 0
        else:
            penalty_anchor = (anchor[anchor_mask] - self.positions[anchor_mask]).pow(2).sum(1).sqrt()
            penalty_pos = (positive[pos_mask] - self.positions[pos_mask]).pow(2).sum(1).sqrt()
            penalty_neg = (negative[neg_mask] - self.positions[neg_mask]).pow(2).sum(1).sqrt()

            penalty = (penalty_anchor + penalty_pos + penalty_neg) / available_data
        
        if size_average:
            losses_triplet = F.relu(distance_positive - distance_negative + self.margin)

            losses = losses_triplet.mean() + self.alpha * penalty.mean()
            # losses = losses.mean()
        else:
            losses = losses.sum()


        
        return losses, len(anchor)
    

class PairFewShotLoss(nn.Module):
    def __init__(self, data_idx, locations, alpha=1):
        super(PairFewShotLoss, self).__init__()
        self.data_idx = data_idx
        self.positions = torch.tensor(locations)
        self.alpha = alpha
    
    def forward(self, x, y, f_x, f_y, target, size_average = True):
        distance_feature = (x - y).pow(2).sum(1).sqrt()
        distance_embedding = (f_x - f_y).pow(2).sum(1).sqrt()
        mask = (distance_feature > 0) & (distance_feature < .8)

        # x_mask = [True if i in self.data_idx else False for i in target[:, 0]]
        # y_mask = [True if i in self.data_idx else False for i in target[:, 1]]
        x_mask = []
        y_mask = []
        target = target.detach().numpy()
        for i, j in target:
            if i in self.data_idx:
                x_mask.append(1)
            else:
                x_mask.append(0)

            if j in self.data_idx:
                y_mask.append(1)
            else:
                y_mask.append(0)

        available_data = sum(x_mask) + sum(y_mask) 
        if available_data == 0:
            penalty = 0
        else:
            penalty_x = (f_x[x_mask] - self.positions[x_mask]).pow(2).sum(1).sqrt()
            penalty_y = (f_y[x_mask] - self.positions[y_mask]).pow(2).sum(1).sqrt()

            penalty = (penalty_x + penalty_y) / available_data

        if size_average:
            losses = ((distance_embedding[mask] - distance_feature[mask]).pow(2) / distance_feature[mask]).mean() + self.alpha * penalty.mean() 

            # losses = losses.mean()
        else:
            losses = losses.sum()
        
        return losses

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()
        
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
    

class TripletMetricLoss(nn.Module):
    def __init__(self, margin):
        super(TripletMetricLoss, self).__init__()
        self.margin = margin
    
    def forward(self, d_xy, d_xz, size_average=True):

        losses = F.relu(d_xy - d_xz + self.margin)
        
        # if size_average:
        #     losses = losses.mean()
        # else:
        #     losses = losses.sum()
        
        return losses.mean()