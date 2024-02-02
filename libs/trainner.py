from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from geopy.distance import geodesic
from torch.utils.tensorboard import SummaryWriter
import subprocess as sp


class Measurement(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, features, labels):
        """
        """
        self.features = features#.values
        self.labels = labels#.values


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature, label = self.features[idx].astype(np.float32), self.labels[idx]
    #     geo_labels = np.array(list(map(lambda tx: geodesic(label, tx).m, self.labels)))
    #     # is it necessary to fabricate the perfect triplet at beginning?
    #     positive = self.features[idx-1].astype(np.float32)
    #     negative = self.features[geo_labels.argmax()].astype(np.float32)

    #     return (torch.from_numpy(feature), torch.from_numpy(positive), torch.from_numpy(negative)), label
        return torch.from_numpy(feature), torch.from_numpy(label)
    

class OfflineTriplet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, features, labels):
        """
        """
        self.features = features.values
        self.labels = labels.values


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature, label = self.features[idx].astype(np.float32), self.labels[idx]
        geo_dists = np.linalg.norm(label-self.labels, axis=0)
        geo_masks = geo_dists < 5
        # geo_labels = np.array(list(map(lambda tx: geodesic(label, tx).m, self.labels)))
        # is it necessary to fabricate the perfect triplet at beginning?
        label_indices = np.where(geo_masks)[0]
        
        if len(label_indices) < 2:
            positive = self.features[idx - 1].astype(np.float32)
            negative = self.features[np.random.choice(len(self.features))]
        else:
            positive = self.features[np.random.choice(np.where(geo_masks)[0])].astype(np.float32)
            negative = self.features[np.random.choice(np.where(~geo_masks)[0])].astype(np.float32)

        return (torch.from_numpy(feature), torch.from_numpy(positive), torch.from_numpy(negative)), label
        # return torch.from_numpy(feature), torch.from_numpy(label)


def training(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],start_epoch=0):
    train_ls = []
    test_ls = []
    writer = SummaryWriter()

    # for epoch in range(0, start_epoch):
    #     scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        # for metric in metrics:
        #     message += '\t{}: {}'.format(metric.name(), metric.value())

        ## Test stage
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)
        train_ls.append(train_loss)
        test_ls.append(val_loss)

        if cuda:
            sp.run('nvidia-smi')

        writer.flush()

    writer.close()
    return train_ls, test_ls


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics=[]):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        # if cuda: 
        #     print(f'Data device is {data[0].get_device()}. Cuda status is {cuda}')


        optimizer.zero_grad()
        outputs = model(*np.swapaxes(data[0], 0, 1))

        # if type(outputs) not in (tuple, list):
        #     outputs = (outputs,)

        # loss_inputs = (outputs, )
        # if target is not None:
        #     # target = (target,)
        #     loss_inputs += (target, )
        # print(len(loss_inputs))

        loss_outputs = loss_fn(*outputs, target) # Triplet selection, return mined triplets
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics=[]):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*np.swapaxes(data[0], 0, 1))

            # if type(outputs) not in (tuple, list):
            #     outputs = (outputs,)
            # loss_inputs = outputs
            # if target is not None:
            #     target = (target,)
            #     loss_inputs += target

            loss_outputs = loss_fn(*outputs, target)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics