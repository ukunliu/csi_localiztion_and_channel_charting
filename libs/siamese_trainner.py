from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import subprocess as sp


def pair_training(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, log_interval,start_epoch=0, few_shot=0):
    train_ls = []
    test_ls = []
    writer = SummaryWriter()

    # for epoch in range(0, start_epoch):
    #     scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, log_interval, few_shot)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        # for metric in metrics:
        #     message += '\t{}: {}'.format(metric.name(), metric.value())

        ## Test stage
        val_loss = test_epoch(val_loader, model, loss_fn, few_shot)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)
        train_ls.append(train_loss)
        test_ls.append(val_loss)

        writer.flush()

    writer.close()
    return train_ls, test_ls


def train_epoch(train_loader, model, loss_fn, optimizer,  log_interval, few_shot, metrics=[]):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        # if cuda:
        #     data = tuple(d.cuda() for d in data)
        #     if target is not None:
        #         target = target.cuda()

        # if cuda: 
        #     print(f'Data device is {data[0].get_device()}. Cuda status is {cuda}')


        optimizer.zero_grad()
        data_reshape = np.swapaxes(data[0], 0, 1)
        outputs = model(*data_reshape)

        # if type(outputs) not in (tuple, list):
        #     outputs = (outputs,)

        # loss_inputs = (outputs, )
        # if target is not None:
        #     # target = (target,)
        #     loss_inputs += (target, )
        # print(len(loss_inputs))
        if few_shot:
            loss_outputs = loss_fn(*data_reshape, *outputs, target)
        else:
            loss_outputs = loss_fn(*data_reshape, *outputs) # Triplet selection, return mined triplets
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


def test_epoch(val_loader, model, loss_fn, few_shot):
    with torch.no_grad():
        # for metric in metrics:
        #     metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            # if cuda:
            #     data = tuple(d.cuda() for d in data)
            #     if target is not None:
            #         target = target.cuda()
            data_reshape = np.swapaxes(data[0], 0, 1)
            outputs = model(*data_reshape)

            # if type(outputs) not in (tuple, list):
            #     outputs = (outputs,)
            # loss_inputs = outputs
            # if target is not None:
            #     target = (target,)
            #     loss_inputs += target
            if few_shot:
                        loss_outputs = loss_fn(*data_reshape, *outputs, target)
            else:
                loss_outputs = loss_fn(*data_reshape, *outputs) # Triplet selection, return mined triplets
       
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            # for metric in metrics:
            #     metric(outputs, target, loss_outputs)

    return val_loss