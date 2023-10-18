#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, datasetname):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.datasetname = datasetname

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.datasetname == 'news':
            return self.dataset[self.idxs[item]]
        else:
            image, label = self.dataset[self.idxs[item]]
            return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, self.args.dataset), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, epoch):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)  # learning rate decay
        scheduler.step(epoch)


        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, data in enumerate(self.ldr_train):
                net.zero_grad()

                if self.args.dataset == 'news':
                    ids = data['ids'].to(self.args.device, dtype = torch.long)
                    mask = data['mask'].to(self.args.device, dtype = torch.long)
                    labels = data['targets'].to(self.args.device, dtype = torch.long)
                    log_probs = net(ids, mask)
                else:
                    images, labels = data[0],data[1]
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(labels), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

