#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
from models.Fed import FedAvg
from models.test import test_img

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs,datasetname):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.datasetname = datasetname

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs_users=None, dict_users=None):
        self.args = args
        # client_freeze
        self.epochs = args.epochs

        self.dict_users = dict_users
        self.idxs_users = idxs_users
        self.dataset = dataset
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train_list_small = []
        self.ldr_train_list_large = []
        self.dict_users = dict_users
        for idx in self.idxs_users:
            idxs = self.dict_users[idx]
            self.ldr_train_list_small.append(iter(DataLoader(DatasetSplit(self.dataset, idxs, self.args.dataset), batch_size=self.args.local_bs//len(self.idxs_users), shuffle=True)))
            self.ldr_train_list_large.append(DataLoader(DatasetSplit(self.dataset, idxs, self.args.dataset), batch_size=self.args.local_bs, shuffle=True))

    def reinitial_dataset(self):
        self.ldr_train_list_small = []
        for idx in self.idxs_users:
            idxs = self.dict_users[idx]
            self.ldr_train_list_small.append(iter(DataLoader(DatasetSplit(self.dataset, idxs, self.args.dataset), batch_size=self.args.local_bs//len(self.idxs_users), shuffle=True)))

    def global_train_client(self, local_glob, epoch, cloud, cloud_optim, cloud_sch):
        cloud.train()
        # train and update
        cloud_sch.step(epoch)

        w_locals = []

        local_nets = [copy.deepcopy(local_glob).to(self.args.device) for _ in range(len(self.ldr_train_list_small))]
        parameters = list(local_nets[0].parameters())
        for local_net in local_nets[1:]:
            parameters += list(local_net.parameters())
        optimizer = torch.optim.SGD(parameters, lr=self.args.lr, momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)  # learning rate decay
        # scheduler.step(epoch)
        del parameters

        batch_loss = []
        for batch_idx in range(len(self.ldr_train_list_small[0])):
            cloud_optim.zero_grad()
            acts_locals = []
            label_locals = []

            for i,ldr_train,local_net in zip(range(len(self.ldr_train_list_small)),self.ldr_train_list_small,local_nets):
                local_net.zero_grad()
                local_net.train()
                try:
                    images, labels = next(ldr_train)
                except:
                    self.ldr_train_list_small[i] = iter(DataLoader(DatasetSplit(self.dataset, self.dict_users[i], self.args.dataset), batch_size=self.args.local_bs//len(self.idxs_users), shuffle=True))
                    images, labels = next(self.ldr_train_list_small[i])
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                acts_local = local_net(images)

                acts_locals.append(acts_local)
                label_locals.append(labels)
   

            acts_locals = torch.cat(acts_locals, dim = 0).to(self.args.device)
            label_locals = torch.cat(label_locals, dim = 0).to(self.args.device)

            log_probs = cloud(acts_locals)
            
            loss = self.loss_func(log_probs, label_locals)
            loss.backward()
            optimizer.step()
            cloud_optim.step()

            if self.args.verbose and batch_idx % 10 == 0:
                print('Update Batch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx, batch_idx * len(labels), len(self.dict_users[0]),
                        100. * batch_idx* len(labels) / len(self.dict_users[0]), loss.item()))
            batch_loss.append(loss.item())
        for local_net in local_nets:
            w_locals.append(copy.deepcopy(local_net.state_dict()))

        return w_locals, sum(batch_loss) / len(batch_loss)


    def train(self, local_glob, epoch, cloud, cloud_optim, cloud_sch, dataset_test=None,net_glob=None,logger=None,last_acc=0):
        return self.global_train_client(local_glob,epoch,cloud,cloud_optim,cloud_sch)

