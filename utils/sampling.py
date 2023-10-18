#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_noniid(dataset, num_users,num_classes,beta=0.1):
    """
    Sample non-I.I.D. client data from CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    beta=beta
    labels = np.array(dataset.targets)
    num_data = len(dataset)
    _lst_sample = 0
    K = num_classes
    min_size = 0
    y_train = labels


    least_idx = np.zeros((num_users, num_classes, _lst_sample), dtype=np.int)

    for i in range(num_classes):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
    least_idx = np.reshape(least_idx, (num_users, -1))


    least_idx_set = set(np.reshape(least_idx, (-1)))
    local_idx = np.random.choice(list(set(range(50000))-least_idx_set), len(list(set(range(50000))-least_idx_set)), replace=False)

    N = y_train.shape[0]
    net_dataidx_map = {}
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    while min_size < 10:
        idx_batch = [[] for _ in range(num_users)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            idx_k = [id for id in idx_k if id in local_idx]
            
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]  
        dict_users[j] = np.concatenate((dict_users[j], least_idx[j]), axis=0)          

    user_catogory = {}
    for i in range(num_users):

        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(num_classes)] )
        user_catogory[i] = cnts
        print('[client-{}] labels:{} sum:{} classes:{}'.format(i," ".join([str(cnt) for cnt in cnts]),sum(cnts), sum(cnts!=0)))


    return dict_users, user_catogory

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def news_iid(dataset,num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)


