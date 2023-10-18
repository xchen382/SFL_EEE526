#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    test_loss_local = 0
    correct = 0
    correct_local = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, data_target in enumerate(data_loader):
        if args.dataset == 'news':
            data = data_target['ids'].cuda().to(dtype = torch.long)
            mask = data_target['mask'].cuda().to(dtype = torch.long)
            target = data_target['targets'].cuda().to(dtype = torch.long)
            log_probs = net_g(data, mask)
        else:
            (data, target) = data_target
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net_g(data)
            log_probs_local = net_g.local.local_forward(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        test_loss_local += F.cross_entropy(log_probs_local, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        y_pred_local = log_probs_local.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        correct_local += y_pred_local.eq(target.data.view_as(y_pred_local)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    test_loss_local /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    accuracy_local = 100.00 * correct_local / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss, accuracy_local, test_loss_local

