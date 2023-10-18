#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
from thop import profile,clever_format


from utils.dataset import getdata
from utils.options import args_parser
from models.Update import LocalUpdate
from models.resnet import resnet18
from models.Fed import FedAvg
from models.test import test_img

# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="LargeScaleSFL",
# )

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    dataset_train,dataset_test,dict_users,user_catogory = getdata(args)

    # build model
    if args.model == 'resnet' and args.dataset == 'cifar100':
        net_glob = resnet18().to(args.device)
        # input = torch.randn(1,3,32,32).cuda()
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # macs, params = profile(net_glob, inputs=(input, ))
    # macs, params = clever_format([macs, params], "%.3f")
    # print(macs, params)
    # exit()
    
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    for iter in range(args.epochs):
        loss_locals = []
        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        # randomly select m clients
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(copy.deepcopy(net_glob).to(args.device),iter)
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)

        # testing
        net_glob.eval()
        acc_test, _ = test_img(net_glob, dataset_test, args)



        print('Round {:3d}, Average loss {:.3f}, Test acc {:.3f}'.format(iter, loss_avg, acc_test))
        # log metrics to wandb
        wandb.log({"Sync Round": iter, "accuracy":acc_test, 'loss':loss_avg
                   })

        loss_train.append(loss_avg)
        cv_acc.append(acc_test)

        


    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    wandb.finish()
