#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from thop import profile,clever_format
import torchvision.models as models
from utils.pretrain import load_pretrained_resnet


from utils.dataset import getdata
from utils.options import args_parser
from models.Update_sfl import LocalUpdate
from models.resnet_sfl_imagenet import resnet18
from models.Fed import FedAvg
from models.test import test_img

# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="LargeScaleSFL",
# )

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    dataset_train,dataset_test,dict_users,image_size = getdata(args)

    # build model
    if args.model == 'resnet' and args.dataset == 'cifar':
        if args.finetune:
            classes = 100
        else:
            classes = 1000

        net_glob = resnet18(args.norm,num_classes=classes).to(args.device)
        local_glob,cloud_glob = net_glob.local, net_glob.cloud
    else:
        exit('Error: unrecognized model')
    cloud_optimizer = torch.optim.SGD(cloud_glob.parameters(), lr=args.server_lr, momentum=args.momentum)
    cloud_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cloud_optimizer, T_max=args.epochs)  # learning rate decay

    # input = torch.randn(1,3,224,224).cuda()
    # input = torch.randn(1,3,32,32).cuda()
    # macs, params = profile(net_glob, inputs=(input, ))
    # macs, params = clever_format([macs, params], "%.3f")
    # print(macs, params)

    if args.finetune:
        load_pretrained_resnet(net_glob)


    local_glob.train()
    cloud_glob.train()

    # copy weights
    w_local_glob = local_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    for iter in range(args.epochs):
        m = max(int(args.frac * args.num_users), 1)
        # randomly select m clients
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        local = LocalUpdate(args=args, dataset=dataset_train, idxs_users=idxs_users, dict_users=dict_users)
        w_locals, loss_avg = local.train(local_glob,iter,
                                            cloud=cloud_glob, 
                                            cloud_optim=cloud_optimizer, 
                                            cloud_sch=cloud_scheduler)

        # update global weights
        w_local_glob = FedAvg(w_locals)

        # copy weight to net_glob
        local_glob.load_state_dict(w_local_glob)

        # testing
        net_glob.eval()
        acc_test, _ = test_img(net_glob, dataset_test, args)
        # print loss
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
