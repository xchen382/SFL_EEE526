#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
# import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.dataset import getdata
from utils.options import args_parser
from models.Update_sfl import LocalUpdate
from models.resnet_sfl import resnet18
from models.vgg_sfl import vgg11_bn
from models.Fed import FedAvg
from models.test import test_img
from utils.setup_logger import setup_logger

# # # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="LargeScaleSFL_timelySFL",
# )

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    dataset_train,dataset_test,dict_users,user_catogory = getdata(args)

    # build model
    if args.model == 'resnet':
        if args.dataset == 'cifar100':
            net_glob = resnet18(args.norm,args.cut,num_classes=100).to(args.device)
        elif args.dataset == 'cifar10':
            net_glob = resnet18(args.norm,args.cut,num_classes=10).to(args.device)

        local_glob,cloud_glob = net_glob.local, net_glob.cloud
        input = torch.randn(1,3,32,32).cuda()
    elif args.model == 'vgg11_bn':
        if args.dataset == 'cifar100':
            net_glob = vgg11_bn(args.norm,args.cut,num_classes=100).to(args.device)
        elif args.dataset == 'cifar10':
            net_glob = vgg11_bn(args.norm,args.cut,num_classes=10).to(args.device)
    else:
        exit('Error: unrecognized model')
    
    local_glob,cloud_glob = net_glob.local, net_glob.cloud
    cloud_optimizer = torch.optim.SGD(cloud_glob.parameters(), lr=args.server_lr, momentum=args.momentum)
    cloud_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cloud_optimizer, T_max=args.epochs)  # learning rate decay
    
    ## setup save folder
    save_dir = "./saves/"+str(args.save_dir) + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # setup logger
    model_log_file = save_dir + '/MIA.log'
    logger = setup_logger('{}_logger'.format(str(save_dir)),model_log_file)

    logger.info('local_glob {}'.format(local_glob))
    logger.info('cloud_glob {}'.format(cloud_glob))

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

    # testing
    net_glob.eval()
    acc_test, _, _, _ = test_img(net_glob, dataset_test, args)
    activation_buffer = []

    for iter in range(args.epochs):
        m = max(int(args.frac * args.num_users), 1)
        # randomly select m clients
        idxs_users = []
        if args.sampler=='normal':
            raise ValueError('normal case WIP')
            while len(idxs_users)<m:
                idxs_users = list(idxs_users)
                idxs_users_tmp = np.round(80*np.random.randn(m-len(idxs_users))+250).astype(int)
                idxs_users_tmp[idxs_users_tmp<=0] = 7
                idxs_users_tmp[idxs_users_tmp>=args.num_users] = 135
                idxs_users += list(idxs_users_tmp)
                idxs_users = set(idxs_users)

        elif args.sampler=='iid':
            min_std = 10000
            for _ in range(100):
                idxs_users_tmp = np.random.choice(range(args.num_users), m, replace=False)
                sum_tmp = np.array(user_catogory[idxs_users_tmp[0]])
                for client_item in idxs_users_tmp[1:]:
                    sum_tmp += np.array(user_catogory[client_item])
                
                std_tmp = np.std(sum_tmp)/np.mean(sum_tmp)

                if std_tmp<min_std:
                    idxs_users = idxs_users_tmp
                    min_std = min(std_tmp,min_std)
        else:
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        local = LocalUpdate(args=args, dataset=dataset_train, idxs_users=idxs_users, dict_users=dict_users)
        
        # for name,paras in cloud_glob.named_parameters():
        #     print('cloud before updating',name,paras[0,0,:,:])
        #     break

        # for name,paras in local_glob.named_parameters():
        #     if name == 'head.0.weight' or name == 'local_classifier.head.0.weight':
        #         print('client before updating',name,paras[0,0,:,:])

        w_locals, loss_avg = local.train(local_glob,iter,
                                                cloud=cloud_glob, 
                                                cloud_optim=cloud_optimizer, 
                                                cloud_sch=cloud_scheduler,
                                                dataset_test=dataset_test,
                                                net_glob = net_glob,
                                                logger = logger,
                                                last_acc = acc_test)


        # update global weights
        w_local_glob = FedAvg(w_locals)

        # copy weight to net_glob
        local_glob.load_state_dict(w_local_glob)
        del w_local_glob,w_locals,local

        # testing
        net_glob.eval()
        acc_test, _, acc_test_local, _ = test_img(net_glob, dataset_test, args)
        # print loss
        logger.info('Round {:3d}, Average loss {:.3f}, Test acc {:.3f}, Local acc {:.3f}'.format(iter, loss_avg, acc_test, acc_test_local))
        # # log metrics to wandb
        # wandb.log({"Sync Round": iter, "accuracy":acc_test, 'loss':loss_avg
        #            })
        loss_train.append(loss_avg)
        cv_acc.append(acc_test)


    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}_sampler{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid,args.sampler))

    # testing
    net_glob.eval()
    acc_train,loss_train,_,_= test_img(net_glob, dataset_train, args)
    acc_test,loss_test,_,_ = test_img(net_glob, dataset_test, args)
    logger.info("Training accuracy: {:.2f}".format(acc_train))
    logger.info("Testing accuracy: {:.2f}".format(acc_test))

    # wandb.finish()
