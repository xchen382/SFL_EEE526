from .sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid,news_iid
from torchvision import datasets, transforms


CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

def getdata(args):
    user_catogory = {}
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar10':
        CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_TRAIN_STD = (0.247, 0.243, 0.261)
        
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
    ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
        ])

        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users,user_catogory = cifar_noniid(dataset_train, args.num_users,num_classes=10)
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])

        dataset_train = datasets.CIFAR100('../data/cifar', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR100('../data/cifar', train=False, download=True, transform=transform_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users,user_catogory = cifar_noniid(dataset_train, args.num_users,num_classes=100)
    elif args.dataset == 'news':
        path='./data/nlp/newsCorpora.csv'
        df=process_nlp_data(path)
        dataset_train, dataset_test = getdataset(df)
        dict_users = news_iid(dataset_train,args.num_users)
    else:
        exit('Error: unrecognized dataset')
    
    return dataset_train,dataset_test,dict_users,user_catogory