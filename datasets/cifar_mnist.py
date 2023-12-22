"""
download the required dataset, split the data among the clients, and generate DataLoader for training
"""
import json
import os
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import h5py
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn

cudnn.banchmark = True
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from options import args_parser


class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        super(DatasetSplit, self).__init__()
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, target = self.dataset[self.idxs[item]]
        return image, target


def gen_ran_sum(_sum, num_users):
    base = 100 * np.ones(num_users, dtype=np.int32)
    _sum = _sum - 100 * num_users
    p = np.random.dirichlet(np.ones(num_users), size=1)
    print(p.sum())
    p = p[0]
    size_users = np.random.multinomial(_sum, p, size=1)[0]
    size_users = size_users + base
    print(size_users.sum())
    return size_users


def get_mean_and_std(dataset):
    """
    compute the mean and std value of dataset
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("=>compute mean and std")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def iid_esize_split(dataset, args, kwargs, is_shuffle=True):
    """
    根据指定的类别自定义每个客户端的数据。
    Args:
        dataset: 数据集。
        args: 包含num_clients, class_mapping等参数的对象。
        kwargs: DataLoader的额外参数。
        is_shuffle (bool): 是否打乱数据。
    Returns:
        list: 客户端数据加载器列表。
    """
    data_loaders = dict()

    if args.self_class == 0:  # iid随机分配类别
        num_samples_per_client = int(len(dataset) / args.num_clients)
        all_idxs = [i for i in range(len(dataset))]
        for i in range(args.num_clients):
            dict_users = np.random.choice(all_idxs, num_samples_per_client, replace=False)
            all_idxs = list(set(all_idxs) - set(dict_users))
            data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users), batch_size=args.batch_size, shuffle=is_shuffle, **kwargs)

    else:   # 自定义每客户类别开启
        class_mapping = json.loads(args.client_class_mapping)
        all_labels = [label for _, label in dataset]
        class_indices = {k: [] for k in set(all_labels)}
        # 为每个类别收集样本的索引
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)
        samples_per_client = int(len(dataset) / args.num_clients)
        for client_id_str, classes in class_mapping.items():
            client_id = int(client_id_str)  # 将客户端 ID 转换为从 0 开始的索引
            client_indices = []
            # 每个客户端分配的样本数应当相等，除以客户端的类别数以平均分配
            samples_per_class = samples_per_client // len(classes)
            for class_label in classes:
                # class_label = class_label_pre-1
                available_samples = class_indices[class_label]
                if len(available_samples) >= samples_per_class:
                    selected_samples = np.random.choice(available_samples, samples_per_class, replace=False)
                    # 从可用样本中移除已选择的样本
                    class_indices[class_label] = list(set(class_indices[class_label]) - set(selected_samples))
                else:  # 如果可用样本不足，允许重复选择
                    selected_samples = np.random.choice(available_samples, samples_per_class, replace=True)
                client_indices.extend(selected_samples)

            # 如果由于类别不均造成某客户端样本不足，从已分配的样本中随机选择补充
            while len(client_indices) < samples_per_client:
                extra_samples = samples_per_client - len(client_indices)
                extra_indices = np.random.choice(client_indices, extra_samples, replace=True)
                client_indices.extend(extra_indices)

            data_loaders[client_id] = DataLoader(DatasetSplit(dataset, client_indices),
                                                 batch_size=args.batch_size, shuffle=is_shuffle, **kwargs)
            # print(len(data_loaders[client_id].dataset))

    return data_loaders

# def iid_esize_split(dataset, args, kwargs, is_shuffle=True):
#     """
#     split the dataset to users
#     Return:
#         dict of the data_loaders
#     可自定义每个客户端的训练样本量
#     """
#     # 数据装载初始化
#     data_loaders = [0] * args.num_clients
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     # if num_samples_per_client == -1, then use all samples
#     if args.self_class == 0:  # iid随机分配类别
#         num_samples_per_client = int(len(dataset) / args.num_clients)
#         # change from dict to list
#         # print('start')
#         for i in range(args.num_clients):
#             dict_users[i] = np.random.choice(all_idxs, num_samples_per_client, replace=False)
#             all_idxs = list(set(all_idxs) - set(dict_users[i]))
#             data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
#                                          batch_size=args.batch_size,
#                                          shuffle=is_shuffle, **kwargs)
#     else:  # 自定义每客户类别开启
#         # 提取映射关系参数并将其解析为JSON对象
#         class_mapping = json.loads(args.class_mappingn)
#         for i in range(args.num_clients):  # 读取每个客户的类别信息
#             # 客户按id分配样本量
#             class_dict = class_mapping[str(i)]
#             if sample == -1: sample = int(len(dataset) / args.num_clients)
#             dict_users[i] = np.random.choice(all_idxs, sample, replace=False)
#             all_idxs = list(set(all_idxs) - set(dict_users[i]))
#             data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
#                                          batch_size=args.batch_size,
#                                          shuffle=is_shuffle, **kwargs)
#
#     return data_loaders

def iid_nesize_split(dataset, args, kwargs, is_shuffle=True):
    sum_samples = len(dataset)
    num_samples_per_client = gen_ran_sum(sum_samples, args.num_clients)
    # change from dict to list
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for (i, num_samples_client) in enumerate(num_samples_per_client):
        dict_users[i] = np.random.choice(all_idxs, num_samples_client, replace=False)
        # dict_users[i] = dict_users[i].astype(int)
        # dict_users[i] = set(dict_users[i])
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle, **kwargs)

    return data_loaders


def niid_esize_split(dataset, args, kwargs, is_shuffle=True):
    data_loaders = [0] * args.num_clients
    # each client has only two classes of the network
    num_shards = 2 * args.num_clients
    # the number of images in one shard
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    # is_shuffle is used to differentiate between train and test

    if args.dataset != "femnist":
        # original
        # editer: Ultraman6 20230928
        # torch>=1.4.0
        labels = dataset.targets
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        # sort the data according to their label
        idxs = idxs_labels[0, :]
        idxs = idxs.astype(int)
    else:
        # custom
        labels = np.array(dataset.targets)  # 将labels转换为NumPy数组
        idxs_labels = np.vstack((idxs[:len(labels)], labels[:len(idxs)]))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        idxs = idxs.astype(int)

    # divide and assign
    for i in range(args.num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle, **kwargs)
    return data_loaders


def niid_esize_split_train(dataset, args, kwargs, is_shuffle=True):
    data_loaders = [0] * args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    #     no need to judge train ans test here
    labels = dataset.train_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
    #     divide and assign
    #     and record the split patter
    split_pattern = {i: [] for i in range(args.num_clients)}
    for i in range(args.num_clients):
        rand_set = np.random.choice(idx_shard, 2, replace=False)
        split_pattern[i].append(rand_set)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, split_pattern


def niid_esize_split_test(dataset, args, kwargs, split_pattern, is_shuffle=False):
    data_loaders = [0] * args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    #     no need to judge train ans test here
    labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
    #     divide and assign
    for i in range(args.num_clients):
        rand_set = split_pattern[i][0]
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, None


def niid_esize_split_train_large(dataset, args, kwargs, is_shuffle=True):
    data_loaders = [0] * args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)

    split_pattern = {i: [] for i in range(args.num_clients)}
    for i in range(args.num_clients):
        rand_set = np.random.choice(idx_shard, 2, replace=False)
        # split_pattern[i].append(rand_set)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
            # store the label
            split_pattern[i].append(dataset.__getitem__(idxs[rand * num_imgs])[1])
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, split_pattern


def niid_esize_split_test_large(dataset, args, kwargs, split_pattern, is_shuffle=False):
    """
    :param dataset: test dataset
    :param args:
    :param kwargs:
    :param split_pattern: split pattern from trainloaders
    :param test_size: length of testloader of each client
    :param is_shuffle: False for testloader
    :return:
    """
    data_loaders = [0] * args.num_clients
    # for mnist and cifar 10, only 10 classes
    num_shards = 10
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(len(dataset))
    #     no need to judge train ans test here
    labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
    #     divide and assign
    for i in range(args.num_clients):
        rand_set = split_pattern[i]
        # idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, None


def niid_esize_split_oneclass(dataset, args, kwargs, is_shuffle=True):
    data_loaders = [0] * args.num_clients
    # one class perclients
    # any requirements on the number of clients?
    num_shards = args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)

    if args.dataset != "femnist":
        # original
        # editer: Ultraman6 20230928
        # torch>=1.4.0
        labels = dataset.targets
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        idxs = idxs.astype(int)
    else:
        # custom
        labels = np.array(dataset.targets)  # 将labels转换为NumPy数组
        idxs_labels = np.vstack((idxs[:len(labels)], labels[:len(idxs)]))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        idxs = idxs.astype(int)

    # divide and assign
    for i in range(args.num_clients):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle, **kwargs)
    return data_loaders


# 如何调整本地训练样本数量
def split_data(dataset, args, kwargs, is_shuffle=True):
    """
    return dataloaders
    """
    if args.iid == 1:
        data_loaders = iid_esize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == 0:
        data_loaders = niid_esize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == -1:
        data_loaders = iid_nesize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == -2:
        data_loaders = niid_esize_split_oneclass(dataset, args, kwargs, is_shuffle)
    else:
        raise ValueError('Data Distribution pattern `{}` not implemented '.format(args.iid))
    return data_loaders


def get_dataset(dataset_root, dataset, args):
    # trains, train_loaders, tests, test_loaders = {}, {}, {}, {}
    if dataset == 'mnist':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_mnist(dataset_root, args)
    elif dataset == 'cifar10':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_cifar10(dataset_root, args)
    elif dataset == 'femnist':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_femnist(dataset_root, args)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))
    return train_loaders, test_loaders, v_train_loader, v_test_loader


def get_mnist(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=True,
                           download=True, transform=transform)
    test = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=False,
                          download=True, transform=transform)
    # note: is_shuffle here also is a flag for differentiating train and test
    train_loaders = split_data(train, args, kwargs, is_shuffle=True)

    test_loaders = []
    if args.test_on_all_samples == 1:
        # 将整个测试集分配给每个客户端
        for i in range(args.num_clients):
            test_loader = torch.utils.data.DataLoader(
                test, batch_size=args.batch_size, shuffle=False, **kwargs
            )
            test_loaders.append(test_loader)
    else:
        test_loaders = split_data(test, args, kwargs, is_shuffle=False)

    # the actual batch_size may need to change.... Depend on the actual gradient...
    # originally written to get the gradient of the whole dataset
    # but now it seems to be able to improve speed of getting accuracy of virtual sequence
    v_train_loader = DataLoader(train, batch_size=args.batch_size * args.num_clients,
                                shuffle=True, **kwargs)
    v_test_loader = DataLoader(test, batch_size=args.batch_size * args.num_clients,
                               shuffle=False, **kwargs)

    return train_loaders, test_loaders, v_train_loader, v_test_loader


def get_cifar10(dataset_root, args):  # cifa10数据集下只能使用cnn_complex和resnet18模型
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
    if args.model == 'cnn_complex':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.model == 'resnet18':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise ValueError("this nn for cifar10 not implemented")
    train = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=True,
                             download=True, transform=transform_train)
    test = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=False,
                            download=True, transform=transform_test)
    v_train_loader = DataLoader(train, batch_size=args.batch_size,
                                shuffle=True, **kwargs)
    v_test_loader = DataLoader(test, batch_size=args.batch_size,
                               shuffle=False, **kwargs)
    train_loaders = split_data(train, args, kwargs)

    test_loaders = []
    if args.test_on_all_samples == 1:
        # 将整个测试集分配给每个客户端
        for i in range(args.num_clients):
            test_loader = torch.utils.data.DataLoader(
                test, batch_size=args.batch_size, shuffle=False, **kwargs
            )
            test_loaders.append(test_loader)
    else:
        test_loaders = split_data(test, args, kwargs)

    return train_loaders, test_loaders, v_train_loader, v_test_loader


def get_femnist(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}

    train_h5 = h5py.File(os.path.join(dataset_root, 'femnist/fed_emnist_train.h5'), "r")
    test_h5 = h5py.File(os.path.join(dataset_root, 'femnist/fed_emnist_test.h5'), "r")
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    _EXAMPLE = "examples"
    _IMGAE = "pixels"
    _LABEL = "label"

    client_ids_train = list(train_h5[_EXAMPLE].keys())
    client_ids_test = list(test_h5[_EXAMPLE].keys())
    train_ids = client_ids_train
    test_ids = client_ids_test

    for client_id in train_ids:
        train_x.append(train_h5[_EXAMPLE][client_id][_IMGAE][()])
        train_y.append(train_h5[_EXAMPLE][client_id][_LABEL][()].squeeze())
    train_x = np.vstack(train_x)
    train_y = np.hstack(train_y)

    for client_id in test_ids:
        test_x.append(test_h5[_EXAMPLE][client_id][_IMGAE][()])
        test_y.append(test_h5[_EXAMPLE][client_id][_LABEL][()].squeeze())
    test_x = np.vstack(test_x)
    test_y = np.hstack(test_y)

    train_x = train_x.reshape(-1, 1, 28, 28)
    test_x = test_x.reshape(-1, 1, 28, 28)

    # train_ds = data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
    # train_ds.targets = train_y  # 添加targets属性
    # # train_loader = data.DataLoader(
    # #     dataset=train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs
    # # )
    # test_ds = data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long))
    # test_ds.targets = test_y  # 添加targets属性
    # # test_loader = data.DataLoader(
    # #     dataset=test_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, **kwargs
    # # )

    train_ds = data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
    train_ds.targets = train_y  # 添加targets属性
    test_ds = data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long))
    test_ds.targets = test_y  # 添加targets属性

    v_train_loader = DataLoader(train_ds, batch_size=args.batch_size * args.num_clients,
                                shuffle=True, **kwargs)
    v_test_loader = DataLoader(test_ds, batch_size=args.batch_size * args.num_clients,
                               shuffle=False, **kwargs)

    train_loaders = split_data(train_ds, args, kwargs, is_shuffle=True)
    test_loaders = split_data(test_ds, args, kwargs, is_shuffle=False)

    train_h5.close()
    test_h5.close()

    return train_loaders, test_loaders, v_train_loader, v_test_loader


def show_distribution(dataloader, args):
    """
    show the distribution of the data on certain client with dataloader
    return:
        percentage of each class of the label
    """
    if args.dataset == 'femnist':
        labels = dataloader.dataset.dataset.targets
    elif args.dataset == 'mnist':
        try:
            labels = dataloader.dataset.dataset.train_labels.numpy()
        except:
            print(f"Using test_labels")
            labels = dataloader.dataset.dataset.test_labels.numpy()
        # labels = dataloader.dataset.dataset.train_labels.numpy()
    elif args.dataset == 'cifar10':
        try:
            labels = dataloader.dataset.dataset.targets
        except:
            print(f"Using test_labels")
            labels = dataloader.dataset.dataset.targets
    elif args.dataset == 'fsdd':
        labels = dataloader.dataset.labels
    else:
        raise ValueError("`{}` dataset not included".format(args.dataset))
    num_samples = len(dataloader.dataset)
    # print(num_samples)
    idxs = [i for i in range(num_samples)]
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    distribution = [0] * len(unique_labels)
    for idx in idxs:
        img, label = dataloader.dataset[idx]
        distribution[label] += 1
    distribution = np.array(distribution)
    distribution = distribution / num_samples
    return distribution


if __name__ == '__main__':
    args = args_parser()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loaders, test_loaders, _, _ = get_dataset(args.dataset_root, args.dataset, args)
    print(f"The dataset is {args.dataset} divided into {args.num_clients} clients/tasks in an iid = {args.iid} way")
    for i in range(args.num_clients):
        train_loader = train_loaders[i]
        print(len(train_loader.dataset))
        distribution = show_distribution(train_loader, args)
        print("dataloader {} distribution".format(i))
        print(distribution)
