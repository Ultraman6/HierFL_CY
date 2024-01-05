import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
import random


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex

def generate_synthetic(alpha, beta, iid, dimension, NUM_CLASS, NUM_USER):
    samples_per_user = np.random.lognormal(4, 2, (NUM_USER)).astype(int) + 50
    print(samples_per_user)

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]

    #### Define some prior ####
    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    B = np.random.normal(0, beta, NUM_USER)
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    W_global = b_global = None
    if iid == 1:
        W_global = np.random.normal(0, 1, (dimension, NUM_CLASS))
        b_global = np.random.normal(0, 1, NUM_CLASS)

    for i in range(NUM_USER):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]
        else:
            mean_x[i] = np.random.normal(B[i], 1, dimension)

        W = W_global if iid == 1 else np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = b_global if iid == 1 else np.random.normal(mean_b[i], 1, NUM_CLASS)

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))
        X_split[i] = xx.tolist()
        y_split[i] = [int(label) for label in yy]  # 转换为整型

    return X_split, y_split


def get_dataloader(X, y, args):
    train_loaders = []
    test_loaders = []

    global_train_data = []
    global_test_data = []
    if args.self_class == 1:  # 开启自定义类别映射
        # 计算每个客户端理想的数据量
        ideal_samples_per_client = max([len(client_data) for client_data in X])  # 可以选择最大值或其他策略

        # 融合所有客户的数据集和标签
        all_data = [item for sublist in X for item in sublist]
        all_labels = [item for sublist in y for item in sublist]

        # 读入类别映射配置
        class_mapping = json.loads(args.client_class_mapping)
        print(set(all_labels))
        # print(class_mapping)
        # 为每个客户重新构造数据集和标签
        for client_id in range(len(X)):
            client_data = []
            client_labels = []
            # 根据映射收集客户端的数据
            for data, label in zip(all_data, all_labels):
                if label in class_mapping[str(client_id)]:
                    client_data.append(data)
                    client_labels.append(label)

            # 数据增强：随机重复数据点以达到理想的数据量
            while len(client_data) < ideal_samples_per_client:
                index = random.randint(0, len(client_data) - 1)
                client_data.append(client_data[index])
                client_labels.append(client_labels[index])

            # 随机打乱客户端数据
            combined = list(zip(client_data, client_labels))
            random.shuffle(combined)
            client_data[:], client_labels[:] = zip(*combined)

            # 划分训练和测试数据集
            train_len = int(0.9 * len(client_data))
            X_train, X_test = client_data[:train_len], client_data[train_len:]
            y_train, y_test = client_labels[:train_len], client_labels[train_len:]

            # 创建训练和测试 DataLoader
            train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
            test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64))

            train_loaders.append(DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True))
            test_loaders.append(DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False))

            # 聚合全局训练和测试数据
            global_train_data.extend(zip(X_train, y_train))
            global_test_data.extend(zip(X_test, y_test))

        # 创建全局验证 DataLoader
        test_set_size = len(global_test_data)
        subset_size = int(test_set_size * args.test_ratio)
        subset_indices = random.sample(range(test_set_size), subset_size)
        v_test_ds = TensorDataset(torch.tensor([x for x, _ in global_test_data], dtype=torch.float32),
                                  torch.tensor([y for _, y in global_test_data], dtype=torch.int64))
        v_test_subset = Subset(v_test_ds, subset_indices)
        v_test_loader = DataLoader(v_test_subset, batch_size=args.batch_size, shuffle=False)

    else:
        for i in range(len(X)):
            # Split data into training and testing sets for each user
            train_len = int(0.9 * len(X[i]))
            X_train, X_test = X[i][:train_len], X[i][train_len:]
            y_train, y_test = y[i][:train_len], y[i][train_len:]

            # Create DataLoaders for local data
            train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.int64))
            test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.int64))

            train_loaders.append(DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True))
            test_loaders.append(DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False))

            # Aggregate data for shared and global validation sets
            global_train_data.extend(zip(X_train, y_train))
            global_test_data.extend(zip(X_test, y_test))

        # Create global validation DataLoader
        test_set_size = len(global_test_data)
        subset_size = int(test_set_size * args.test_ratio)  # Example: retain 20% of the data for validation
        subset_indices = random.sample(range(test_set_size), subset_size)
        v_test_ds = TensorDataset(torch.tensor([x for x, _ in global_test_data]),
                                  torch.tensor([y for _, y in global_test_data], dtype=torch.int64))
        v_test_subset = Subset(v_test_ds, subset_indices)
        v_test_loader = DataLoader(v_test_subset, batch_size=args.batch_size, shuffle=False)

    return train_loaders, test_loaders, v_test_loader

def get_synthetic(args):
    X, y = generate_synthetic(args.alpha, args.beta, args.iid, args.dimension, args.num_class, args.num_clients)
    train_loaders, test_loaders, v_test_loader = get_dataloader(X, y, args)
    return train_loaders, test_loaders, v_test_loader