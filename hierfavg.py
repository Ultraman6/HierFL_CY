# Flow of the algorithm
# Client update(t_1) -> Edge Aggregate(t_2) -> Cloud Aggregate(t_3)
import json
import random
import time
from threading import Thread

from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Subset, DataLoader, dataset

from average import average_weights_simple, average_weights
from models.synthetic_logistic import LogisticRegression_SYNTHETIC
from options import args_parser
from tensorboardX import SummaryWriter
import torch
from client import Client
from edge import Edge
from cloud import Cloud
from datasets.get_data import get_dataloaders, show_distribution
import copy
import numpy as np
from tqdm import tqdm
from models.mnist_cnn import mnist_lenet, mnist_cnn
from models.cifar_cnn_3conv_layer import cifar_cnn_3conv
from models.cifar_resnet import ResNet18, ResNet18_YWX
from models.mnist_logistic import LogisticRegression_MNIST
import os


def get_client_class(args, clients):
    client_class = []
    client_class_dis = [[], [], [], [], [], [], [], [], [], []]
    for client in clients:
        train_loader = client.train_loader
        distribution = show_distribution(train_loader, args)
        label = np.argmax(distribution)
        client_class.append(label)
        client_class_dis[label].append(client.id)
    print(client_class_dis)
    return client_class_dis


def get_edge_class(args, edges, clients):
    edge_class = [[], [], [], [], []]
    for (i, edge) in enumerate(edges):
        for cid in edge.cids:
            client = clients[cid]
            train_loader = client.train_loader
            distribution = show_distribution(train_loader, args)
            label = np.argmax(distribution)
            edge_class[i].append(label)
    print(f'class distribution among edge {edge_class}')


def initialize_edges_iid(num_edges, clients, args, client_class_dis):
    """
    This function is specially designed for partiion for 10*L users, 1-class per user, but the distribution among edges is iid,
    10 clients per edge, each edge have 10 classes
    :param num_edges: L
    :param clients:
    :param args:
    :return:
    """
    # only assign first (num_edges - 1), neglect the last 1, choose the left
    edges = []
    p_clients = [0.0] * num_edges
    for eid in range(num_edges):
        if eid == num_edges - 1:
            break
        assigned_clients_idxes = []
        for label in range(10):
            #     0-9 labels in total
            assigned_client_idx = np.random.choice(client_class_dis[label], 1, replace=False)
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
        edges.append(Edge(id=eid,
                          cids=assigned_clients_idxes,
                          shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
        [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
        edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
        p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                          for sample in list(edges[eid].sample_registration.values())]
        edges[eid].refresh_edgeserver()
    # And the last one, eid == num_edges -1
    eid = num_edges - 1
    assigned_clients_idxes = []
    for label in range(10):
        if not client_class_dis[label]:
            print("label{} is empty".format(label))
        else:
            assigned_client_idx = client_class_dis[label]
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
    edges.append(Edge(id=eid,
                      cids=assigned_clients_idxes,
                      shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
    [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
    edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
    p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                      for sample in list(edges[eid].sample_registration.values())]
    edges[eid].refresh_edgeserver()
    return edges, p_clients


def initialize_edges_niid(num_edges, clients, args, client_class_dis):
    """
    This function is specially designed for partiion for 10*L users, 1-class per user, but the distribution among edges is iid,
    10 clients per edge, each edge have 5 classes
    :param num_edges: L
    :param clients:
    :param args:
    :return:
    """
    # only assign first (num_edges - 1), neglect the last 1, choose the left
    edges = []
    p_clients = [0.0] * num_edges
    label_ranges = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [5, 6, 7, 8, 9], [6, 7, 8, 9, 0]]
    for eid in range(num_edges):
        if eid == num_edges - 1:
            break
        assigned_clients_idxes = []
        label_range = label_ranges[eid]
        for i in range(2):
            for label in label_range:
                #     5 labels in total
                if len(client_class_dis[label]) > 0:
                    assigned_client_idx = np.random.choice(client_class_dis[label], 1, replace=False)
                    client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
                else:
                    label_backup = 2
                    assigned_client_idx = np.random.choice(client_class_dis[label_backup], 1, replace=False)
                    client_class_dis[label_backup] = list(
                        set(client_class_dis[label_backup]) - set(assigned_client_idx))
                for idx in assigned_client_idx:
                    assigned_clients_idxes.append(idx)
        edges.append(Edge(id=eid,
                          cids=assigned_clients_idxes,
                          shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
        [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
        edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
        p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                          for sample in list(edges[eid].sample_registration.values())]
        edges[eid].refresh_edgeserver()
    # And the last one, eid == num_edges -1
    # Find the last available labels
    eid = num_edges - 1
    assigned_clients_idxes = []
    for label in range(10):
        if not client_class_dis[label]:
            print("label{} is empty".format(label))
        else:
            assigned_client_idx = client_class_dis[label]
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
    edges.append(Edge(id=eid,
                      cids=assigned_clients_idxes,
                      shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
    [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
    edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
    p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                      for sample in list(edges[eid].sample_registration.values())]
    edges[eid].refresh_edgeserver()
    return edges, p_clients


def all_clients_test(server, clients, cids, device):
    [server.send_to_client(clients[cid]) for cid in cids]
    for cid in cids:
        server.send_to_client(clients[cid])
        # The following sentence!
        clients[cid].sync_with_edgeserver()
    correct_edge = 0.0
    total_edge = 0.0
    for cid in cids:
        correct, total = clients[cid].test_model(device)
        correct_edge += correct
        total_edge += total
    return correct_edge, total_edge


def fast_all_clients_test(v_test_loader, global_nn, device):
    correct_all = 0.0
    total_all = 0.0
    loss_all = 0.0
    criterion = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for data in v_test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = global_nn(inputs)
            loss = criterion(outputs, labels)  # pylint: disable=E1102
            _, predicts = torch.max(outputs, 1)
            total_all += labels.size(0)
            correct_all += (predicts == labels).sum().item()
            loss_all += loss.item() * labels.size(0)
    return correct_all/total_all, loss_all/total_all




def initialize_global_nn(args):
    if args.dataset == 'mnist':
        if args.model == 'lenet':
            global_nn = mnist_lenet(input_channels=1, output_channels=10)
        elif args.model == 'logistic':
            global_nn = LogisticRegression_MNIST(input_dim=1, output_dim=10)
        elif args.model == 'cnn':
            global_nn = mnist_cnn(input_channels=1, output_channels=10)
        else:
            raise ValueError(f"Model{args.model} not implemented for mnist")
    elif args.dataset == 'femnist':
        if args.model == 'lenet':
            global_nn = mnist_lenet(input_channels=1, output_channels=62)
        elif args.model == 'logistic':
            global_nn = LogisticRegression_MNIST(input_dim=1, output_dim=62)
        elif args.model == 'cnn':
            global_nn = mnist_cnn(input_channels=1, output_channels=62)
        else:
            raise ValueError(f"Model{args.model} not implemented for femnist")
    elif args.dataset == 'cifar10' or args.dataset == 'cinic10':
        if args.model == 'cnn_complex':
            global_nn = cifar_cnn_3conv(input_channels=3, output_channels=10)
        elif args.model == 'resnet18':
            global_nn = ResNet18()
        elif args.model == 'resnet18_YWX':
            global_nn = ResNet18_YWX()
        else:
            raise ValueError(f"Model{args.model} not implemented for cifar")
    elif args.dataset == 'synthetic':
        if args.model == 'lr':
            global_nn = LogisticRegression_SYNTHETIC(args.dimension, args.num_class)
    else:
        raise ValueError(f"Dataset {args.dataset} Not implemented")
    return global_nn

def get_valid_loader(v_test_loader, subset_ratio = 0.5):
    # 收集每个类别的样本索引
    indices_per_class = {}
    for i, (_, label) in enumerate(v_test_loader.dataset):
        if hasattr(label, 'item'):
            label = label.item()
        if label not in indices_per_class:
            indices_per_class[label] = []
        indices_per_class[label].append(i)
    # 计算每个类别的样本数并按比例抽取
    selected_indices = []
    for label_indices in indices_per_class.values():
        num_samples = int(len(label_indices) * subset_ratio)
        selected_indices.extend(np.random.choice(label_indices, num_samples, replace=False))
    print(len(selected_indices))
    # 创建新的子集
    new_subset = Subset(v_test_loader.dataset, selected_indices)
    # 创建新的 DataLoader
    return DataLoader(new_subset, batch_size=v_test_loader.batch_size, shuffle=False)

# 分层联邦总聚合
def HierFAVG(args):
    # make experiments repeatable
    global avg_acc_v
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cuda_to_use = torch.device(f'cuda:{args.gpu}')
    device = cuda_to_use if torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')
    FILEOUT = f"{args.dataset}_clients{args.num_clients}_edges{args.num_edges}_" \
              f"t1-{args.num_local_update}_t2-{args.num_edge_aggregation}" \
              f"_model_{args.model}iid{args.iid}edgeiid{args.edgeiid}epoch{args.num_communication}" \
              f"bs{args.batch_size}lr{args.lr}lr_decay_rate{args.lr_decay}" \
              f"lr_decay_epoch{args.lr_decay_epoch}momentum{args.momentum}"
    writer = SummaryWriter(comment=FILEOUT)

    # 读取mapping配置信息
    mapping = json.loads(args.edge_client_class_mapping)
    com_client_mapping = json.loads(args.com_client_mapping)
    com_edge_mapping = json.loads(args.com_edge_mapping)

    edge_client_mapping = {}
    client_class_mapping = {}
    for group_number, clients in mapping.items():
        # group_number = int(group.split()[1])
        edge_client_mapping[int(group_number)] = []
        for client, classes in clients.items():
            edge_client_mapping[int(group_number)].append(int(client))
            client_class_mapping[int(client)] = classes
    args.client_class_mapping = json.dumps(client_class_mapping)

    print(edge_client_mapping)
    print(client_class_mapping)
    # Build dataloaders
    train_loaders, test_loaders, v_test_loader = get_dataloaders(args)

    # New an NN model for testing error
    global_nn = initialize_global_nn(args)
    if args.cuda:
        global_nn = global_nn.cuda(device)

    detect_loader = None
    test_nn = None
    if args.mode == 1:
        detect_loader = get_valid_loader(v_test_loader)
        test_nn = copy.deepcopy(global_nn)

    if args.show_dis:
        # 训练集加载器划分
        for i in range(args.num_clients):
            train_loader = train_loaders[i]
            print(len(train_loader.dataset))
            distribution = show_distribution(train_loader, args)
            print("train dataloader {} distribution".format(i))
            print(distribution)
        # 测试集加载器划分
        for i in range(args.num_clients):
            test_loader = test_loaders[i]
            test_size = len(test_loader.dataset)
            print(len(test_loader.dataset))
            if args.test_on_all_samples != 1:
                distribution = show_distribution(test_loader, args)
                print(distribution)
            print("test dataloader {} distribution".format(i))
            print(f"test dataloader size {test_size}")

        print("Cloud valid data size: {}".format(len(v_test_loader.dataset)))
        valid_data_distribution = show_distribution(v_test_loader, args)
        print("Cloud valid data distribution: {}".format(valid_data_distribution))
        if args.mode == 1:
            print("Cloud quality detect data size: {}".format(len(detect_loader.dataset)))
            valid_data_distribution = show_distribution(detect_loader, args)
            print("Cloud quality detect data distribution: {}".format(valid_data_distribution))
    # Assuming the provided image data is stored as a dictionary string


    # initialize clients and server
    clients = []
    for i in range(args.num_clients):
        # 初始化客户端
        clients.append(Client(id=i,
                              train_loader=train_loaders[i],
                              test_loader=test_loaders[i],
                              args=args,
                              device=device,com_params=com_client_mapping[str(i)]) )

    initilize_parameters = list(clients[0].model.shared_layers.parameters())
    nc = len(initilize_parameters)
    for client in clients:
        user_parameters = list(client.model.shared_layers.parameters())
        for i in range(nc):
            user_parameters[i].data[:] = initilize_parameters[i].data[:]

    # Initialize edge server and assign clients to the edge server
    edges = []
    cids = np.arange(args.num_clients)
    clients_per_edge = int(args.num_clients / args.num_edges)
    p_clients = [0.0] * args.num_edges


    if args.iid == -2:
        if args.edgeiid == 1:
            client_class_dis = get_client_class(args, clients)
            edges, p_clients = initialize_edges_iid(num_edges=args.num_edges,
                                                    clients=clients,
                                                    args=args,
                                                    client_class_dis=client_class_dis)
        elif args.edgeiid == 0:
            client_class_dis = get_client_class(args, clients)
            edges, p_clients = initialize_edges_niid(num_edges=args.num_edges,
                                                     clients=clients,
                                                     args=args,
                                                     client_class_dis=client_class_dis)
    else:
        # This is randomly assign the clients to edges
        for i in range(args.num_edges):
            # 根据映射关系进行选择
            selected_cids = edge_client_mapping[i]
            print(f"Edge {i} has clients {selected_cids}")
            cids = list(set(cids) - set(selected_cids))
            edges.append(Edge(id=i,
                              cids=selected_cids,
                              shared_layers=copy.deepcopy(clients[0].model.shared_layers),com_params=com_edge_mapping[str(i)],valid_loader=detect_loader, valid_nn=test_nn))
            [edges[i].client_register(clients[cid]) for cid in selected_cids]
            edges[i].all_trainsample_num = sum(edges[i].sample_registration.values())
            p_clients[i] = [sample / float(edges[i].all_trainsample_num) for sample in
                            list(edges[i].sample_registration.values())]

    # Initialize cloud server
    cloud = Cloud(shared_layers=copy.deepcopy(clients[0].model.shared_layers), valid_loader=detect_loader, valid_nn=test_nn)
    # First the clients report to the edge server their training samples
    [cloud.edge_register(edge=edge) for edge in edges]
    p_edge = [sample / sum(cloud.sample_registration.values()) for sample in
              list(cloud.sample_registration.values())]
    cloud.refresh_cloudserver()

    # 开始训练
    # accs_edge_avg = []  # 记录云端的平均边缘测试精度
    # losses_edge_avg = []  # 记录云端的平均边缘损失
    accs_cloud = [0.0]  # 记录每轮云端聚合的精度
    times = [0]  # 记录每个云端轮结束的时间戳
    # 获取初始时间戳（训练开始时）
    start_time = time.time()
    for num_comm in tqdm(range(args.num_communication)):  # 云聚合
        cloud.refresh_cloudserver()
        [cloud.edge_register(edge=edge) for edge in edges]
        all_loss_sum = 0.0
        all_acc_sum = 0.0
        print(f"云端更新   第 {num_comm} 轮")
        for num_edgeagg in range(args.num_edge_aggregation):  # 边缘聚合
            print(f"边缘更新   第 {num_edgeagg} 轮")
            if torch.cuda.is_available():
                torch.cuda.set_device("cuda:0")
            # 多线程的边缘迭代
            edge_threads = []
            edge_loss = [0.0] * len(edges)
            edge_sample = [0] * len(edges)
            for edge in edges:
                edge_thread = Thread(target=process_edge, args=(edge, clients, args, device, edge_loss, edge_sample))
                edge_threads.append(edge_thread)
                edge_thread.start()
                print(f"Edge {edge.id} thread started.")  # 确认线程启动的日志
            for edge_thread in edge_threads:
                edge_thread.join()
                print(f"Edge thread joined.")  # 确认线程结束的日志

            # 统计边缘迭代的损失和样本
            total_samples = sum(edge_sample)
            if total_samples > 0:
                all_loss = sum([e_loss * e_sample for e_loss, e_sample in zip(edge_loss, edge_sample)]) / total_samples
                all_loss_sum += all_loss
            else:
                print("Warning: Total number of samples is zero. Cannot compute all_loss.")
            print("train loss per edge on all samples: {}".format(edge_loss))

        # print(models_are_equal(edges[0].shared_state_dict, edges[1].shared_state_dict))
        # 开始云端聚合
        for edge in edges:
            edge.send_to_cloudserver(cloud)

        # if args.mode == 1:  # 开启基线
        #     cloud.quality_aggregate(device, args.threshold, args.score_init)
        # else:
        cloud.aggregate(args)
        print(f"Cloud 聚合")

        for edge in edges:
            cloud.send_to_edge(edge)
        global_nn.load_state_dict(state_dict=copy.deepcopy(cloud.shared_state_dict))
        global_nn.train(False)

        # 云端测试
        print(f"Cloud 测试")
        avg_acc_v, _ = fast_all_clients_test(v_test_loader, global_nn, device)
        print('Cloud Valid Accuracy {}'.format(avg_acc_v))
        # 在轮次结束时记录相对于开始时间的时间差, 记录云端轮的测试精度
        times.append(time.time() - start_time)
        accs_cloud.append(avg_acc_v)

    # 画出云端的精度-时间曲线图
    plt.plot(times, accs_cloud, marker='v', color='r', label="HierFL")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Test Model Accuracy')
    plt.title('Test Accuracy over Time')
    plt.show()

def train_client(client, edge, num_iter, device, return_dict):
    try:
        print(f"Client {client.id} training started.")  # 开始日志
        # 客户端与边缘服务器同步
        edge.send_to_client(client)
        client.sync_with_edgeserver()
        # 执行本地迭代
        client_loss = client.local_update(num_iter=num_iter, device=device)
        # 将迭代后的模型发送回边缘服务器
        client.send_to_edgeserver(edge)
        # 存储结果
        return_dict[client.id] = client_loss
        print(f"Client {client.id} training finished.")  # 结束日志
    except Exception as e:
        print(f"Error in client {client.id} training: {e}")  # 异常日志


def process_edge(edge, clients, args, device, edge_loss, edge_sample):
    # 一次边缘迭更新 = n个本地迭代+ 一次边缘聚合
    # print(f"Edge {edge.id} 边缘更新开始")
    # 使用多线程进行客户迭代
    threads = []
    return_dict = {}  # 在线程中，可以直接使用普通字典
    for selected_cid in edge.cids:
        client = clients[selected_cid]
        thread = Thread(target=train_client,
                        args=(client, edge, args.num_local_update, device, return_dict))
        threads.append(thread)
        thread.start()
        print(f"Client {selected_cid} thread started.")  # 确认线程启动的日志
    # 等待所有线程完成
    for thread in threads:
        thread.join()
        print(f"Client thread joined.")  # 确认线程结束的日志
    # 边缘聚合
    if args.mode == 1:  # 开启基线
        edge.quality_aggregate(device, args.threshold, args.score_init)
    else:
        edge.aggregate(args)
    # 更新边缘训练损失
    edge_loss[edge.id] = sum(return_dict.values())
    edge_sample[edge.id] = sum(edge.sample_registration.values())




def main():
    args = args_parser()
    print(args.dataset_root)
    HierFAVG(args)


if __name__ == '__main__':
    main()
