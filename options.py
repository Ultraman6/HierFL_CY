import argparse
import json
import os

import torch


def args_parser():
    parser = argparse.ArgumentParser()
    # dataset and model
    parser.add_argument(
        '--dataset',
        type = str,
        default = 'synthetic',
        help = 'name of the dataset: mnist, cifar10, femnist, synthetic, cinic10'
    )
    parser.add_argument(
        '--model',
        type = str,
        default = 'lr',
        help='name of model. mnist: logistic, lenet, cnn; '
             'cifar10、cinic10: resnet18, resnet18_YWX, cnn_complex; femnist: logistic, lenet, cnn; synthetic:lr'
    )
    parser.add_argument(
        '--input_channels',
        type=int,
        default=1,
        help='input channels. femnist:1, mnist:1, cifar10 :3'
    )
    parser.add_argument(
        '--output_channels',
        type=int,
        default=62,
        help='output channels. femnist:62'
    )
    # nn training hyper parameter
    #nn training hyper parameter
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 10,
        help = 'batch size when trained on client'
    )
    parser.add_argument(
        '--num_workers',
        type = int,
        default = 4,
        help = 'numworks for dataloader'
    )
    parser.add_argument(
        '--test_ratio',
        type = int,
        default = 0.1,
        help = 'ratio of test dataset'
    )
    # -------------云聚合轮次、边缘聚合轮次、本地更新轮次
    parser.add_argument(
        '--num_communication',
        type=int,
        default=100,
        help='number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--num_edge_aggregation',
        type=int,
        default=2,
        help='number of edge aggregation (K_2)'
    )
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=1,
        help='number of local update (K_1)'
    )
    parser.add_argument(
        '--lr',
        type = float,
        default = 0.06,
        help = 'learning rate of the SGD when trained on client'
    )
    parser.add_argument(
        '--lr_decay',
        type = float,
        default= '1',
        help = 'lr decay rate'
    )
    parser.add_argument(
        '--lr_decay_epoch',
        type = int,
        default=1,
        help= 'lr decay epoch'
    )
    parser.add_argument(
        '--momentum',
        type = float,
        default = 0,
        help = 'SGD momentum'
    )
    parser.add_argument(
        '--weight_decay',
        type = float,
        default = 0,
        help= 'The weight decay rate'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
        help='verbose for print progress bar'
    )
    # setting for federeated learning
    parser.add_argument(
        '--iid',
        type = int,
        default = 1,
        help = 'distribution of the data, 1,0,-1,-2 分别表示iid同大小、niid同大小、iid不同大小、niid同大小且仅一类(one-class)'
    )
    parser.add_argument(
        '--edgeiid',
        type=int,
        default=1,
        help='distribution of the data under edges, 1 (edgeiid),0 (edgeniid) (used only when iid = -2)'
    )
    parser.add_argument(
        '--frac',
        type=float,
        default=1,
        help='fraction of participated clients'
    )
    # -------------客户端数、边缘服务器数、客户端训练样本量
    parser.add_argument(
        '--num_clients',
        type=int,
        default=21,
        help='number of all available clients'
    )
    parser.add_argument(
        '--num_edges',
        type=int,
        default=3,
        help='number of edges'
    )

    # editer: Extrodinary 20231215
    # 定义clients及其分配样本量的关系（弃用）
    parser.add_argument(
        '--self_sample',
        default=-1,
        type=int,
        help='>=0: set sample of each client， -1: all samples'
    )
    # 将映射关系转换为JSON格式，主键个数必须等于num_edges，value为-1表示all samples
    sample_mapping_json = json.dumps({
        "0": 3000,
        "1": 2000,
        "2": 5000,
        "3": 10000,
        "4": -1,
    })
    parser.add_argument(
        '--sample_mapping',
        type=str,
        default=sample_mapping_json,
        help='mapping of clients and their samples'
    )
    # 定义client的通信参数
    # 将映射关系转换为JSON格式，主键个数必须等于num_clients，value表示三个参数
    com_client_mapping_json = json.dumps({
        "0": [1, 1, 1], "1": [1, 1, 1], "2": [1, 1, 1], "3": [1, 1, 1], "4": [1, 1, 1],
        "5": [1, 1, 1], "6": [1, 1, 1], "7": [1, 1, 1], "8": [1, 1, 1], "9": [1, 1, 1],
        "10": [1, 1, 1], "11": [1, 1, 1], "12": [1, 1, 1], "13": [1, 1, 1], "14": [1, 1, 1],
        "15": [1, 1, 1], "16": [1, 1, 1], "17": [1, 1, 1], "18": [1, 1, 1], "19": [1, 1, 1],"20": [1, 1, 1],
    })
    parser.add_argument(
        '--com_client_mapping',
        type=str,
        default=com_client_mapping_json,
        help='mapping of clients and their params in communication'
    )
    # 定义edge的通信参数
    # 将映射关系转换为JSON格式，主键个数必须等于num_edges，value表示三个参数
    com_edge_mapping_json = json.dumps({
        "0": [1, 1, 1],
        "1": [1, 1, 1],
        "2": [1, 1, 1],
    })
    parser.add_argument(
        '--com_edge_mapping',
        type=str,
        default=com_edge_mapping_json,
        help='mapping of edges and their params in communication'
    )

    parser.add_argument(
        '--self_class',
        default=1,
        type=int,
        help='>=1: set class of each client， 0: auto'
    )
    mapping_json = json.dumps({
        0: {0: [0], 1: [1], 2: [2], 3: [2], 4: [2], 5: [3]},
        1: {6: [4], 7: [4], 8: [5], 9: [5], 10: [6], 11: [6], 12: [9]},
        2: {13: [7], 14: [7], 15: [8], 16: [8], 17: [8], 18: [7], 19: [5], 20: [3]}
    })
    # 将映射关系转换为JSON格式，主键个数必须等于num_edges，value为-1表示all samples
    parser.add_argument(
        '-edge_client_class_mapping',
        type=str,
        default=mapping_json,
        help='mapping of clients and their class'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed (defaul: 1)'
    )

    # 设置数据集的根目录为家目录下的 train_data 文件夹
    dataset_root = os.path.join(os.path.expanduser('~'), 'train_data')
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)

    parser.add_argument(
        '--dataset_root',
        type=str,
        default=dataset_root,
        help='dataset root folder'
    )
    parser.add_argument(
        '--show_dis',
        type=int,
        default=1,
        help='whether to show distribution'
    )
    parser.add_argument(
        '--classes_per_client',
        type=int,
        default=2,
        help='under artificial non-iid distribution, the classes per client'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU to be selected, 0, 1, 2, 3'
    )

    parser.add_argument(
        '--mtl_model',
        default=0,
        type=int
    )
    parser.add_argument(
        '--global_model',
        default=1,
        type=int
    )
    parser.add_argument(
        '--local_model',
        default=0,
        type=int
    )

    # editer: Sensorjang 20230925
    parser.add_argument(
        '--test_on_all_samples',
        type=int,
        default=1,
        help='1 means test on all samples, 0 means test samples will be split averagely to each client'
    )


    # synthetic数据集用参数
    parser.add_argument(
        '--alpha',
        type = int,
        default = 1,
        help = 'means the mean of distributions among clients'
    )
    parser.add_argument(
        '--beta',
        type = float,
        default = 1,
        help = 'means the variance  of distributions among clients'
    )
    parser.add_argument(
        '--dimension',
        type = int,
        default = 60,
        help = '1 means mapping is active, 0 means mapping is inactive'
    )
    parser.add_argument(
        '--num_class',
        type = int,
        default = 10,
        help = '1 means mapping is active, 0 means mapping is inactive'
    )

    # 新~数据划分方法
    parser.add_argument(
        '--partition',
        type = str,
        default = 'noniid-labeldir',
        help = '划分类型，homo、 noniid-labeldir、 noniid-#label1、 iid-diff-quantity, '
               '其中label后的数字表示每个客户的类别数'
    )
    parser.add_argument(
        '--beta_new',
        type = float,
        default = 1,
        help = 'dir分布的超参数'
    )


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
