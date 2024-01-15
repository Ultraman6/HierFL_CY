import argparse
import json
import os

import torch


def args_parser():
    parser = argparse.ArgumentParser()
    # dataset and model
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        help='name of the dataset: mnist, cifar10, femnist, synthetic, cinic10'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='cnn',
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
    # nn training hyper parameter
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20,
        help='batch size when trained on client'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=6,
        help='numworks for dataloader'
    )
    parser.add_argument(
        '--test_ratio',
        type=int,
        default=0.1,
        help='ratio of test dataset'
    )
    # -------------云聚合轮次、边缘聚合轮次、本地更新轮次
    parser.add_argument(
        '--num_communication',
        type=int,
        default=200,
        help='number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--num_edge_aggregation',
        type=int,
        default=10,
        help='number of edge aggregation (K_2)'
    )
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=6,
        help='number of local update (K_1)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='learning rate of the SGD when trained on client'
    )
    parser.add_argument(
        '--lr_decay',
        type=float,
        default='1',
        help='lr decay rate'
    )
    parser.add_argument(
        '--lr_decay_epoch',
        type=int,
        default=1,
        help='lr decay epoch'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='SGD momentum'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=5e-3,
        help='The weight decay rate'
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
        type=int,
        default=1,
        help='distribution of the data, 1,0,-1,-2 分别表示iid同大小、niid同大小、iid不同大小、niid同大小且仅一类(one-class)'
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
        default=50,
        help='number of all available clients'
    )
    parser.add_argument(
        '--num_edges',
        type=int,
        default=5,
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
        "15": [1, 1, 1], "16": [1, 1, 1], "17": [1, 1, 1], "18": [1, 1, 1], "19": [1, 1, 1], "20": [1, 1, 1],
        "21": [1, 1, 1], "22": [1, 1, 1], "23": [1, 1, 1], "24": [1, 1, 1],
        "25": [1, 1, 1], "26": [1, 1, 1], "27": [1, 1, 1], "28": [1, 1, 1], "29": [1, 1, 1],
        "30": [1, 1, 1], "31": [1, 1, 1], "32": [1, 1, 1], "33": [1, 1, 1], "34": [1, 1, 1],
        "35": [1, 1, 1], "36": [1, 1, 1], "37": [1, 1, 1], "38": [1, 1, 1], "39": [1, 1, 1], "40": [1, 1, 1],
        "41": [1, 1, 1], "42": [1, 1, 1], "43": [1, 1, 1], "44": [1, 1, 1],
        "45": [1, 1, 1], "46": [1, 1, 1], "47": [1, 1, 1], "48": [1, 1, 1], "49": [1, 1, 1],
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
        "3": [1, 1, 1],
        "4": [1, 1, 1],
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
    # AvgJSD = 0.6931471805599453
    # mapping_json = json.dumps({
    #     0: {0: [1], 1: [1], 2: [1], 3: [1], 4: [1], 5: [2], 6: [2], 7: [2], 8: [2], 9: [2]},
    #     1: {10: [6], 11: [6], 12: [6], 13: [6], 14: [6], 15: [7], 16: [7], 17: [7], 18: [7], 19: [7]},
    #     2: {20: [4], 21: [4], 22: [4], 23: [4], 24: [4], 25: [8], 26: [8], 27: [8], 28: [8], 29: [8]},
    #     3: {30: [3], 31: [3], 32: [3], 33: [3], 34: [3], 35: [9], 36: [9], 37: [9], 38: [9], 39: [9]},
    #     4: {40: [0], 41: [0], 42: [0], 43: [0], 44: [0], 45: [5], 46: [5], 47: [5], 48: [5], 49: [5]}
    # })

    # AvJSD = 0
    mapping_json = json.dumps({
        0: {49: [0], 0: [1], 3: [2], 4: [3], 23: [4], 8: [5], 45: [6], 14: [7], 46: [8], 38: [9]},
        1: {19: [0], 41: [1], 2: [2], 33: [3], 6: [4], 35: [5], 10: [6], 13: [7], 15: [8], 18: [9]},
        2: {29: [0], 30: [1], 21: [2], 32: [3], 22: [4], 43: [5], 25: [6], 37: [7], 26: [8], 16: [9]},
        3: {28: [0], 31: [1], 1: [2], 34: [3], 24: [4], 42: [5], 11: [6], 12: [7], 27: [8], 39: [9]},
        4: {48: [0], 40: [1], 20: [2], 5: [3], 7: [4], 9: [5], 44: [6], 36: [7], 47: [8], 17: [9]}
    })

    # AvgJSD = 0.4744959201471498
    # mapping_json = json.dumps({
    #     0: {0: [1], 1: [1], 2: [1], 3: [2], 4: [2], 5: [2], 6: [3], 7: [3], 8: [3], 9: [4]},
    #     1: {10: [5], 11: [5], 12: [5], 13: [6], 14: [6], 15: [6], 16: [7], 17: [7], 18: [7], 19: [8]},
    #     2: {20: [9], 21: [9], 22: [9], 23: [0], 24: [0], 25: [0], 26: [4], 27: [4], 28: [4], 29: [5]},
    #     3: {30: [8], 31: [8], 32: [8], 33: [0], 34: [0], 35: [1], 36: [1], 37: [2], 38: [2], 39: [3]},
    #     4: {40: [3], 41: [4], 42: [5], 43: [6], 44: [6], 45: [7], 46: [7], 47: [8], 48: [9], 49: [9]}
    # })

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
    dataset_root = os.path.join(os.getcwd(), 'train_data')
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
        default=0,
        help='1 means test on all samples, 0 means test samples will be split averagely to each client'
    )

    # synthetic数据集用参数
    parser.add_argument(
        '--alpha',
        type=int,
        default=1,
        help='means the mean of distributions among clients'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=1,
        help='means the variance  of distributions among clients'
    )
    parser.add_argument(
        '--dimension',
        type=int,
        default=60,
        help='1 means mapping is active, 0 means mapping is inactive'
    )
    parser.add_argument(
        '--num_class',
        type=int,
        default=10,
        help='1 means mapping is active, 0 means mapping is inactive'
    )

    # 新~数据划分方法
    parser.add_argument(
        '--partition',
        type=str,
        default='noniid-labeldir',
        help='划分类型，homo、 noniid-labeldir、 noniid-#label1、 iid-diff-quantity, '
             '其中label后的数字表示每个客户的类别数'
    )
    parser.add_argument(
        '--beta_new',
        type=float,
        default=1,
        help='dir分布的超参数'
    )

    # 基于边际测试损失的聚合
    parser.add_argument(
        '--mode',
        type=float,
        default=1,
        help='1 表示开启, 0 表示关闭'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=-0.01,
        help='边际损失阈值'
    )
    parser.add_argument(
        '--score_init',
        type=float,
        default=1,
        help='初始得分'
    )

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
