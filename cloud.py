# The structure of the server
# The server should include the following functions:
# 1. Server initialization
# 2. Server reveives updates from the user
# 3. Server send the aggregated information back to clients
import copy
from scipy.stats import norm
import numpy as np
import torch
from torch import nn
from average import average_weights, average_weights_simple


class Cloud():

    def __init__(self, shared_layers, valid_loader=None, valid_nn=None):
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        # self.shared_state_dict = shared_layers.state_dict()
        self.clock = []
        self.valid_loader = valid_loader
        self.valid_nn = valid_nn
        self.virtual_queue = []  # 边缘虚拟队列 （更新后选择）
        self.latency_queue = []  # 边缘延迟队列 （积分更新）

    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def receive_from_edge(self, edge_id, eshared_state_dict):
        self.receiver_buffer[edge_id] = eshared_state_dict
        return None

    def aggregate(self, args):
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w=received_dict,
                                                 s_num=sample_num)
        return None

    # 边缘选择（基于历史时间估计）
    def filter_edge_models(self):

    # 边缘权重更新
    def cal_edge_weights(self):

    def send_to_edge(self, edge):
        edge.receive_from_cloudserver(copy.deepcopy(self.shared_state_dict))
        return None

    def quality_aggregate(self, device, delta_threshold, score_init):
        self.valid_nn.train(False)
        _, global_loss = valid_loss_test(self.valid_loader, self.valid_nn, device)  # Get global model's loss

        w_locals_pass = []  # Models that pass quality detection
        alpha_values = []  # Weights for models that pass quality detection
        deltas = []  # Marginal losses for models
        min_delta = float('inf')

        all_weights = list(self.receiver_buffer.values())

        for edge_id, w in self.receiver_buffer.items():
            other_weights = [model_w for model_w in all_weights if model_w is not w]
            if other_weights:
                aggregated_weights = average_weights_simple(other_weights)
                self.valid_nn.load_state_dict(aggregated_weights)
                _, loss_i = valid_loss_test(self.valid_loader, self.valid_nn, device)
            else:
                loss_i = global_loss

            # Calculate the delta loss
            delta_i = loss_i - global_loss
            print(delta_i)
            if delta_i >= delta_threshold:
                deltas.append(delta_i)
                w_locals_pass.append(w)
                min_delta = min(min_delta, delta_i)

        score_sum = sum(delta - min_delta for delta in deltas)
        for delta in deltas:
            score = (delta - min_delta) / score_sum
            alpha = score_init + score
            alpha_values.append(alpha)

        # 质量得分聚合
        self.shared_state_dict = average_weights(w_locals_pass, alpha_values)


def valid_loss_test(v_test_loader, global_nn, device):
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
    return correct_all / total_all, loss_all / total_all


def estimate_posterior_parameters(prior_mean, prior_variance, data, sample_size=100):
    """
    根据先验参数和观测数据，估计后验分布参数，并生成预测样本。

    参数:
        prior_mean (float): 先验均值
        prior_variance (float): 先验方差
        data (np.array): 观测数据
        sample_size (int): 生成的样本数量

    返回:
        tuple: 包含更新后的均值、估计的方差和预测样本数组
    """
    # 数据数量
    n_data = len(data)

    # 使用观测数据来估计方差
    data_variance = np.var(data, ddof=1)

    # 估计后验方差的点估计
    posterior_variance_point_estimate = 1 / (1 / prior_variance + n_data / data_variance)

    # 更新均值
    posterior_mean = (prior_mean / prior_variance + np.sum(data) / data_variance) / (
            1 / prior_variance + n_data / data_variance)

    # 预估下一轮数据的分布
    predicted_samples = norm.rvs(loc=posterior_mean, scale=np.sqrt(posterior_variance_point_estimate), size=sample_size)

    return posterior_mean, posterior_variance_point_estimate, predicted_samples