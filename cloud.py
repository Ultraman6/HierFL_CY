# The structure of the server
# The server should include the following functions:
# 1. Server initialization
# 2. Server reveives updates from the user
# 3. Server send the aggregated information back to clients
import copy

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
