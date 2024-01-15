# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

import copy
from average import average_weights, average_weights_simple
from cloud import valid_loss_test


class Edge():

    def __init__(self, id, cids, shared_layers, com_params, valid_loader=None, valid_nn=None):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_state_dict: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param shared_layers: Structure of the shared layers
        :return:
        """
        self.id = id
        self.cids = cids
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.shared_state_dict = shared_layers.state_dict()
        self.clock = []
        # 边缘服务器上行链路用参数
        self.X=com_params[0]
        self.Q=com_params[1]
        self.W=com_params[2]
        # 存放每边缘轮客户上传的速率, 键值对存放
        self.client_rates={id:0 for id in cids}
        self.valid_loader = valid_loader
        self.valid_nn = valid_nn

    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def client_register(self, client):
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = len(client.train_loader.dataset)
        return None

    def receive_from_client(self, client_id, cshared_state_dict, rate):
        self.receiver_buffer[client_id] = cshared_state_dict
        self.client_rates[client_id] = rate
        return None

    def aggregate(self, args):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w = received_dict,
                                                 s_num= sample_num)

    def quality_aggregate(self, device, delta_threshold, score_init):
        w_locals_pass = []  # Models that pass quality detection
        alpha_values = []  # Weights for models that pass quality detection
        deltas = []  # Marginal losses for models

        all_weights = list(self.receiver_buffer.values())
        self.valid_nn.load_state_dict(average_weights_simple(all_weights))
        self.valid_nn.train(False)
        _, global_loss = valid_loss_test(self.valid_loader, self.valid_nn, device)  # Get global model's loss

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

        min_delta = min(deltas)
        score_sum = sum(delta - min_delta for delta in deltas)
        for delta in deltas:
            score = (delta - min_delta) / score_sum
            alpha = score_init + score
            alpha_values.append(alpha)

        alpha_sum = sum(alpha_values)
        alpha_values = [alpha / alpha_sum for alpha in alpha_values]

        if len(w_locals_pass) != 0:
            # 质量得分聚合
            self.shared_state_dict = average_weights(w_locals_pass, alpha_values)


    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.shared_state_dict))
        return None

    def send_to_cloudserver(self, cloud):
        cloud.receive_from_edge(edge_id=self.id,
                                eshared_state_dict= copy.deepcopy(
                                    self.shared_state_dict))
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        self.shared_state_dict = shared_state_dict
        return None



