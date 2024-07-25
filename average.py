import copy
import torch
from torch import nn
# 权重平均聚合
def average_weights(w, s_num):
    #copy the first client's weights
    total_sample_num = sum(s_num)
    # print(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  #the nn layer loop
        for i in range(1, len(w)):   #the client loop
            # w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)
            # result type Float can't be cast to the desired output type Long
            w_avg[k] = w_avg[k] + torch.mul(w[i][k], s_num[i] / temp_sample_num)
        w_avg[k] = torch.mul(w_avg[k], temp_sample_num/total_sample_num)
    return w_avg

def average_weights_simple(w):
    # copy the first client's weights
    total_num = len(w)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  # the nn layer loop
        for i in range(1, len(w)):  # the client loop
            # w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)
            # result type Float can't be cast to the desired output type Long
            w_avg[k] = w_avg[k] + w[i][k]
        w_avg[k] = torch.mul(w_avg[k], 1 / total_num)
    return w_avg

def _modeldict_scale(md, c):
    """模型参数缩放"""
    res = {}
    for layer in md.keys():
        if md[layer] is None:
            res[layer] = None
            continue
        res[layer] = md[layer] * c
    return res

def _modeldict_add(md1, md2):
    """模型参数相加"""
    res = {}
    for layer in md1.keys():
        if md1[layer] is None:
            res[layer] = None
            continue
        res[layer] = md1[layer] + md2[layer]
    return res