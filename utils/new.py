import numpy as np
import torch


def generate_actions_pytorch(dbs_torch, config):
    batch_size = config.batch_size
    num_actions = config.num_shape_params*2
    action_ch = torch.zeros((batch_size, num_actions))
    for batch_idx,max_abs_ind in enumerate(torch.argmax(abs(dbs_torch),axis = 1)):
        if dbs_torch[batch_idx][max_abs_ind] > 0:
            action_ch[batch_idx][max_abs_ind*2] = 1
        else:
            action_ch[batch_idx][max_abs_ind*2+1] = 1
    action_ind_ch = torch.argmax(action_ch,axis = 1)
    # print("action:",action_ch)
    # print("action index:",action_ind_ch)
    return action_ch,action_ind_ch    


def generate_actions_np(dbs, config):
    batch_size = config.batch_size
    num_actions = config.num_shape_params*2
    action_ch = np.zeros((batch_size, num_actions))
    for batch_idx,max_abs_ind in enumerate(np.argmax(abs(dbs),axis = 1)):
        if dbs[batch_idx][max_abs_ind] > 0:
            action_ch[batch_idx][max_abs_ind*2] = 1
        else:
            action_ch[batch_idx][max_abs_ind*2+1] = 1
    action_ind_ch = np.argmax(action_ch,axis = 1)
    # print("action:",action_ch)
    # print("action index:",action_ind_ch)
    return action_ch,action_ind_ch    