#Code for training one step

### Pytorch
import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
import numpy as np
from . import autoencoder, new

def calculate_direct_coord_loss(bs, yc, yr, yc_gt, yr_gt, config, bs_gt):
    bs = torch.from_numpy(bs).float().to(config.device)
    # bs_gt_pytorch = torch.from_numpy(bs_gt).float().to(config.device)#[1,3*num_landmarks] == [1,6]
    bs_gt_pytorch = bs_gt.to(config.device)
    bs_gt_pytorch = torch.reshape(bs_gt_pytorch, (config.landmark_count,-1))
    action_prob = torch.exp(yc - torch.unsqueeze(torch.amax(yc,axis = 1),1))
    action_prob = action_prob/ torch.unsqueeze(torch.sum(action_prob,axis = 1),1)
    bs = bs - yr*torch.amax(torch.reshape(action_prob,(bs.size()[0],bs.size()[1],2)),axis = 2)
    
    updated_predicted_landmarks_pytorch = torch.reshape(bs,(config.batch_size,config.landmark_count,-1))
    updated_predicted_landmarks_pytorch = torch.mean(updated_predicted_landmarks_pytorch,axis = 0)
    ret = nn.MSELoss()(updated_predicted_landmarks_pytorch, bs_gt_pytorch)
    return ret

#train_pairs(step_i, patches, actions, dbs, config, models, criterions, optimizers, bs, bs_gt)
def train_pairs(step_i, patches_train, yc_, yr_, config, models, criterions, optimizers, bs, bs_gt):
    """
    Args:
        patches_train: x (input), [batch_size, box_size, box_size, 3*num_landmarks]
        actions_train: classification labels
        dbs_train: regression labels
        config: fixed parameters
        models: models
    """
    models['model'].train()
    patches_train = torch.from_numpy(patches_train).float().to(config.device)
    #[batch_size, box_size, box_size, 3*num_landmarks] -> [batch_size, 3*num_landmarks, box_size, box_size]
    patches_train = patches_train.permute(0, 3, 1, 2)
    # print(patches_train.size())
    # print("patches_train shape: {}".format(patches_train.size()))

    yc_,yr_ = yc_.to(config.device), yr_.to(config.device)
    yc, yr = models['model'](patches_train)

    ### Calculate Loss
    loss_c = criterions['cls'](yc_, logits = yc)
    loss_r = torch.sqrt(torch.sqrt(torch.sqrt(criterions['reg'](yr_,yr))))#RMSE
    loss = config.alpha*loss_c + (1-config.alpha)*loss_r# + loss_d
    loss.backward()
    optimizers['optimizer'].step()
    optimizers['optimizer'].zero_grad()
    action_ind = torch.argmax(yc,axis = 1)
    action_prob = nn.Softmax(dim = 1)(yc)
    gt_cls = torch.argmax(yc_,axis = 1)
    
    assert yr.size() == yr_.size()
    assert yc.size() == yc_.size()
    
    # updated_bs = bs - yr*np.amax(np.reshape(action_prob,(bs.shape[0], bs.shape[1],2)),axis = 2)
    updated_bs = bs - yr*torch.amax(torch.reshape(action_prob,(bs.size()[0],bs.size()[1],2)),2)
    
    landmarks = autoencoder.b2landmarks(updated_bs,models['shape_model'],config)
    
    udpated_predicted_landmarks = landmarks.reshape((config.batch_size, config.landmark_count, -1))
    udpated_predicted_landmarks = torch.mean(udpated_predicted_landmarks,axis = 0)
    ground_truth_landmarks = autoencoder.b2landmarks(bs_gt, models['shape_model'], config)
    ground_truth_landmarks = ground_truth_landmarks.reshape((config.batch_size, config.landmark_count, -1))
    ground_truth_landmarks = torch.mean(ground_truth_landmarks,axis = 0)
    assert udpated_predicted_landmarks.shape == (config.landmark_count, 3)
    
    if step_i%config.print_freq == 0:
        print("\n\n\n")
        print("========Is regression right????========")
        print("predicted dbs: {}".format(yr))
        print("GT dbs: {}".format(yr_))
        print("action_ind:",action_ind.cpu().detach().numpy())
        print("gt cls:",gt_cls)
        for i in range(config.landmark_count):
            print("predicted landmark {}: ({}, {}, {})".format(i, udpated_predicted_landmarks[i,0].item(),udpated_predicted_landmarks[i,1].item(),udpated_predicted_landmarks[i,2].item()))
            print("GT landmark {}: ({}, {}, {})".format(i, ground_truth_landmarks[i,0].item(),ground_truth_landmarks[i,1].item(),ground_truth_landmarks[i,2].item()))
            print()
        print()
    return models['model'], loss_c.item(), loss_r.item(), loss.item(), bs