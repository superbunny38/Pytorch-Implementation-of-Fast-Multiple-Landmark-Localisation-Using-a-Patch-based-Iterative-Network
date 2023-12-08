#Code for training one step

### Pytorch
import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
import numpy as np


def train_pairs(step_i, patches_train, yc_, yr_, config, models, criterions, optimizers, bs):
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
    # print("patches_train shape: {}".format(patches_train.size()))
    
    yc_, yr_ = torch.from_numpy(yc_).float().to(config.device), torch.from_numpy(yr_).float().to(config.device)
    yc, yr = models['model'](patches_train)
    yc = nn.Softmax(dim=1)(yc)
    
    # print(f"\n\n\nclassification label: {yc_[0]}, classifcation prediction: {yc[0]}")
    # print(f"\nregression label: {yr_[0]}, regression prediction: {yr[0]}")
    
    ### Calculate Loss
    loss_c = criterions['cls'](yc_,logits = yc)
    loss_r = criterions['reg'](yr_,yr)
    loss = config.alpha*loss_c + (1-config.alpha)*loss_r
    
    if config.landmark_count > 3:
        print("\nSorry, not implemented yet")
        NotImplementedError()
        
    optimizers['optimizer'].zero_grad()
    loss.backward()
    optimizers['optimizer'].step()
    
    #get predictions
    
    #cls loss, reg loss, total loss   
    
    action_ind = torch.argmax(yc)
    action_prob = nn.Softmax()(yc)
    
    yr = yr.detach().cpu().numpy()
    yr_ = yr_.detach().cpu().numpy()
    yc = yc.detach().cpu().numpy()
    yc_ = yc_.detach().cpu().numpy()
    
    action_prob = np.exp(yc - np.expand_dims(np.amax(yc, axis=1), 1))
    action_prob = action_prob / np.expand_dims(np.sum(action_prob, axis=1), 1)  # action_prob=[num_examples, 2*num_shape_params]
    
    bs = bs - yr*np.amax(np.reshape(action_prob,(bs.shape[0], bs.shape[1],2)),axis = 2)
    
    udpated_predicted_landmarks = bs.reshape((config.batch_size,config.landmark_count, -1))
    udpated_predicted_landmarks = np.mean(udpated_predicted_landmarks,axis = 0)
    assert udpated_predicted_landmarks.shape == (config.landmark_count, 3)
    
    if step_i%config.print_freq == 0:
        print("========Is regression right????========")
        print("predicted dbs: {}".format(yr[:4]))
        print("GT dbs: {}".format(yr_[:4]))
        print("action_ind:",action_ind.cpu().detach().numpy())
        print("gt cls:",np.argmax(yc_))
        for i in range(config.landmark_count):
            print("predicted landmark {}: {}".format(i, udpated_predicted_landmarks[i,:]))
            print("GT landmark {}: {}".format(i, config.gt_label_cord[0][i]))
        print()
    return models['model'], loss_c.item(), loss_r.item(), loss.item(), bs
