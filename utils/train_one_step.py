#Code for training one step

### Pytorch
import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms


def train_pairs(patches_train, yc_, yr_, config, models, criterions, optimizers):
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
    loss_c = criterions['cls'](yc_,yc)
    loss_r = criterions['reg'](yr_,yr)
    loss = config.alpha*loss_c + (1-config.alpha)*loss_r
    
    if config.landmark_count > 3:
        print("\nSorry, not implemented yet")
        NotImplementedError()
        
    optimizers['optimizer'].zero_grad()
    loss.backward()
    optimizers['optimizer'].step()
    
    #cls loss, reg loss, total loss   
    return loss_c.item(), loss_r.item(), loss.item()
