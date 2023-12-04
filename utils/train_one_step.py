#Code for training one step

### Pytorch
import torch
import torch.nn as nn
import torch.optim

def train_pairs(patches_train, yc_, yr_, config, models, criterions, optimizers):
    """
    Args:
        patches_train: x (input)
        actions_train: classification labels
        dbs_train: regression labels
        config: fixed parameters
        models: models
    """
    
    
    cls_loss = []
    reg_loss = []
    
    models['model'].train()
    patches_train = torch.from_numpy(patches_train).float().to(config.device)
    yc, yr = models['model'](patches_train, actions_train, dbs_train)
    
    ### Calculate Loss
    loss_c = criterions['cls'](yc_,yc)
    loss_r = criterions['reg'](yr_,yr)
    loss = config.alpha*loss_c + (1-config.alpha)*loss_r
    
    
    #cls loss, reg loss, total loss
    return
