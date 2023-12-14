#ref: https://github.com/yuanwei1989/landmark-detection/blob/
import os
import numpy as np
import argparse
from utils import input_data, network, patch, train_one_step
from viz import support
global args
from tqdm import tqdm
import glob

#to track time, time functions
from datetime import datetime
from datetime import timedelta
import time

### Pytorch
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Argparse')

#PIN
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--alpha', type=float, default=0.2, help='Weighting given to the loss (weight on cls)')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Baseline learning rate')
parser.add_argument('--box_size', type=float, default=101, help='Patch size (should be odd number)')
parser.add_argument('--landmark_count', type=int, default=2, help='Number of landmark points')
parser.add_argument('--drop_out', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--print_config', type=bool, default=False, help='Whether to print out the configuration')
parser.add_argument('--dimension', type=int, default=3, help='Dimensionality of the input data')
parser.add_argument('--device_id', type=int, default=0, help='Device ID')
parser.add_argument('--max_steps', type=int, default=100000, help='Maximum number of steps to train')
parser.add_argument('--write_log', type=bool, default=True, help='Whether to write the experiment log')
parser.add_argument('--get_info', type=bool, default=False, help='Whether to get the information of the code')
parser.add_argument('--save_viz', type=bool, default=True, help='Whether to save the visualization')
parser.add_argument('--print_freq', type=int, default=1000, help='How often to print out the loss')
parser.add_argument('--save_model', type =bool, default=True, help='Whether to save the trained model')
parser.add_argument('--save_log',type=bool, default=False, help='Whether to save the experiment log')
parser.add_argument('--reg_loss_type', type=str, default='mse', help='The type of regression loss')
parser.add_argument('--backbone_resnet',type = bool, default = False, help='Whether to use resnet backbone (Takes a lot of time)')
parser.add_argument('--is_ctp',type = bool, default = True, help='Whether to use CTP dataset')

##Autoencoder
parser.add_argument('--learning_rate_ae', type=float, default=0.001, help='Learning rate for autoencoder')
args = parser.parse_args()

                    
class Config(object):
    """Training configurations."""
    # File paths
    data_dir = './data/Images_CTP'
    label_dir = './data/Landmarks_CTP'
    train_list_file = './data/list_train_CTP.txt'
    assert os.path.exists(train_list_file) == True, print("train_list_file does not exist")    
    test_list_file = './data/list_test_CTP.txt'
    assert os.path.exists(test_list_file) == True, print("test_list_file does not exist")
    log_dir = './logs'
    save_model_dir = './ckpt/models/'
    # Shape model parameters
    shape_model_file = './shape_model/shape_model/ShapeModel.mat'
    eigvec_per = 0.995      # Percentage of eigenvectors to keep
    sd = 3.0                # Standard deviation of shape parameters
    landmark_count = args.landmark_count     # Number of landmarks
    landmark_unwant = []#[0, 8, 9, 13, 14, 15]     # list of unwanted landmark indices
    # Training parameters
    resume = False          # Whether to train from scratch or resume previous training
    box_size = args.box_size          # patch size (odd number)
    assert box_size % 2 == 1, print("box_size should be odd number")
    alpha = args.alpha             # Weighting given to the loss (0<=alpha<=1). loss = alpha*loss_c + (1-alpha)*loss_r
    learning_rate = args.learning_rate  # Baseline learning rate
    max_steps = args.max_steps      # Number of steps to train
    save_interval = 25000   # Number of steps in between saving each model
    batch_size = args.batch_size        # Training batch size
    dropout = args.drop_out        # Dropout rate
    
    #Newly added by Chaeun Ryu
    dimension = args.dimension     # Dimensionality of the input data
    device = torch.device("cuda:{}".format(args.device_id)) if torch.cuda.is_available() else torch.device("cpu")
    write_log = args.write_log
    learning_rate_ae = args.learning_rate_ae
    get_info = args.get_info
    save_viz = args.save_viz
    print_freq = args.print_freq
    save_model = args.save_model
    save_log = args.save_log
    reg_loss_type = args.reg_loss_type
    backbone_resnet = args.backbone_resnet
    is_ctp = args.is_ctp

def get_train_pairs(step_i, batch_size, images, labels, config, num_actions, num_regression_output, models, bs, random_init = True):
    '''
    Args:
        batch_size: Number of examples in a mini-batch
        images: List of images (n_images, 512, 512, 23, 1) for ctp dataset
        labels: List of labels (=landmarks)
        config: Training configurations
        num_actions: Number of classification outputs
        num_regression_output: Number of regression outputs
        sd: standard deviation, bounds from which to sample bs
        
    
    Returns:
        patches: 2D image patches, [batch_size, box_size, box_size, 3*num_landmarks]
        actions: Ground truth classification outputs, [batch_size, num_actions] each row is one hot vector [positive or negative for each shape parameter]
        dbs: Ground truth regression output. [batch_size, num_regression_output]. dbs = bs - bs_gt.
        bs: sampled shape parameters [batch_size, num_regression_output]
        
    '''
    img_count = len(images)
    # print("number of images: {}".format(img_count))
    # print("initial")
    # print(np.array(images).shape)
    # print("labels:",labels)
    # print("labels shape: {}".format(labels.shape)) <- 맞음
    

    # else:#num_landmarks가 3개 이하면 compression 필요 없어보임
    bs_gt = labels.reshape(labels.shape[0],-1)
    decoded_landmarks = labels
    landmarks = labels
        
    num_landmarks = config.landmark_count
    box_r = int((config.box_size-1)/2)
    patches = np.zeros((batch_size, config.box_size, config.box_size, int(3*num_landmarks)), dtype=np.float32)
    actions_ind = np.zeros((batch_size), dtype=np.float32)
    actions = np.zeros((batch_size, num_actions), np.float32)
    
    #get image indices randomly for a mini-batch
    ind = np.random.randint(img_count, size = batch_size)
    # print("ind: {}".format(ind))
    
    #Randomly sampled x from V
    #dGT = xGT - x
    if step_i == 0 or random_init:
        bounds = config.sd*np.sqrt(config.landmark_count)
        bs = np.random.rand(config.batch_size, num_regression_output) * 2*bounds - bounds
        
    #Extract image patch
    # print("image size: {}".format(np.array(images).shape))
    for i in range(config.batch_size):
        # print("ind[i]: {}".format(ind[i]))
        # print("len(patches): {}".format(len(patches)))
        # print("len(landmarks): {}".format(len(landmarks)))
        image = images[ind[i]]
        patches[i] = patch.extract_patch_all_landmarks(image, landmarks[ind[i]], box_r)# <- 원래 코드: patches[i] = patch.extract_patch_all_landmarks(image, landmarks[i], box_r)
    
    #Regression values (distances between predicted and GT)    
    dbs = bs - bs_gt
    assert dbs.shape[1] == config.landmark_count *3, print("wrong shape for regression output/label!")

    #Extract classification labels as a one-hot vector
    max_db_ind = np.argmax(np.abs(dbs), axis = 1)
    max_db = dbs[np.arange(dbs.shape[0]), max_db_ind]
    is_positive = (max_db > 0.5)
    actions_ind[is_positive] = max_db_ind[is_positive]*2
    actions_ind[np.logical_not(is_positive)] = max_db_ind[np.logical_not(is_positive)]*2 + 1
    # print("actions shape:", actions.shape)
    # print("actions_ind:", actions_ind)
    # print("actions_ind shape: {}".format(actions_ind.shape))
    actions_ind = actions_ind.astype(int)
    # print(actions_ind.dtype)
    # print("actions:", actions)
    # print(actions_ind.dtype)
    # print(np.arange(config.batch_size))
    actions[np.ix_(np.arange(config.batch_size), actions_ind)] = 1#original code: actions = actions[np.arange(config.batch_size), actions_ind] = 1
    assert actions.shape[1] == config.landmark_count*3*2, print("wrong shape for classification output/label!")
    
    #Shape assertions
    assert patches.shape[0] == config.batch_size, print("wrong shape of patches (1st dim)")
    assert patches.shape[1] == config.box_size, print("wrong shape of patches (2nd dim)")
    assert dbs.shape[0] == config.batch_size, print("wrong shape of dbs (1st dim)")
    assert dbs.shape[1] == num_regression_output, print("wrong shape of db (2nd dim)")
    return patches, actions, dbs, bs, bs_gt

def main():
    config = Config()
    
    #ref: https://gist.github.com/EdisonLeeeee/f67205683603f9c11b2940c71557410b
    def softmax_cross_entropy_with_logits(labels, logits, dim=-1,config = config):
        return torch.mean((-labels * F.log_softmax(logits, dim=dim)).sum(dim=dim)).to(config.device) 
    
    if args.print_config:
        support.print_config_train(config)
    if args.get_info:
        support.print_info(config)
    
    bs = None
    print("\n\n\n\n\n\n\n\n\n\n")
    print("================[Starting training]================")
    print("\n\nLoading shape model(=autoencoder) and PIN... ")
    start_time = time.time()
    num_cnn_output_c, num_cnn_output_r = 2*config.landmark_count*config.dimension, config.landmark_count*config.dimension
    # print("num_cnn_output_c: {}, num_cnn_output_r: {}".format(num_cnn_output_c, num_cnn_output_r))
    #num_cnn_output_c: 3(axis)x2(pos/neg)xlandmark_count
    #num_cnn_output_r: 3(axis)xlandmark_count
    models = dict()
    if config.landmark_count > 3:
        shape_model = autoencoder.load_model(config.landmark_count*3, config.device)
    if config.backbone_resnet:
        model = network.ResNet18(num_cnn_output_c, num_cnn_output_r)
    else:
        model = network.cnn(num_cnn_output_c, num_cnn_output_r)
    model.to(config.device)
    if config.landmark_count > 3:
        shape_model.to(config.device)
        models['shape_model'] = shape_model
    models['model'] = model
    
    # print("landmark_count: {}".format(config.landmark_count))
    print(">>successful!")
    print("\n\nLoading data...")
    train_dataset, test_dataset = input_data.read_data_sets(config.data_dir, config.label_dir, config.train_list_file, config.test_list_file, config.dimension, config.landmark_count, config.landmark_unwant)
    # print(train_dataset.images[0].shape)
    print(f"Number of training images: {len(train_dataset.images)}")
    print(">>successful!")
    
    config.gt_label_cord = train_dataset.labels
    
    support.patch_support(train_dataset.images, config.box_size)
    support.patch_support(test_dataset.images, config.box_size)
    
    print("\n\nLoading Loss and optimizers for shape model and PIN... ")
    #Define Loss for training
    criterions = dict()
    criterions['cls'] = softmax_cross_entropy_with_logits
    if config.reg_loss_type == 'mse':
        criterions['reg'] = nn.MSELoss().to(config.device)
    
    if config.landmark_count > 3:
        criterions['autoencoder'] = nn.BCELoss()
        #Define Loss for autoencoder
    
    #Define Optimizer
    optimizers = dict()
    optimizer = torch.optim.Adam(models['model'].parameters(), lr = config.learning_rate)
    optimizers['optimizer'] = optimizer
    
    if config.landmark_count > 3:
        optimizer_autoencoder = torch.optim.Adam(shape_model.parameters(), lr = config.learning_rate_ae)
        optimizers['optimizer_autoencoder'] = optimizer_autoencoder
    print(">>successful!")
    
    print("\n\nTraining pairs...")
    save_loss_c = []
    save_loss_r = []
    save_loss_d = []
    save_loss = []
    
    for step_i in tqdm(range(config.max_steps), desc='Training... (Patch extraction -> Train pairs)'):
        #generate training pairs via patch extraction
        patches, actions, dbs, bs, bs_gt = get_train_pairs(step_i,config.batch_size,
                                                    train_dataset.images,
                                                    train_dataset.labels,
                                                    config,
                                                    num_cnn_output_c,
                                                    num_cnn_output_r,
                                                    models, bs)
        
        #train the model with the generated training pairs
        #params: patches_train, actions_train, dbs_train, config, models
        models['model'], loss_c, loss_r, loss, bs, loss_d = train_one_step.train_pairs(step_i, patches, actions, dbs, config, models, criterions, optimizers, bs, bs_gt)
        
        if step_i%config.print_freq == 0:
            print("step_i: {} || loss_c: {}, loss_r: {}, loss_d:{}, total loss: {}".format(step_i, loss_c, loss_r, loss_d, loss))
            save_loss_c.append(loss_c)
            save_loss_r.append(loss_r)
            save_loss_d.append(loss_d)
            save_loss.append(loss)
            
    
    #모든 타임프레임에 대해서 input을 받은 후에 최종 Loss에 도입해야 할듯
    #ex.) cord_1 = model(x), cord_2 = model(x),..., cord_30 = model(x)
    #Loss = loss(cord_1, cord_2,..., cord_30)
    losses = {"save_loss_c": save_loss_c, "save_loss_r": save_loss_r, "save_loss_d":save_loss_d, "save_loss": save_loss}
    elapsed_time = time.time() - start_time
    print("Finished Training! Elapsed time:",str(timedelta(seconds= elapsed_time)))
    
    
    ### Saving the results of the training ###
    dt_string = datetime.now().strftime("%d_%m_%H_%M_%S").replace('/','.')
    if config.save_viz:
        support.save_loss_plot(losses)
        support.write_loss(losses, "./ckpt/logs/",dt_string)
    if config.save_log:
        save_log_dict = dict()
        save_log_dict['elapsed_time'] = elapsed_time
    if config.save_model:
        save_dict = dict()
        save_dict['model'] = models['model'].state_dict()
        save_dict['optimizer'] = optimizers['optimizer'].state_dict()
        torch.save(save_dict, config.save_model_dir + dt_string + "_model.pth")
    

if __name__ == '__main__':
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    main()
    print("All done!")