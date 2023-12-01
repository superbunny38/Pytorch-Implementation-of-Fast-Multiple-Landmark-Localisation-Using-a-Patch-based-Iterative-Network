#ref: https://github.com/yuanwei1989/landmark-detection/blob/
import os
import numpy as np
import argparse
from utils import autoencoder, input_data, network
global args
from tqdm.notebook import tqdm
import glob

### Pytorch
import torch
import torch.nn as nn
import torch.optim


parser = argparse.ArgumentParser(description='Argparse')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--alpha', type=float, default=0.5, help='Weighting given to the loss')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Baseline learning rate')
parser.add_argument('--box_size', type=float, default=101, help='Patch size (should be odd number)')
parser.add_argument('--landmark_count', type=int, default=2, help='Number of landmark points')
parser.add_argument('--drop_out', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--print_config', type=bool, default=False, help='Whether to print out the configuration')
parser.add_argument('--dimension', type=int, default=3, help='Dimensionality of the input data')
parser.add_argument('--device_id',type=int, default=0, help='Device ID')
args = parser.parse_args()

                    
class Config(object):
    """Training configurations."""
    # File paths
    data_dir = './data/Images'
    label_dir = './data/Landmarks'
    train_list_file = './data/list_train.txt'
    assert os.path.exists(train_list_file) == True, print("train_list_file does not exist")    
    test_list_file = './data/list_test.txt'
    assert os.path.exists(test_list_file) == True, print("test_list_file does not exist")
    log_dir = './logs'
    model_dir = './cnn_model'
    # Shape model parameters
    shape_model_file = './shape_model/shape_model/ShapeModel.mat'
    eigvec_per = 0.995      # Percentage of eigenvectors to keep
    sd = 3.0                # Standard deviation of shape parameters
    landmark_count = args.landmark_count     # Number of landmarks
    landmark_unwant = []#[0, 8, 9, 13, 14, 15]     # list of unwanted landmark indices
    # Training parameters
    resume = False          # Whether to train from scratch or resume previous training
    box_size = args.box_size          # patch size (odd number)
    alpha = args.alpha             # Weighting given to the loss (0<=alpha<=1). loss = alpha*loss_c + (1-alpha)*loss_r
    learning_rate = args.learning_rate  # Baseline learning rate
    max_steps = 100000      # Number of steps to train
    save_interval = 25000   # Number of steps in between saving each model
    batch_size = args.batch_size        # Training batch size
    dropout = args.drop_out        # Dropout rate
    
    #Newly added by Chaeun Ryu
    dimension = args.dimension     # Dimensionality of the input data
    device = torch.device("cuda:{}".format(args.device_id)) if torch.cuda.is_available() else torch.device("cpu")

def print_config(config):
    print("\n\n\n\n========= Configuration Info. =========")
    print("\n=======File paths=======")
    print("data_dir: {}".format(config.data_dir))
    print("label_dir: {}".format(config.label_dir))
    print("train_list_file: {}".format(config.train_list_file))
    print("test_list_file: {}".format(config.test_list_file))
    print("log_dir: {}".format(config.log_dir))
    print("model_dir: {}".format(config.model_dir))
    
    print("\n=======Shape model parameters=======")
    print("shape_model_file: {}".format(config.shape_model_file))
    print("eigvec_per: {}".format(config.eigvec_per))
    print("sd: {}".format(config.sd))
    print("landmark_count: {}".format(config.landmark_count))
    print("landmark_unwant: {}".format(config.landmark_unwant))
    
    print("\n=======Experiment parameters=======")
    print("resume: {}".format(config.resume))
    print("box_size (patch size (odd number)): {}".format(config.box_size))
    print("alpha: {}".format(config.alpha))
    print("learning_rate: {}".format(config.learning_rate))
    print("max_steps: {}".format(config.max_steps))
    print("save_interval: {}".format(config.save_interval))
    print("batch_size: {}".format(config.batch_size))
    print("dropout: {}".format(config.dropout))
    print("running on device: {}".format(config.device))
    print("=====================================\n\n\n\n")
    
    

def train_epoch(epoch, train_loader, model, criterions, optimizer, device, config):
    cls_loss = []
    reg_loss = []
    model.to(device)
    model.train()
    
    for image, label in train_loader:
        image, label = image.to(device), label.to(device)
        cls_out, reg_out = model(image)
        
    
    return np.mean(cls_loss), np.mean(reg_loss)

def train(num_epochs,train_loader, model, criterions, optimizer, device, config):
    save_train_loss = {"cls":[], "reg":[]}
    save_val_loss = {"cls":[], "reg":[]}
    
    for epoch in tqdm(range(num_epochs)):
        train_loss_cls, train_loss_reg = train_epoch(epoch, train_loader, model, criterion, optimizer, device)
    

def main():
    config = Config()
    if args.print_config:
        print_config(config)
    print("\n\n\n\n\n\n\n\n\n\n")
    print("================[Starting training]================")
    print("\n\nLoading shape model(=conv autoencoder) and PIN... ")
    
    num_cnn_output_c, num_cnn_output_r = 2*config.landmark_count*config.dimension, config.landmark_count*config.dimension
    shape_model = autoencoder.load_model(config.device)
    model = network.cnn(num_cnn_output_c, num_cnn_output_r)
    model.to(config.device)
    shape_model.to(config.device)
    # print("landmark_count: {}".format(config.landmark_count))
    print(">>successful!")
    print("\n\nLoading data...")
    data = input_data.read_data_sets(config.data_dir, config.label_dir, config.train_list_file, config.test_list_file, config.dimension, config.landmark_count, config.landmark_unwant)
    print(">>successful!")
    
    #Define Loss for training
    criterions = dict()
    criterions['cls'] = nn.CrossEntropyLoss()
    criterions['reg'] = nn.MSELoss()
    
    #Define Loss for autoencoder
    
    #Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
    oprimizer_autoencoder = torch.optim.Adam(shape_model.parameters(), lr = config.learning_rate)
    
    
    #모든 타임프레임에 대해서 input을 받은 후에 최종 Loss에 도입해야 할듯
    #ex.) cord_1 = model(x), cord_2 = model(x),..., cord_30 = model(x)
    #Loss = loss(cord_1, cord_2,..., cord_30)
    
if __name__ == '__main__':
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    main()