#ref: https://github.com/yuanwei1989/landmark-detection/blob/
import os
import numpy as np
import argparse
from utils import shape_model_func
global args

parser = argparse.ArgumentParser(description='Argparse')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--alpha', type=float, default=0.5, help='Weighting given to the loss')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Baseline learning rate')
parser.add_argument('--box_size', type=float, default=101, help='Patch size (should be odd number)')
parser.add_argument('--landmark_count', type=int, default=2, help='Number of landmark points')
parser.add_argument('--print_config', type=bool, default=False, help='Whether to print out the configuration')
args = parser.parse_args()

                    
class Config(object):
    """Training configurations."""
    # File paths
    data_dir = './data/Images'
    label_dir = './data/Landmarks'
    train_list_file = './data/list_train.txt'
    test_list_file = './data/list_test.txt'
    log_dir = './logs'
    model_dir = './cnn_model'
    # Shape model parameters
    shape_model_file = './shape_model/shape_model/ShapeModel.mat'
    eigvec_per = 0.995      # Percentage of eigenvectors to keep
    sd = 3.0                # Standard deviation of shape parameters
    landmark_count = args.landmark_count     # Number of landmarks
    landmark_unwant = [0, 8, 9, 13, 14, 15]     # list of unwanted landmark indices
    # Training parameters
    resume = False          # Whether to train from scratch or resume previous training
    box_size = args.box_size          # patch size (odd number)
    alpha = args.alpha             # Weighting given to the loss (0<=alpha<=1). loss = alpha*loss_c + (1-alpha)*loss_r
    learning_rate = 0.001
    max_steps = 100000      # Number of steps to train
    save_interval = 25000   # Number of steps in between saving each model
    batch_size = 64         # Training batch size
    dropout = 0.5

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
    
    print("\n=======Training parameters=======")
    print("resume: {}".format(config.resume))
    print("box_size (patch size (odd number)): {}".format(config.box_size))
    print("alpha: {}".format(config.alpha))
    print("learning_rate: {}".format(config.learning_rate))
    print("max_steps: {}".format(config.max_steps))
    print("save_interval: {}".format(config.save_interval))
    print("batch_size: {}".format(config.batch_size))
    print("dropout: {}".format(config.dropout))
    print("=====================================\n\n\n\n")
    

def main():
    config = Config()
    if args.print_config:
        print_config(config)
    print("\n\n\n\n\n\n\n\n\n\n")
    print("================[Starting training]================")
    print("Loading shape model...")
    shape_model = shape_model_func.load_shape_model(config.shape_model_file, config.eigvec_per)
    
    
if __name__ == '__main__':
    main()