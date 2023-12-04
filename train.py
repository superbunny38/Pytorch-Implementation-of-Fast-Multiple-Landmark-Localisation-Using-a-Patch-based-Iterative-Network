#ref: https://github.com/yuanwei1989/landmark-detection/blob/
import os
import numpy as np
import argparse
from utils import autoencoder, input_data, network
from viz import support
global args
from tqdm.notebook import tqdm
import glob

### Pytorch
import torch
import torch.nn as nn
import torch.optim


parser = argparse.ArgumentParser(description='Argparse')

#PIN
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--alpha', type=float, default=0.5, help='Weighting given to the loss')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Baseline learning rate')
parser.add_argument('--box_size', type=float, default=101, help='Patch size (should be odd number)')
parser.add_argument('--landmark_count', type=int, default=2, help='Number of landmark points')
parser.add_argument('--drop_out', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--print_config', type=bool, default=False, help='Whether to print out the configuration')
parser.add_argument('--dimension', type=int, default=3, help='Dimensionality of the input data')
parser.add_argument('--device_id', type=int, default=0, help='Device ID')
parser.add_argument('--max_steps', type=int, default=100000, help='Maximum number of steps to train')
parser.add_argument('--write_log', type=bool, default=True, help='Whether to write the experiment log')
parser.add_argument('--get_info', type=bool, default=False, help='Whether to get the information of the code')

##Autoencoder
parser.add_argument('--learning_rate_ae', type=float, default=0.001, help='Learning rate for autoencoder')
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

def landmarks2b(landmarks, ae, config, is_train = True):
    landmarks = np.reshape(landmarks, (landmarks.shape[0], landmarks.shape[1]*landmarks.shape[2]))  # Reshape to [num_examples, 3*num_landmarks]
    landmarks = torch.from_numpy(landmarks).float().to(config.device)#convert to pytorch tensor
    # print(landmarks.size())#[num_examples, 3*num_landmarks] == [1,3*2]
    
    ae.train()
    bs_gt = ae.encoder(landmarks)
    decoded_landmarks = ae.decoder(bs_gt)
    return bs_gt, decoded_landmarks

def get_train_pairs(batch_size, images, labels, config, num_actions, num_regression_outputs, models, is_train = True):
    img_count = len(images)
    if config.landmark_count > 3:
        bs_gt, decoded_landmarks = landmarks2b(labels, models['shape_model'], config, is_train)
    else:#num_landmarks가 3개 이하면 compression 필요 없어보임
        bs_gt = labels
        decoded_landmarks = labels
        
    num_landmarks = config.landmark_count
    box_r = int((config.box_size-1)/2)
    patches = np.zeros((batch_size, config.box_size, config.box_size, int(3*num_landmarks)), dtype=np.float32)
    actions_ind = np.zeros((batch_size, num_actions), dtype=np.float32)
    actions = np.zeros((batch_size, num_actions, np.float32))
    
    #get image indices randomly for a mini-batch
    ind = np.random.randint(img_count, size = batch_size)
    
    return

def train_pairs(patches_train, actions_train, dbs_train, config, models):
    """
    Args:
        patches_train: x (input)
        actions_train: classification labels
        dbs_train: regression labels
        config: fixed parameters
        models: models
    """
    
    models['model'].train()
    patches_train = torch.from_numpy(patches_train).float().to(config.device)
    yc, yr = models['model'](patches_train, actions_train, dbs_train)
    

def main():
    config = Config()
    
    if args.print_config:
        support.print_config(config)
    if args.get_info:
        support.print_info(config)
        
    print("\n\n\n\n\n\n\n\n\n\n")
    print("================[Starting training]================")
    print("\n\nLoading shape model(=autoencoder) and PIN... ")
    
    num_cnn_output_c, num_cnn_output_r = 2*config.landmark_count*config.dimension, config.landmark_count*config.dimension
    models = dict()
    if config.landmark_count > 3:
        shape_model = autoencoder.load_model(config.landmark_count*3, config.device)
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
    print(">>successful!")
    
    print("\n\nLoading Loss and optimizers for shape model and PIN... ")
    #Define Loss for training
    criterions = dict()
    criterions['cls'] = nn.CrossEntropyLoss()
    criterions['reg'] = nn.MSELoss()
    
    if config.landmark_count > 3:
        criterions['autoencoder'] = nn.BCELoss()
        #Define Loss for autoencoder
    
    #Define Optimizer
    optimizers = dict()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
    optimizers['optimizer'] = optimizer
    if config.landmark_count > 3:
        optimizer_autoencoder = torch.optim.Adam(shape_model.parameters(), lr = config.learning_rate_ae)
        optimizers['optimizer_autoencoder'] = optimizer_autoencoder
    print(">>successful!")
    
    print("\n\nTraining pairs...")
    for step_i in tqdm(range(config.max_steps), desc='Training...'):
        #generate training pairs via patch extraction
        get_train_pairs(config.batch_size,
                        train_dataset.images,
                        train_dataset.labels,
                        config,
                        num_cnn_output_c,
                        num_cnn_output_r,
                        models)
        
        #train the model with the generated training pairs
        train_pairs()
    
    #모든 타임프레임에 대해서 input을 받은 후에 최종 Loss에 도입해야 할듯
    #ex.) cord_1 = model(x), cord_2 = model(x),..., cord_30 = model(x)
    #Loss = loss(cord_1, cord_2,..., cord_30)


if __name__ == '__main__':
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    main()
    print("Successfully finished!")