import numpy as np
import argparse
import os

from utils import autoencoder, input_data
from viz import support

#Pytorch
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Argparse')

parser.add_argument('--landmark_count', type=int, default=2, help = "Number of landmarks")
parser.add_argument('--dimension', type=int, default=3, help = "Dimension of the image")
parser.add_argument('--landmark_unwant', type = list, default = [], help = "Unwanted landmarks")
parser.add_argument('--num_shape_params', type = int, default = 15, help = "Number of shape parameters (compression size)")

parser.add_argument('--epochs', type=int, default=100000, help = "Number of epochs to train")
parser.add_argument('--batch_size', type=int, default=2, help = "Batch size")
parser.add_argument('--learning_rate', type=float, default=0.005, help = "Learning rate")
parser.add_argument('--save_viz', type=bool, default=True, help = "Whether to save the loss plot")
parser.add_argument('--is_finetune', type=bool, default=False, help = "Whether to finetune the existing model")
parser.add_argument('--device_id', type=int, default='0', help = "Device id to use")
parser.add_argument('--minimum_loss', type=int, default=100, help = "Minimum loss to mark")
parser.add_argument('--print_freq', type=int, default=40, help = "Print frequency")
parser.add_argument('--save_model', type=bool, default=True, help = "Whether to save the model")
parser.add_argument('--n_val', type=int, default=3, help = "Number of validations")

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
    
    landmark_count = args.landmark_count
    dimension = args.dimension
    landmark_count = args.landmark_count
    num_shape_params = args.num_shape_params
    landmark_unwant = args.landmark_unwant
    
    lr = args.learning_rate
    minimum_loss= args.minimum_loss
    batch_size = args.batch_size
    epochs = args.epochs
    is_finetune = args.is_finetune
    save_viz = args.save_viz
    print_freq = args.print_freq
    pretrained_model_dir = './ckpt/models/'
    device = torch.device("cuda:{}".format(args.device_id)) if torch.cuda.is_available() else torch.device("cpu")
    save_model = args.save_model
    n_val = args.n_val

#### Custom Datasets ####

class Dataset_CTP_AE(torch.utils.data.Dataset):
    def __init__(self, labels):
        self.labels = labels
        self.transforms = transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.transforms(self.labels[idx])

def train_epoch(epoch_idx, train_dataloader, model, criterion, optimizer, config):
    save_loss = []
    model.train()
    for batch_idx, labels in enumerate(train_dataloader):
        labels = labels.float().reshape(-1,config.landmark_count*3).to(config.device)
        output_e, output_d = model(labels)
        optimizer.zero_grad()
        loss = criterion(labels, output_d)
        save_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    
    return np.mean(save_loss)

def validate_epoch(epoch_idx, val_dataloader, model, criterion, config):
    save_val_loss = []
    model.eval()
    with torch.no_grad():
        for batch_idx, labels in enumerate(val_dataloader):
            labels = labels.float().reshape(-1,config.landmark_count*3).to(config.device)
            output_e, output_d = model(labels)
            loss = criterion(labels, output_d)
            save_val_loss.append(loss.item())
            
    return np.mean(save_val_loss)

def reload_model(model, config):
    model = autoencoder.load_model(config)
    model.to(config.device)
    return model

def main():
    config = Config()
    
    print("\n\n\n\n\n\n\n\n\n\n")
    print("================[Starting training shape model]================")
    print("\n\nLoading shape model(=autoencoder)... ")
    model = autoencoder.load_model(config)
    model.to(config.device)
    print(">>successful!")
    print("\n\nLoading data...")
    train_dataset, test_dataset = input_data.read_data_sets(config.data_dir, config.label_dir, config.train_list_file, config.test_list_file, config.dimension, config.landmark_count, config.landmark_unwant)
    print(f"Number of training images: {len(train_dataset.images)}")
    print(">>successful!")
    
    print("\n\nConstructing Pytorch Dataset & Dataloader...")
    train_dataset_CTP = Dataset_CTP_AE(train_dataset.labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset_CTP, config.batch_size, shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(train_dataset_CTP, 1, shuffle = True)
    
    print(">>successful!")

    print("Constructing criterion and optimizer...")
    criterion = nn.MSELoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    print(">>>successful!")
    
    if config.is_finetune == True:
        print("\n\n Starting training from an existing model...")
        model.load_state_dict(torch.load(config.pretrained_model_dir)['model_w'])
        optimizer.load_state_dict(torch.load(config.pretrained_model_dir)['optimizer_w'])

    print("\n\n Starting training from scratch...")

    save_loss = []
    best_loss = config.minimum_loss
    n_prints = 0
    max_patience = 3
    cur_patience = 0
    
    for epoch_idx in range(config.epochs):
        loss = train_epoch(epoch_idx, train_dataloader, model, criterion, optimizer, config)
        
        if epoch_idx%config.print_freq == 0:
            save_loss.append(loss)
            val_loss_ = []
            for _ in range(config.n_val):
                val_loss_.append(validate_epoch(epoch_idx, val_dataloader, model, criterion, config))
            val_loss = np.mean(val_loss_)
            val_std= np.std(val_loss_)
            
            print("epoch: {}(/{}), train loss: {} val loss: {} (std: {})".format(epoch_idx, config.epochs, loss, val_loss, np.round(val_std,3)))
            n_prints +=1
            
            if n_prints > 2 and loss > 800 and cur_patience > max_patience:
                model = reload_model(model, config)
                model.to(config.device)
                optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
                epoch_idx = 0
                n_prints = 0
                print("reloading model...\n\n")
                cur_patience = 0
                continue
            
            elif loss < 500:
                cur_patience = 0
            else:
                cur_patience += 1 
                
            if val_loss < best_loss:
                best_loss = val_loss
                print("best val loss so far: {}".format(val_loss))
                
                if config.save_model == True and epoch_idx > 400:
                    print("Saving model...\n\n")
                    hist = dict()
                    hist['epoch_idx'] = epoch_idx
                    hist['train_loss'] = loss
                    hist['val_loss'] = val_loss
                    hist['model_w'] = model.state_dict()
                    hist['optimizer_w'] = optimizer.state_dict()
                    loss_str = str(val_loss)[:4].replace(".","_")
                    torch.save(hist, f"./ckpt/models/history{loss_str}.pt")

    if config.save_viz:
        support.save_loss_plot(save_loss)
        

if __name__ == '__main__':
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    main()
    print("All done!")