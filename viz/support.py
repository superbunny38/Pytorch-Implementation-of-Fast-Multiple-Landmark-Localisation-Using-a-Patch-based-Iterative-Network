from matplotlib import pyplot as plt
import pandas as pd
import torch
import glob
import os


def print_config_train(config):
    print("\n\n\n\n========= Configuration Info. =========")
    print("\n=======File paths=======")
    print("data_dir: {}".format(config.data_dir))
    print("label_dir: {}".format(config.label_dir))
    print("train_list_file: {}".format(config.train_list_file))
    print("test_list_file: {}".format(config.test_list_file))
    print("log_dir: {}".format(config.log_dir))
    print("saving model to dir: {}".format(config.save_model_dir))
    
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
    print("Regression Loss: {}".format(config.reg_loss_type))
    print("=====================================\n\n\n\n")
    


def print_info(config):
    print("\n\n================================")
    print("Official repository:https://github.com/yuanwei1989/landmark-detection")
    print("Official arxiv: https://arxiv.org/abs/1806.06987v2")
    print("Official paper: https://arxiv.org/pdf/1806.06987.pdf ")
    
    print()
    print()
    print("Writer of this repo: Chaeeun Ryu")
    print("Modifications made compared to the original repo:")
    print("1. Replaced the shape model (i.e., PCA) with the autoencoder")
    print("2. Wrote code in Pytorch.")
    print("3. It is set to compress the landmarks representation only if the number of landmarks is greater than 3.")
    print()
    print("For detailed information about the library versions, please refer to requirements.txt file")
    print("================================")
    

def save_loss_plot_(save_losses, trace_from = 10):
    plt.title(loss_type)
    for loss_type, loss_list in save_losses.items():
        
        plt.plot(loss_list[trace_from:])    
        # plt.show()
        plt.legend()
    plt.savefig("trace_loss.png")
    
    
from matplotlib.pyplot import cm
import numpy as np
def save_loss_plot(save_losses, trace_from = 5):
    color=iter(cm.rainbow(np.linspace(0,1,len(save_losses))))
    plt.title("Train Losses")
    for key, c in zip(save_losses, color):
        plt.plot(save_losses[key], c=c, label = key)
        # for idx, item in enumerate(save_losses[key]):
        #     if idx == 0:
        #         plt.plot(item[trace_from:], c=c, label=key)
        #     else:
        #         plt.plot(item[trace_from:], c=c)

    plt.legend()
    plt.savefig("trace_loss.png")


def save_single_loss_plot(loss_list,trace_from = 5):
    plt.title("Loss")
    plt.plot(loss_list[trace_from:])
    plt.xlabel("epochs")
    plt.savefig("loss_ae.png")

def print_config_inference(config):
    print("\n\n\n\n========= Configuration Info. =========")
    print("\n=======File paths=======")
    print("data_dir: {}".format(config.data_dir))
    print("label_dir: {}".format(config.label_dir))
    print("train_list_file: {}".format(config.train_list_file))
    print("test_list_file: {}".format(config.test_list_file))
    print("model_dir: {}".format(config.model_dir))
    
    print("\n=======Shape model parameters=======")
    print("shape_model_file: {}".format(config.shape_model_file))
    print("eigvec_per: {}\n".format(config.eigvec_per))
    print("sd: {}\n".format(config.sd))
    print("landmark_count: {}\n".format(config.landmark_count))
    print("landmark_unwant: {}\n".format(config.landmark_unwant))
    
    print("\n=======Testing parameters=======")
    print("box_size (patch size (odd number)): {}".format(config.box_size))
    print("max_test_steps: {}".format(config.max_test_steps))
    print("num_random_init: {}".format(config.num_random_init))
    print("predict_mode: {}".format(config.predict_mode))
    print("running on device: {}".format(config.device))
    
    print("\n=======Visualisation parameters=======")
    print("save_viz: {}".format(config.save_viz))
    print("print_config: {}".format(config.print_config))
    
    print("\n=======Experiment parameters=======")
    print("number of landmarks: {}".format(config.landmark_count))
    print("patch size: {}".format(config.patch_size))
    print("dimension: {}".format(config.dimension))
    print("device: {}\n".format(config.device))
    print("save log: {}".format(config.save_log))
    print("=====================================\n\n\n\n")

def patch_support(images, patch_size):
    n_images = len(images)
    h,w = images[0].shape[0], images[0].shape[1]
    if patch_size < max(h,w)/3:
        print("It is recommended to enlarge the patch size for better performance.")
        
def update_rule_help(config):
    print("==============[Update Rule Info.]==================")
    print("Predict mode 0 (default)")
    print(": Hard classification. Move regressed distance only in the direction with maximum probability.\n")
    print("Predict mode 1")
    print(": Soft classification. Multiply classification probabilities with regressed distances.\n")
    print("Predict mode 2")
    print(": Regression only.\n")
    print("Predict mode 3")
    print(": Classification only\n")
    print("====================================================\n\n\n")

def save_as_pt(dictionary, filename):
    """Save dictionary containing log as pt file.

    Args:
        dictionary: dictionary containing log 
        filename: file directory + file name to save the file
    """
    torch.save(dictionary, filename+".pt")#.to_csv(filename, index=False)
    
def get_the_latest_ckpt(ckpt_dir = "./ckpt/models"):
    list_of_files = glob.glob(ckpt_dir + "/*.pth")
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def write_loss(losses, log_dir, dt_string):
    df = pd.DataFrame.from_dict(losses)
    with open(os.path.join(log_dir, f"train_loss_{dt_string}.txt"), "a") as f:
        df_string = df.to_string()
        f.write(df_string)
        
def get_the_best_ae_ckpt(ckpt_dir = "./ckpt/models/"):
    ckpt_list = os.listdir(ckpt_dir)
    
    losses = []
    ae_list = []
    for file_name in ckpt_list:
        if 'new_history' in file_name:
            ae_list.append(file_name)
    
    for file_name in ae_list:
        loss_str = file_name.split("history")[-1].split(".")[0]
        loss_str = loss_str.replace("_",".")
        losses.append(float(loss_str))
    
    min_loss = min(losses)
    ret_file_name = "new_history" + str(min_loss).replace(".", "_") + ".pt"
    ret_ckpt_dir = os.path.join(ckpt_dir, ret_file_name)
    return ret_ckpt_dir