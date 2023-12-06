import numpy as np
import argparse
from viz import support
from utils import input_data, network, patch, update
import torch
from tqdm import tqdm
import glob

#to track time, time functions
from datetime import datetime
from datetime import timedelta
import time

parser = argparse.ArgumentParser(description='Argparse')
parser.add_argument('--max_test_steps', type=int, default=10, help='Number of inference steps. (Recommended: 10 for Rule B and Rule C, 350 for Rule A)')
#single landmark localisation 할 때는 num_random_init = 19 (one at center)
parser.add_argument('--num_random_init', type=int, default=5, help='Number of random initialisations used.')
parser.add_argument('--predict_mode', type=int, default=1, help='How the new patch position is computed.')
parser.add_argument('--save_viz', type=bool, help='Whether to save visualisation.')
parser.add_argument('--print_config', type=bool, default=False, help='Whether to print out the configuration')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to run gpu on.')
parser.add_argument('--print_info', type=bool, default=False, help='Whether to print out information about this code.')
parser.add_argument('--dimension', type=int, default=3, help='Dimension of the data.')
parser.add_argument('--landmark_count', type=int, default=2, help='Number of landmarks.')
parser.add_argument('--patch_size', type=int, default=101, help='Patch size (odd), Recommended: at least 1/3 of max(height, width).')
parser.add_argument('--update_rule_help', type=bool, default=False, help='Whether get help with info about the update rule')
args = parser.parse_args()

class Config(object):
    """Inference configurations."""
    # File paths
    data_dir = './data/Images'
    label_dir = './data/Landmarks'
    train_list_file = './data/list_train.txt'
    test_list_file = './data/list_test.txt'
    model_dir = './ckpt/models/04_12_18_46_26_model.pth'
    # Shape model parameters
    shape_model_file = ''
    eigvec_per = 0.995      # Percentage of eigenvectors to keep
    sd = 3.0                # Standard deviation of shape parameters
    landmark_count = args.landmark_count     # Number of landmarks
    landmark_unwant = []     # list of unwanted landmark indices
    # Testing parameters
    patch_size = args.patch_size          # patch size (odd number)
    assert patch_size % 2 != 0, print("it is recommended that the patch size is an odd number.")
    max_test_steps = args.max_test_steps     # Number of inference steps
    num_random_init = args.num_random_init     # Number of random initialisations used
    predict_mode = args.predict_mode        # How the new patch position is computed.
                            # 0: Classification and regression. Hard classification
                            # 1: Classification and regression. Soft classification. Multiply classification probabilities with regressed distances
                            # 2: Regression only
                            # 3: Classification only
    # Visualisation parameters
    save_viz = args.save_viz           # Whether to save visualisation
    
    #Chaeeun parameters
    print_config = args.print_config
    device = args.device
    print_info = args.print_info
    dimension = args.dimension
    update_rule_help = args.update_rule_help

def landmark_initialization(landmarks, config, single_image):
    """Landmark Initialization for landmark inference.
    (One should be at the center of the image)
    
    In paper, this initialisation is supposed to set the other initial landmarks except for the center location at fixed distance of one-quarter image size around it, but I made it random.

    Args:
        landmarks: zero array of shape (num_examples=num_random_init+1, num_landmarks, 3)
        config: configuration
        single_image: single 3D image [width, height, depth, channel]
        
    """
    ### Center
    center_x, center_y, center_z = single_image.shape[0]/2, single_image.shape[1]/2, single_image.shape[2]/2
    center = []
    for i in range(config.landmark_count):
        center.append([center_x, center_y, center_z])
    landmarks[0] = center
    
    ### Random initialisations
    for i in range(config.num_random_init):
        random_x = np.random.uniform(0, single_image.shape[0])
        random_y = np.random.uniform(0, single_image.shape[1])
        random_z = np.random.uniform(0, single_image.shape[2])
        random_landmark = []
        for j in range(config.landmark_count):
            random_landmark.append([random_x, random_y, random_z])
        landmarks[i+1] = random_landmark
    
    return landmarks
    

#original repo params: predict_landmarks(images[i], config, shape_model, sess, x, action_ind, yc, yr, keep_prob)
def predict_landmarks(image, config, model):# Predict one image.
    """Predict landmarks iteratively.

    Args:
        image (_type_): _description_
        config (_type_): _description_
    """
    num_landmarks = config.landmark_count
    max_test_steps = config.max_test_steps
    patch_size = config.patch_size
    patch_r = int((patch_size - 1)/2)
    #num_examples: number of patches to extract for one image according to the landmarks
    num_examples = config.num_random_init + 1
    
    #Initialize initial landmarks with zeros because no representation compression is done by PCA
    landmarks = np.zeros((num_examples, num_landmarks, 3))
    landmarks = landmark_initialization(landmarks, config, image)#initialize (n_init+1) points (one at the volume center)

    #Extract patches from landmarks
    patches = np.zeros((num_examples, patch_size, patch_size, num_landmarks*3))
    
    
    for idx in range(num_examples):
        patches[idx] = patch.extract_patch_all_landmarks(image, landmarks[idx], patch_r)
        
    
    landmarks_all_steps = np.zeros((max_test_steps+1, num_examples, num_landmarks, 3))
    landmarks_all_steps[0] = landmarks
    
    #Find path of landmark iteratively
    for jdx in range(config.max_test_steps):
        action_ind_val, yc_val, yr_val = network.predict_cnn(patches, config, model)
        
        #Compute classification probabilities
        action_prob = np.exp(yc_val-np.expand_dims(np.amax(yc_val, axis=1), axis=1))
        action_prob = action_prob/np.expand_dims(np.sum(action_prob, axis=1), axis=1)
        
        #update landmarks
        landmarks = update.update_landmarks(landmarks, action_prob, yr_val, config)#update rule A, B, C
        
        landmarks_all_steps[jdx+1] = landmarks
        
        #Extract patches from landmarks
        for kdx in range(num_examples):
            patches[kdx] = patch.extract_patch_all_landmarks(image, landmarks[kdx], patch_r)
    
    #Compute mean of all initialisations
    landmarks_mean = np.mean(landmarks_all_steps[-1,:,:,:], axis=0)#landmarks_mean: [num_landmarks, 3]
    # print("landmarks mean shape: ", landmarks_mean.shape)
    assert landmarks_mean.shape[0] == config.landmark_count, print("wrong landmarks mean shape [0]:", landmarks_mean[0].shape)
    assert landmarks_mean.shape[1] == 3, print("wrong landmarks mean shape [1]:", landmarks_mean[1].shape)
        
    return landmarks_all_steps, landmarks_mean

def compute_err(landmarks, landmarks_gt, pix_dim):
    """Compute error between predicted landmarks and ground truth landmarks.

    Args:
        landmarks: Predicted landmarks [img_count, num_landmarks, 3].
        landmarks_gt: Ground truth landmarks. [img_count, num_landmarks, 3]
        pix_dim: Pixel spacing. [img_count, 3]
        
    Returns:
      err: distance error in voxel. [img_count, num_landmarks]
      err_mm: distance error in mm. [img_count, num_landmarks]
    """
    #Shape assertions
    assert landmarks.shape[2] == 3, print("wrong landmarks shape [2]:", landmarks[2].shape)
    assert landmarks_gt.shape[2] == 3, print("wrong landmarks_gt shape [2]:", landmarks_gt[2].shape)
    assert landmarks.shape[0] == landmarks_gt.shape[0], print("landmarks shape and landmarks_gt shape doesn't match: ", landmarks.shape, landmarks_gt.shape)
    assert landmarks.shape[1] == landmarks_gt.shape[1], print("landmarks shape and landmarks_gt shape doesn't match: ", landmarks.shape, landmarks_gt.shape)

    num_landmarks = landmarks.shape[1]
    
    
    return err, err_mm

def predict(dataset, config, model):#Predict landmarks for entire images.
    """Find the path of the landmark iteratively, and evaluate the results.

    Args:
        dataset: dataset to predict 
        config: configurations
    """

    images = dataset.images # [num_images, height, width, channels] <- pytorch version으로 바꿔야 함
    landmarks_gt = dataset.labels
    names = dataset.names
    pix_dim = dataset.pix_dim# pix_dim: mm of each voxel. [img_count, 3]
    num_landmarks = config.landmark_count
    img_count = len(images)
    max_test_steps = config.max_test_steps
    num_examples = config.num_random_init + 1
    
    landmarks_all_steps = np.zeros((img_count, max_test_steps+1, num_examples, num_landmarks,3))
    landmarks_mean = np.zeros((img_count, num_landmarks,3), dtype=np.float32)
    landmarks_mean_unscale = np.zeros((img_count, num_landmarks,3), dtype=np.float32)
    landmarks_gt_unscale = np.zeros((img_count, num_landmarks, 3), dtype=np.float32)
    images_unscale = []
    time_elapsed = np.zeros(img_count)
    
    for i in tqdm(range(img_count), desc = "Predict landmarks for all images one by one..."):
        start_time_img = time.time()
        landmarks_all_steps[i], landmarks_mean[i] = predict_landmarks(images[i], config, model)
        end_time_img = time.time()
        time_elapsed[i] = end_time_img - start_time_img
        
        
        #convert the scaling back to that of the original image
        #난 0.5를 곱한 적이 없는데...?
        landmarks_mean_unscale[i] = landmarks_mean[i]*pix_dim[i]/0.5
        landmarks_gt_unscale[i] = landmarks_gt[i]*pix_dim[i]/0.5
        images_unscale.append(scipy.ndimage.zoom(images[i][:,:,:,0],pix_dim[i]/0.5))
        
        print(f"Time elapsed for image {i}/{img_count}: {names[i]} {time_elapsed[i]:.2f} s")
            
    #Time
    time_elapsed_mean = np.mean(time_elapsed)
    print("Mean running time = {:.10f}s\n".format(time_elapsed_mean))

    #Evaluate distance error
    err, err_mm = compute_err(landmarks_mean, landmarks_gt, pix_dim)
    
def main():
    config = Config()
    
    if args.print_config:
        support.print_config_inference(config)
    if args.print_info:
        support.print_info(config)
    if args.update_rule_help:
        support.update_rule_help(config)
    
    print("\n\n\n\n\n\n\n\n\n\n")
    print("================[Starting Inference]================")
    start_time = time.time()
    num_cnn_output_c, num_cnn_output_r = 2*config.landmark_count*config.dimension, config.landmark_count*config.dimension
    print("\n\nLoading data...")
    _, test_dataset = input_data.read_data_sets(config.data_dir, config.label_dir, config.train_list_file, config.test_list_file, config.dimension, config.landmark_count, config.landmark_unwant)
    print(">>successful!")
    
    support.patch_support(test_dataset.images, config.patch_size)
    
    print(f"\n\n Load trained model on {config.device} gpu...")
    model = network.cnn(num_cnn_output_c, num_cnn_output_r)
    model.load_state_dict(torch.load(config.model_dir)['model'])
    model.to(config.device)
    print(">>successful!")
    
    print("\n\n Starting prediction...")
    predict(test_dataset, config, model)


if __name__ == '__main__':
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    main()
    print("Successfully finished!")