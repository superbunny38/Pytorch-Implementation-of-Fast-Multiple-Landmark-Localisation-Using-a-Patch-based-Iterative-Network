import numpy as np
import argparse
from viz import support
from utils import input_data, network, patch, predict, visual, plane
import torch
import scipy.ndimage
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
parser.add_argument('--save_viz', type=bool, default = True, help='Whether to save visualisation.')
parser.add_argument('--print_config', type=bool, default=False, help='Whether to print out the configuration')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to run gpu on.')
parser.add_argument('--print_info', type=bool, default=False, help='Whether to print out information about this code.')
parser.add_argument('--dimension', type=int, default=3, help='Dimension of the data.')
parser.add_argument('--landmark_count', type=int, default=2, help='Number of landmarks.')
parser.add_argument('--patch_size', type=int, default=101, help='Patch size (odd), Recommended: at least 1/3 of max(height, width).')
parser.add_argument('--update_rule_help', type=bool, default=False, help='Whether get help with info about the update rule.')
parser.add_argument('--save_log', type=bool, default=False, help='whether to save inference log.')
parser.add_argument('--get_latest_trained_model', type = bool, default = True, help='Whether to get the latest trained model.')
parser.add_argument('--model_dir', type = str, default = './ckpt/models/04_12_18_46_26_model.pth', help = 'Directory to save the trained model.')
parser.add_argument('--train', type=bool, default = True, help='Whether to check inference performance w/ training data.')
args = parser.parse_args()

class Config(object):
    """Inference configurations."""
    # File paths
    data_dir = './data/Images'
    label_dir = './data/Landmarks'
    train_list_file = './data/list_train.txt'
    test_list_file = './data/list_test.txt'
    if args.get_latest_trained_model:
        model_dir = support.get_the_latest_ckpt()
        print("model dir: {}".format(model_dir))
    else:
        model_dir = args.model_dir
    
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
    visual = args.save_viz           # Whether to save visualisation
    
    #Chaeeun parameters
    print_config = args.print_config
    device = args.device
    print_info = args.print_info
    dimension = args.dimension
    update_rule_help = args.update_rule_help
    save_log = args.save_log
    train = args.train

###################################### Configuration done! ######################################




################################ Compute Error ############################################
def compute_err(landmarks, landmarks_gt, pix_dim):
    """Compute the distance error between predicted and GT landmarks.

    Args:
      landmarks: Predicted landmarks [img_count, num_landmarks, 3].
      landmarks_gt: Ground truth landmarks. [img_count, num_landmarks, 3]
      pix_dim: Pixel spacing. [img_count, 3]

    Returns:
      err: distance error in voxel. [img_count, num_landmarks]
      err_mm: distance error in mm. [img_count, num_landmarks]

    """
    num_landmarks = landmarks.shape[1]
    err = np.sqrt(np.sum(np.square(landmarks - landmarks_gt), axis=-1))
    err_mm = np.sqrt(np.sum(np.square((landmarks - landmarks_gt) * pix_dim[:, np.newaxis, :]), axis=-1))
    err_mm_landmark_mean = np.mean(err_mm, axis=0)
    err_mm_landmark_std = np.std(err_mm, axis=0)
    err_mm_mean = np.mean(err_mm)
    err_mm_std = np.std(err_mm)
    str = "Mean distance error (mm): "
    for j in xrange(num_landmarks):
        str += ("{:.10f} ".format(err_mm_landmark_mean[j]))
    print("{}".format(str))
    str = "Std distance error (mm): "
    for j in xrange(num_landmarks):
        str += ("{:.10f} ".format(err_mm_landmark_std[j]))
    print("{}".format(str))
    print("Mean distance error (mm) = {:.10f} \nStd distance error (mm) = {:.10f}\n".format(err_mm_mean, err_mm_std))
    return err, err_mm




###################################### Main function ############################################


def main():
    config = Config()
    
    print("\n\n\n\n\n\n\n\n\n\n")
    print("================[Starting inference]================")

    # Load one image
    num_cnn_output_c, num_cnn_output_r = 2*config.landmark_count*config.dimension, config.landmark_count*config.dimension
    config.num_cnn_output_c, config.num_cnn_output_r = num_cnn_output_c, num_cnn_output_r
    
    #Load model
    print("\n\nLoading PIN... ")
    model = network.cnn(num_cnn_output_c, num_cnn_output_r)
    model.load_state_dict(torch.load(config.model_dir, map_location=torch.device(config.device))['model'])
    model.to(config.device)
    print(">>successful!")
    
    #Load data
    print("\n\nLoading data...")
    if config.train:
        train_dataset, test_dataset = input_data.read_data_sets(config.data_dir, config.label_dir, config.train_list_file, config.test_list_file, config.dimension, config.landmark_count, config.landmark_unwant)
    else:
        train_dataset, test_dataset = input_data.read_data_sets(config.data_dir, config.label_dir, config.train_list_file, config.test_list_file, config.dimension, config.landmark_count, config.landmark_unwant)
    print(">>successful!")
   
    print("\n\nPredicting landmarks...")
    #img shape (324,207,279) -> img[..., np.newaxis] shape: (324,207,279,1)
    # print(train_dataset.images[0].shape) = (324, 207, 279, 1)
    # print(train_dataset.labels[0].shape) = (2,3)
    final_predicted_landmarks, landmark_prediction_steps = predict.predict_landmarks(train_dataset,config, model, train = True)
    print(">>successful!")

    print("\n\nComputing errors...")
    landmark_idx = 1
    for pos_hat, pos_gt in zip(final_predicted_landmarks[0], train_dataset.labels[0]):
        print(f"predicted landmark {landmark_idx} position:",pos_hat)
        print(f"GT landmark {landmark_idx} position:",pos_gt)
        landmark_idx +=1
        print()
        

    print(">>successful!")
    
    #Save visualizations
    print("Saving visualizations...")
    if config.visual:
        for i in range(len(train_dataset.images)):
            visual.plot_landmarks_3d('./results/landmarks_visual3D', config.train, train_dataset.names[i], final_predicted_landmarks[i],
                                     train_dataset.labels[i], train_dataset.images[i].shape)
            visual.plot_landmarks_path('./results/landmark_path', config.train, train_dataset.names[i], landmark_prediction_steps[i],
                                       train_dataset.labels[i], train_dataset.images[i].shape)





if __name__ == '__main__':
    main()