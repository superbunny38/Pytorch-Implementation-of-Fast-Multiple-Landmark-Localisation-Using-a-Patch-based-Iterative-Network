import numpy as np
import argparse
from viz import support
from utils import input_data, network

parser = argparse.ArgumentParser(description='Argparse')
parser.add_argument('--max_test_steps', type=int, default=10, help='Number of inference steps.')
parser.add_argument('--num_random_init', type=int, default=5, help='Number of random initialisations used.')
parse.add_argument('--predict_mode', type=int, default=1, help='How the new patch position is computed.")
parser.add_argument('--save_viz', type=bool, help='Whether to save visualisation.')
parser.add_argument('--print_config', type=bool, default=False, help='Whether to print out the configuration')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to run gpu on.')
parser.add_argument('--print_info', type=bool, default=False, help='Whether to print out information about this code.')
parser.add_argument('--dimension', type=int, default=3, help='Dimension of the data.')
args = parser.parse_args()


class Config(object):
    """Inference configurations."""
    # File paths
    data_dir = './data/Images'
    label_dir = './data/Landmarks'
    train_list_file = './data/list_train.txt'
    test_list_file = './data/list_test.txt'
    model_dir = './ckpt/models/cnn_model.pt'
    # Shape model parameters
    shape_model_file = './shape_model/shape_model/ShapeModel.mat'
    eigvec_per = 0.995      # Percentage of eigenvectors to keep
    sd = 3.0                # Standard deviation of shape parameters
    landmark_count = 2     # Number of landmarks
    landmark_unwant = []     # list of unwanted landmark indices
    # Testing parameters
    box_size = 101          # patch size (odd number)
    max_test_steps = args.max_test_steps     # Number of inference steps
    num_random_init = parser.num_random_init     # Number of random initialisations used
    predict_mode = parser.predict_mode        # How the new patch position is computed.
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

def predict():
    return

def main():
    config = Config()
    
    if args.print_config:
        support.print_config_inference(config)
    if args.get_info:
        support.print_info(config)
    
    print("\n\n\n\n\n\n\n\n\n\n")
    print("================[Starting Inference]================")
    num_cnn_output_c, num_cnn_output_r = 2*config.landmark_count*config.dimension, config.landmark_count*config.dimension
    print("\n\nLoading data...")
    _, test_dataset = input_data.read_data_sets(config.data_dir, config.label_dir, config.train_list_file, config.test_list_file, config.dimension, config.landmark_count, config.landmark_unwant)
    print(">>successful!")
    
    print(f"\n\n Load trained model on {config.device} gpu...")
    model = network.cnn(num_cnn_output_c, num_cnn_output_r)
    model.load_state_dict(torch.load(config.model_dir))
    model.to(config.device)
    print(">>successful!")
    
    predict(test_dataset, config)