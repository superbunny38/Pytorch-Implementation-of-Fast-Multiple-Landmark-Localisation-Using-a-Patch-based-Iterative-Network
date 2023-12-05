from matplotlib import pyplot as plt

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
    

def save_loss_plot(save_losses):
    plt.subplot(131)
    plt.title("Total Loss")
    plt.plot(save_losses['save_loss'])
    plt.xlabel("Iterations")
    plt.subplot(132)
    plt.title("Classification Loss")
    plt.plot(save_losses['save_loss_c'])
    plt.xlabel("Iterations")
    plt.subplot(133)
    plt.title("Regression Loss")
    plt.plot(save_losses['save_loss_r'])
    plt.xlabel("Iterations")
    plt.savefig("trace_loss.png")
    
    
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
    print("=====================================\n\n\n\n")

def patch_support(images, patch_size):
    n_images = len(images)
    h,w = images[0].shape[0], images[0].shape[1]
    if patch_size < max(h,w)/3:
        print("It is recommended to enlarge the patch size for better performance.")