import os
import numpy as np
import torch
import nibabel as nib

# class DataSet(object):
#   def __init__(self,
#                names,
#                images,
#                labels,
#                shape_params,
#                pix_dim):
#     assert len(images) == labels.shape[0], ('len(images): %s labels.shape: %s' % (len(images), labels.shape))
#     self.num_examples = len(images)
#     self.names = names
#     self.images = images
#     self.labels = labels
#     self.shape_params = shape_params
#     self.pix_dim = pix_dim

#train_names, train_images, train_labels, train_pix_dim
class CTPDataset(torch.utils.data.Dataset):
    def __init__(self, names, images, labels, pix_dim):
        assert len(images) == labels.shape[0], ('len(images): %s labels.shape: %s' % (len(images), labels.shape))
        self.num_examples = len(images)
        self.names = names
        self.images = images
        self.labels = labels
        self.pix_dim = pix_dim

    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, index):#train_pix_dim도 로딩해줘야하나?
        pass

def extract_image(filename):
    """Extract the image into a 3D numpy array [x, y, z].

    Args:
        filename: Path and name of nifti file.

    Returns:
        data: A 3D numpy array [x, y, z]
        pix_dim: pixel spacings

    """
    img = nib.load(filename)
    data = np.array(img.dataobj)#img.get_data()
    data[np.isnan(data)] = 0
    pix_dim = np.array(img.header.get_zooms())
    return data, pix_dim

def extract_label(filename):
    """Extract the labels (landmark coordinates) into a 2D float64 numpy array.

    Args:
        filename: Path and name of txt file containing the landmarks. One row per landmark.

    Returns:
        labels: a 2D float64 numpy array. [landmark_count, 3]
    """
    with open(filename) as f:
        labels = np.empty([0, 3], dtype=np.float64)
        for line in f:
            labels = np.vstack((labels, map(float, line.split())))
    return labels
    

def get_file_list(txt_file):
    """Get a list of filenames

    Args:
        txt_file: Name of a txt file containing a list of filenames for the images.

    Returns:
        filenames: A list of filenames for the images.

    """
    with open(txt_file) as f:
        filenames = f.read().splitlines()
    return filenames


def extract_all_image_and_label(file_list, data_dir, label_dir, landmark_count, landmark_unwant):
    #원래는 여기서 data compression이 진행됨..
    
    file_names = get_file_list(file_list)
    file_count = len(file_names)
    images = []
    labels = np.zeros((file_count, landmark_count, 3),dtype=np.float64)
    pix_dim = np.zeros((file_count,3))
    
    for i in range(len(file_names)):
        file_name = file_names[i]
        #load image
        img, pix_dim[i] = extract_image(os.path.join(data_dir, file_name+'.nii.gz'))
        #load landmarks
        label = extract_label(os.path.join(label_dir, filename+'_ps.txt'))
        
        #Store extracted data
        images.append(np.expand_dims(img,axis=3))
        labels[i,:,:] = label
    
    return filenames, images, labels, pix_dim
    

def read_data_sets(data_dir, label_dir, train_list_file, test_list_file,landmark_count,landmark_unwant,shape_model):
    """Load training and test dataset.

    Args:
        data_dir: Directory storing images.
        label_dir: Directory storing labels.
        train_list_file: txt file containing list of filenames for train images
        test_list_file: txt file containing list of filenames for test images
        landmark_count: Number of landmarks used (unwanted landmarks removed)
        landmark_unwant: list of unwanted landmark indices
        shape_model: structure storing the shape model

    Returns:
        data: A collections.namedtuple containing fields ['train', 'validation', 'test']

    """
    print(">>Loading train (& val) images...")
    train_names, train_images, train_labels, train_pix_dim = extract_all_image_and_label(train_list_file,data_dir,
                                                                                                            label_dir,
                                                                                                            landmark_count,
                                                                                                            landmark_unwant,
                                                                                                            )
    
    print(">>Loading test images...")
    test_names, test_images, test_labels, test_pix_dim = extract_all_image_and_label(test_list_file,
                                                                                                        data_dir,
                                                                                                        label_dir,
                                                                                                        landmark_count,
                                                                                                        landmark_unwant,
                                                                                                        )
    
    
    train_dataset = CTPDataset(train_names, train_images, train_labels, train_pix_dim)
    test_dataset = CTPDataset(test_names, test_images, test_labels, test_pix_dim)
    return train_dataset, test_dataset