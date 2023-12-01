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

class CTPDataset(torch.utils.data.Dataset):
    def __init__(self, names, images, labels, shape_params, pix_dim):
        assert len(images) == labels.shape[0], ('len(images): %s labels.shape: %s' % (len(images), labels.shape))
        self.num_examples = len(images)
        self.names = names
        self.images = images
        self.labels = labels
        self.shape_params = shape_params
        self.pix_dim = pix_dim

    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, index):
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
    file_names = get_file_list(file_list)

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
  train_names, train_images, train_labels, train_shape_params, train_pix_dim = extract_all_image_and_label(train_list_file,
                                                                                                           data_dir,
                                                                                                           label_dir,
                                                                                                           landmark_count,
                                                                                                           landmark_unwant,
                                                                                                           shape_model)
  
  print(">>Loading test images...")
  test_names, test_images, test_labels, test_shape_params, test_pix_dim = extract_all_image_and_label(test_list_file,
                                                                                                      data_dir,
                                                                                                      label_dir,
                                                                                                      landmark_count,
                                                                                                      landmark_unwant,
                                                                                                      shape_model)
  
  
  
  
  return