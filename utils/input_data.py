import os
import numpy as np
import nibabel as nib

class DataSet(object):
  def __init__(self,
               names,
               images,
               labels,
               shape_params,
               pix_dim):
    assert len(images) == labels.shape[0], ('len(images): %s labels.shape: %s' % (len(images), labels.shape))
    self.num_examples = len(images)
    self.names = names
    self.images = images
    self.labels = labels
    self.shape_params = shape_params
    self.pix_dim = pix_dim

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
  
  return