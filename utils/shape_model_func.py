#reference: https://github.com/yuanwei1989/landmark-detection/blob
"""Functions for calculations with shape model."""

import numpy as np

def load_shape_model(shape_model_file, eigvec_per):
    """Load the shape model.

    Args:
    shape_model_file: path and file name of shape model file (.mat)
    eigvec_per: Percentage of eigenvectors to keep (0-1)

    Returns:
    shape_model: a structure containing the shape model
    """
    
    return