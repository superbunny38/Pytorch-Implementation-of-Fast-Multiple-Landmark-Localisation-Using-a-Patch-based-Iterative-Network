# Autoencoder to replace the shape model
# Multi landmarkds detection인 경우 landmarks만 pca로 차원축소한다는 것 같음

import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(64,12),
            nn.ReLU(True),
            nn.Linear(12, out_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(),
            nn.ReLU(True),
            nn.Linear(),
            nn.ReLU(True),
            nn.Linear(),
            nn.ReLU(True),
            nn.Linear()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
def load_model(in_size, device):
    autoencoder = Autoencoder(in_size).to(device)
    return autoencoder

def landmarks2b(landmarks, autoencoder, device):
    """
    Transform landmarks to compressed representation using the autoencoder.

    Args:
        landmarks: landmark coordinates. [num_examples, num_landmarks, 3]
        autoencoder: used for data compression instead of the shape model.
        device: gpu
        
    Returns:
        compressed_landmarks: compressed landmark coordinates. [num_examples,num_shape_params]
    """
    
    b = autoencoder.encoder(landmarks)
    return b

def b2landmarks(b, autoencoder, device):
    """Transform compressed representation to landmark coordinates using the autoencoder.

    Args:
        b: compressed landmark coordinates. [num_examples, num_shape_params]
        autoencoder: used for data compression instead of the shape model.
        device: gpu
    """
    landmarks = autoencoder.decoder(b)
    return landmarks
