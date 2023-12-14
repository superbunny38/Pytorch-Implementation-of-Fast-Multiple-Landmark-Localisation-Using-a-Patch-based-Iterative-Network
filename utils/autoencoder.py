# Autoencoder to replace the shape model
# Multi landmarkds detection인 경우 landmarks만 pca로 차원축소한다는 것 같음

import torch.nn as nn
import torch.nn.functional as F
    
class Autoencoder(nn.Module):
    """
    Input: landmarks: Landmark coordinates. [num_examples, num_landmarks, 3]
    Returns: b: shape model parameters. [num_examples, num_shape_params]
    """
    
    def __init__(self, num_shape_params, num_landmarks):
        super(Autoencoder, self).__init__()
        self.out1 = 128
        self.out2 = 64
        self.out3 = 48
        self.out4 = 12
        self.final = num_shape_params
        self.encoder = nn.Sequential(
            nn.Linear(3*num_landmarks, 128),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(64,12),
            nn.ReLU(True),
            nn.Linear(12, num_shape_params),
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_shape_params, 12),
            nn.ReLU(True),
            nn.Linear(12,64),
            nn.ReLU(True),
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128,3*num_landmarks)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
def load_model(config):
    autoencoder = Autoencoder(config.num_shape_params, config.landmark_count).to(config.device)
    return autoencoder

def landmarks2b(landmarks, autoencoder, config):
    """
    Transform landmarks to compressed representation using the autoencoder.

    Args:
        landmarks: landmark coordinates. [num_examples, num_landmarks, 3]
        autoencoder: used for data compression instead of the shape model.
        device: gpu
        
    Returns:
        compressed_landmarks: compressed landmark coordinates. [num_examples,num_shape_params]
    """
    
    b = autoencoder.encoder(landmarks.to(config.device))
    return b

def b2landmarks(b, autoencoder, config):
    """Transform compressed representation to landmark coordinates using the autoencoder.

    Args:
        b: compressed landmark coordinates. [num_examples, num_shape_params]
        autoencoder: used for data compression instead of the shape model.
        device: gpu
        
    Returns:
        landmarks: Landmark coordinates. [num_examples, num_landmarks, 3]
    """
    landmarks = autoencoder.decoder(b.to(config.device)).reshape(-1,config.landmark_count,3)
    return landmarks
