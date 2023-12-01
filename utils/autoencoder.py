# Autoencoder to replace the shape model
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 4, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 2, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2, 1, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 2, 3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(2, 4, 3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 8, 3, padding=1)
        )
        
        
    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = F.sigmoid(self.decoder(compressed))
        return compressed, reconstructed


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
