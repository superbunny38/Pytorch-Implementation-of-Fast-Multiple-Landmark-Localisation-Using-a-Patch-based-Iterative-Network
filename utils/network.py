#ref: https://github.com/yuanwei1989/landmark-detection/blob/master/utils/network.py 
#written in Pytorch by Chaeeun Ryu

import torch.nn as nn
import torch

class cnn(nn.Module):
    def __init__(self, num_output_c, num_output_r, prob = 0.5):
        super(cnn, self).__init__()
        self.keep_prob = prob
        self.conv1_1_pool = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=32, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.conv2_1_pool = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.conv3_1_pool = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.conv4_1_pool = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.conv5_1_pool = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        
        ############ CLASSIFICATION LAYER #############
        self.cls = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=num_output_c)
        )
        
        ############ REGRESSION LAYER #############
        self.reg = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.keep_prob),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.keep_prob),
            nn.Linear(in_features=1024, out_features=num_output_r)
        )
        
        
    def forward(self, x):#x: an input tensor with the dimensions (N_examples, width, height, channel).
        #x: 101 x 101 x 3 x n_i
        x = self.conv1_1_pool(x)
        x = self.conv2_1_pool(x)
        x = self.conv3_1_pool(x)
        x = self.conv4_1_pool(x)
        x = self.conv5_1_pool(x)
        x = torch.flatten(x, start_dim=1)
        
        yc = self.cls(x)
        yr = self.reg(x)
        
        return yc, yr