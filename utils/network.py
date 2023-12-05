#ref: https://github.com/yuanwei1989/landmark-detection/blob/master/utils/network.py 
#paper arxiv: https://arxiv.org/abs/1806.06987v2
#written in Pytorch by Chaeeun Ryu

import torch.nn as nn
import torch



class cnn(nn.Module):
    def __init__(self, num_output_c, num_output_r, prob = 0.5):
        super(cnn, self).__init__()
        self.keep_prob = prob
        self.num_landmarks = int(num_output_c/(2*3))
        self.conv1_1_pool = nn.Sequential(
            nn.Conv2d(in_channels=3*self.num_landmarks, out_channels=32, kernel_size=3, stride=1),
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
    
    

def predict_cnn(patches, config, model):
    """Predict with cnn

    Args:
        patches: Patches to inference, [num_examples, patch_size, patch_size, num_landmarks*3]
        config: configuration
        model: loaded model

    Returns:
        action_ind_val: predicted classification label
        yc_val: predicted classification prob
        yr_val: predicted regression value
    """
    model.eval()
    patches = torch.from_numpy(patches).float().to(config.device)
    patches = patches.permute(0, 3, 1, 2)
    yc, yr = model(patches)
    yc = nn.Softmax(dim=1)(yc)
    action_ind_val = torch.argmax(yc, dim=1)
    yc_val = torch.max(yc, dim=1)[0]
    yr_val = torch.max(yr, dim=1)[0]
    return action_ind_val, yc_val, yr_val