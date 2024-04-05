import collections

import numpy as np

import torch
from torch import nn
import torchvision


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()

        model = torchvision.models.vgg16(pretrained=False, num_classes=1)
        self.features = torch.nn.Sequential(
            collections.OrderedDict(
                zip(
                    [
                        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
                        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'
                    ],
                    model.features
                )
            )
        )

        self.classifier = torch.nn.Sequential(
            collections.OrderedDict(
                zip(
                    ['fc6_m', 'relu6_m', 'drop6_m', 'fc7_m', 'relu7_m', 'drop7_m', 'fc8_m'],
                    model.classifier
                )
            )
        )
        
        for layer in self.features:
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

        for layer in self.classifier:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

    def load_model(self, file, debug: bool = False):
        """
        Load model file.

        :param file: the model file to load.
        :param debug: indicate if output the debug info.
        """
        state_dict = torch.load(file, map_location=torch.device('cpu'))
        model_state_dict = self.state_dict()
        loaded_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
        model_state_dict.update(loaded_state_dict)
        self.load_state_dict(model_state_dict)  

    def forward(self, x):
        x = x.float()  # Convert input to float tensor
        out = self.features(x)
        out = torch.flatten(out, start_dim=1, end_dim=-1)  # don't use adaptive avg pooling
        out = self.classifier(out)
        return out

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.base_net = Vgg16()
    
    def forward(self, x1, x2):
        out1 = self.base_net.forward(x1)
        out2 = self.base_net.forward(x2)
        return out1, out2

if __name__ == '__main__':
    # Test the function of loading model
    vgg16 = Vgg16()
    vgg16.load_model("VGG16_Proj_dataset.pt", debug=True)
