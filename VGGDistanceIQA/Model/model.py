import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from PIL import Image
import numpy as np

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_features(image_path, layer_names):
    vgg = models.vgg16(pretrained=True).features
    layer_map = {
        'conv1_1': 0, 'relu1_1': 1, 'conv1_2': 2, 'relu1_2': 3, 'pool1': 4,
        'conv2_1': 5, 'relu2_1': 6, 'conv2_2': 7, 'relu2_2': 8, 'pool2': 9,
        'conv3_1': 10, 'relu3_1': 11, 'conv3_2': 12, 'relu3_2': 13, 'conv3_3': 14, 'relu3_3': 15, 'pool3': 16,
        'conv4_1': 17, 'relu4_1': 18, 'conv4_2': 19, 'relu4_2': 20, 'conv4_3': 21, 'relu4_3': 22, 'pool4': 23,
        'conv5_1': 24, 'relu5_1': 25, 'conv5_2': 26, 'relu5_2': 27, 'conv5_3': 28, 'relu5_3': 29, 'pool5': 30,
    }
    vgg.eval()
    
    image = Image.open(image_path)
    input_image = preprocess(image).unsqueeze(0)
    
    features = []
    
    for layer_name in layer_names:
        layer_index = layer_map[layer_name]

        x = input_image
        for i in range(layer_index + 1):
            x = vgg[i](x)

        layer_features = x.data.numpy()
        
        mean_values = layer_features.mean(axis=(2, 3))
        mean_values = np.squeeze(mean_values)
        
        features.append(mean_values)
    
    features = np.concatenate(features)

    return features


class Multiscale(nn.Module):
    def __init__(self):
        super(Multiscale, self).__init__()
        self.layer1 = nn.Linear(1472, 1472)
        self.layer2 = nn.Linear(1472, 1472)
        self.layer3 = nn.Linear(1472, 1472)
        self.layer4 = nn.Linear(1472, 1472)
        self.layer8 = nn.Linear(1472, 1472) 
        self.layer9 = nn.Linear(1472, 1472) 
        self.layer10 = nn.Linear(1472, 10)

       
        self.dropout = nn.Dropout(p=0.85)
        
        init.kaiming_normal_(self.layer1.weight)
        init.kaiming_normal_(self.layer2.weight)
        init.kaiming_normal_(self.layer3.weight)
        init.kaiming_normal_(self.layer4.weight)
        init.kaiming_normal_(self.layer8.weight)
        init.kaiming_normal_(self.layer9.weight)
        init.kaiming_normal_(self.layer10.weight)

        
    def forward(self, x):
        x = F.relu(self.layer1(x))       
        x = F.relu(self.layer2(x))        
        x = F.relu(self.layer3(x))
        x = self.dropout(x)
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer8(x))
        x = F.relu(self.layer9(x))
        x = self.dropout(x)
        x = self.layer10(x) 
        return F.log_softmax(x, dim=1) 
    
    def load_model(self, file):
        state_dict = torch.load(file, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)