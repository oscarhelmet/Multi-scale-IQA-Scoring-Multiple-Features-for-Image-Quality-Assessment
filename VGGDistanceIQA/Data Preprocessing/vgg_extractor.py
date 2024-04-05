import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

# Define preprocessing
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pretrained VGG16 model
model = models.vgg16(pretrained=True)
features = model.features

# Function to extract features
def extract_features(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)
    first_layer_features = features[0](image)
    first_layer_features = first_layer_features.data.numpy()
    mean_values = first_layer_features.mean(axis=(2, 3))
    mean_values = np.squeeze(mean_values)
    print(f"Computed {image_path}'s VGG-feature Vector")
    return mean_values

# Walk through each directory
start_dir = '.'

# Define output file
output_file = 'data.txt'

# Open the output file
with open(output_file, 'a') as f:
    for dirpath, dirnames, filenames in os.walk(start_dir):
        if dirpath == start_dir:
            continue

        directory_name = os.path.basename(dirpath)
        print(f"Processing {directory_name}'s Dataset")

        reference_file = next((file for file in filenames if file.endswith('_0.jpg')), None)

        if reference_file:
            reference_path = os.path.join(dirpath, reference_file)
            reference_features = extract_features(reference_path)

            for i, filename in enumerate(filenames):
                if filename == reference_file:
                    continue

                image_path = os.path.join(dirpath, filename)

                if image_path.lower().endswith('.jpg'):
                    image_features = extract_features(image_path)
                    diff = image_features - reference_features
                    f.write(' '.join(map(str, diff.tolist())) + '\n')