import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

'''
This script extracts the VGG-16 features to a csv file for training.
'''


# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path, layer_names):
    # Load the pretrained VGG model
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

        # Pass the original input image through the layers up to the selected one
        x = input_image
        for i in range(layer_index + 1):
            x = vgg[i](x)

        # Convert the features to a numpy array
        layer_features = x.data.numpy()
        
        # Compute the mean values across the spatial dimensions
        mean_values = layer_features.mean(axis=(2, 3))
        mean_values = np.squeeze(mean_values)
        
        features.append(mean_values)
    
    features = np.concatenate(features)
    
    print(f"Computed {image_path}'s feature vector")

    return features

# Walk through each directory
start_dir = '..\\..\\dataset\\output'  # Change this to the path of the dataset

# Define output file
output_file = 'data.csv'

conv_extraction = ['conv1_1','conv2_2','conv3_3','conv4_3','conv5_3']



with open(output_file, 'w') as f:
    for dirpath, dirnames, filenames in os.walk(start_dir):
        for degradation_set in sorted(dirnames):
            degradation_set_path = os.path.join(dirpath, degradation_set)
            for subdirname in sorted(os.listdir(degradation_set_path)):
                subdirectory_path = os.path.join(degradation_set_path, subdirname)

                # Check if the directory is indeed a directory
                if not os.path.isdir(subdirectory_path):
                    continue

                # Identify the reference photo, which is always '10.jpg'
                reference_file = '10.jpg'
                reference_path = os.path.join(subdirectory_path, reference_file)
                if not os.path.exists(reference_path):
                    print(f"Reference image not found: {reference_path}")
                    continue

                reference_features = extract_features(reference_path, conv_extraction)

                # Process all jpg files in the subdirectory
                photo_files = [file for file in os.listdir(subdirectory_path) if file.lower().endswith('.jpg')]
                for photo_file in sorted(photo_files):
                    photo_path = os.path.join(subdirectory_path, photo_file)
                    photo_features = extract_features(photo_path, conv_extraction)
                    diff = photo_features - reference_features

                # Save the difference to the file
                f.write(','.join(map(str, diff.tolist())) + '\n')
                print(f"Computed the difference between {subdirectory_path} {reference_path} and {photo_path}")

print("All comparisons are completed and saved.")