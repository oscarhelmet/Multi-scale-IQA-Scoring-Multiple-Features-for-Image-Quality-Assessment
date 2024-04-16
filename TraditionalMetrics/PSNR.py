import cv2
import numpy as np
import os
import csv
from skimage.metrics import peak_signal_noise_ratio

def plcc(x, y):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    if len(x.shape) != 1 or len(y.shape) != 1:
        raise Exception("Please input N (* 1) vector.")
    if x.shape[0] != y.shape[0]:
        raise Exception("The lengths of 2 input vectors are not equal.")

    x = x - np.average(x)
    y = y - np.average(y)
    numerator = np.dot(x, y)
    denominator = np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2))
    return numerator / denominator

def srocc(x, y):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    if len(x.shape) != 1 or len(y.shape) != 1:
        raise Exception("Please input N (* 1) vector.")
    if x.shape[0] != y.shape[0]:
        raise Exception("The lengths of 2 input vectors are not equal.")

    rank_x = x.argsort().argsort()
    rank_y = y.argsort().argsort()
    return plcc(rank_x, rank_y)


# Walk through each directory
start_dir = '../dataset/output'  # Change this to the path of the dataset

# Define output file
output_file = 'PSNR_value.csv'

# Open the output file
with open(output_file, 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Filename', 'PSNR'])  # Write header

    prev_set_id = None
    reference_image = None
    psnr_values = []
    filenames = []

    for dirpath, dirnames, filenames_in_dir in os.walk(start_dir):
        filenames_in_dir.sort()  # Sort the filenames to ensure correct order

        for filename in filenames_in_dir:
            if not filename.lower().endswith('.jpg'):
                continue

            image_path = os.path.join(dirpath, filename)
            set_id = os.path.basename(dirpath)

            if set_id != prev_set_id:
                # New set of images, update the reference image
                reference_file = "10.jpg"
                reference_path = os.path.join(dirpath, reference_file)
                reference_image = cv2.imread(reference_path)
                prev_set_id = set_id

            if filename.endswith('10.jpg'):
                # Skip the reference image
                continue

            # Read the image
            image = cv2.imread(image_path)

            # Calculate PSNR
            psnr = peak_signal_noise_ratio(reference_image, image)
            print(f"calcualted {filename} and PSNR: {psnr}")
            # Store the filename (label) and PSNR value
            csv_writer.writerow([filename, psnr])

            # Append PSNR value and filename for PLCC and SROCC calculation
            psnr_values.append(psnr)
            filenames.append(filename)

# Extract labels from filenames
labels = [float(filename.split('.')[0]) for filename in filenames]

# Calculate PLCC and SROCC
plcc_value = plcc(psnr_values, labels)
srocc_value = srocc(psnr_values, labels)

print(f"PLCC: {plcc_value}")
print(f"SROCC: {srocc_value}")