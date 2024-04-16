import cv2
import numpy as np
import os
import csv
from skimage.metrics import structural_similarity

# Walk through each directory
start_dir = '../dataset/output'  # Change this to the path of the dataset

# Define output file
output_file = 'data_ssim.csv'

# Open the output file
with open(output_file, 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Filename', 'SSIM'])  # Write header

    prev_set_id = None
    reference_image = None
    ssim_values = []
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
                reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
                prev_set_id = set_id

            if filename.endswith('10.jpg'):
                # Skip the reference image
                continue

            # Read the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Calculate SSIM
            ssim = structural_similarity(reference_image, image, data_range=255)

            # Store the filename (label) and SSIM value
            csv_writer.writerow([filename, ssim])

            # Append SSIM value and filename for PLCC and SROCC calculation
            ssim_values.append(ssim)
            filenames.append(filename)