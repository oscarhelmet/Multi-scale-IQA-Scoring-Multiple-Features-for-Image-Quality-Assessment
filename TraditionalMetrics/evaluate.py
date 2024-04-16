import csv
import numpy as np
import math

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

# Define input CSV file
input_file = 'data_msssim.csv'

# Read PSNR values and filenames from the CSV file
psnr_values = []
filenames = []

with open(input_file, 'r') as f:
    csv_reader = csv.reader(f)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        filename = row[0]
        psnr = float(row[1])
        
        if not math.isnan(psnr) and not math.isinf(psnr):
            psnr_values.append(psnr)
            filenames.append(filename)

# Extract labels from filenames
labels = [float(filename.split('.')[0]) for filename in filenames]

# Calculate PLCC and SROCC
plcc_value = plcc(psnr_values, labels)
srocc_value = srocc(psnr_values, labels)

print(f"PLCC: {plcc_value}")
print(f"SROCC: {srocc_value}")