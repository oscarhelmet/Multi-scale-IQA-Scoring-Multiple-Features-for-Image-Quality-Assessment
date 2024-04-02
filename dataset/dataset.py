import cv2
import numpy as np
import os
from scipy import ndimage
from skimage import color, img_as_float, img_as_ubyte
from skimage.util import random_noise
import math

# 1. Intensity Gain
def intensity_gain(img, gain):
    return cv2.convertScaleAbs(img, alpha=gain)

# 2. High Frequency Noise
def add_high_freq_noise(img, amount):
    noise = np.random.normal(0, amount, img.shape)
    noisy_img = cv2.add(img, noise.astype(np.uint8))
    return noisy_img

# 3. Mean Shift (Intensity Shift)
def mean_shift(img, shift_value):
    shifted_img = cv2.add(img, shift_value)
    return shifted_img

# 4. Contrast Change
def contrast_change(img, alpha):
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha)
    return adjusted_img

# 5. Change of Color Saturation
def change_color_saturation(img, saturation_scale):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[:, :, 1] = cv2.multiply(hsv_img[:, :, 1], saturation_scale)
    saturated_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return saturated_img

# 6. Masked Noise
def masked_noise(img, mask_percentage):
    if not (0 <= mask_percentage <= 1):
        raise ValueError("mask_percentage must be between 0 and 1")

    mask = np.random.binomial(1, mask_percentage, img.shape[:2])
    mask = np.dstack([mask] * 3)  # Create 3-channel mask for color image
    noisy_img = random_noise(img, mode='s&p', amount=mask_percentage)
    noisy_img = img_as_ubyte(noisy_img)
    masked_img = img.copy()
    masked_img[mask == 1] = noisy_img[mask == 1]
    return masked_img

# 7. Image Color Quantization with Dither
def color_quantization_with_dither(img, num_colors):
    # Convert to float and scale to [0, 1] for skimage functions
    img_float = img_as_float(img)
    quantized_img = color.rgb2hsv(img_float)
    for i in range(3):
        channel = quantized_img[:, :, i]
        quantized_channel = np.digitize(channel, np.histogram(channel, num_colors)[1][:-1]) / num_colors
        noise = np.random.uniform(-0.5 / num_colors, 0.5 / num_colors, channel.shape)
        quantized_img[:, :, i] = np.clip(quantized_channel + noise, 0, 1)
    quantized_img = color.hsv2rgb(quantized_img)
    return img_as_ubyte(quantized_img)

# 8. Chromatic Aberrations
def chromatic_aberrations(img, shift_value):
    b, g, r = cv2.split(img)
    r_shifted = ndimage.shift(r, (shift_value, shift_value), mode='nearest')
    g_shifted = ndimage.shift(g, (-shift_value, -shift_value), mode='nearest')
    aberrated_img = cv2.merge((b, g_shifted, r_shifted))
    return aberrated_img

# 9. Color Inversion (Gradual)
def color_inversion(img, distortion_level):
    # Create a blending weight based on the distortion level
    alpha = distortion_level / 10
    
    # Invert the color of the image
    inverted_img = cv2.bitwise_not(img)
    
    # Blend the original image with the inverted image based on the alpha value
    blended_img = cv2.addWeighted(img, 1 - alpha, inverted_img, alpha, 0)
    
    return blended_img.astype(np.uint8)

# 10. White Balance Distortion
def white_balance_distortion(img, color_offset):
    color_offset_map = {
        0: (1.0, 1.0, 1.0),    
        1: (1.1, 1.1, 1.0),    
        2: (1.2, 1.2, 1.0),    
        3: (1.3, 1.3, 1.0),    
        4: (1.4, 1.4, 1.0),    
        5: (1.5, 1.5, 1.0),    
        6: (1.6, 1.6, 1.0),    
        7: (1.7, 1.7, 1.0),    
        8: (1.8, 1.8, 1.0),    
        9: (1.9, 1.9, 1.0)     
    }
    
    offset = color_offset_map.get(color_offset, (1, 1, 1))
    
    blue_channel = img[:, :, 0] * offset[0]
    green_channel = img[:, :, 1] * offset[1]
    red_channel = img[:, :, 2] * offset[2]
    
    blue_channel = np.clip(blue_channel, 0, 255).astype(np.uint8)
    green_channel = np.clip(green_channel, 0, 255).astype(np.uint8)
    red_channel = np.clip(red_channel, 0, 255).astype(np.uint8)
    
    distorted_img = cv2.merge((blue_channel, green_channel, red_channel))
    return distorted_img

# 11. Vibrance Distortion
def vibrance_distortion(img, vibrance):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[:, :, 1] = hsv_img[:, :, 1] * vibrance
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
    vibrant_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return vibrant_img

# 12. Gaussian Blur
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# 13. JPEG Compression
def jpeg_compression(img, quality):
    quality = max(0, min(int(quality), 100))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    compressed_img = cv2.imencode('.jpg', img, encode_param)[1]
    return cv2.imdecode(compressed_img, cv2.IMREAD_COLOR)

# 14. Histogram Equalization (Gradual)
def histogram_equalization(img, distortion_level):
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # Calculate the histogram of the Y channel
    hist, _ = np.histogram(ycrcb_img[:, :, 0].flatten(), 256, [0, 256])
    
    # Calculate the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    
    # Calculate the equalization mapping
    equalization_map = np.floor((cdf_normalized * (distortion_level / 10)) * 255).astype(np.uint8)
    
    # Apply the equalization mapping to the Y channel
    ycrcb_img[:, :, 0] = cv2.LUT(ycrcb_img[:, :, 0], equalization_map)
    
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return equalized_img

# 15. Gamma Curve Distortion
def gamma_curve_distortion(img, gamma):
    gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, gamma_table)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_images(input_dir, base_output_dir, processing_categories):
    # Iterate over all files in the input directory
    for file_name in os.listdir(input_dir):
        # Skip if not an image
        if not (file_name.lower().endswith('.jpg') or file_name.lower().endswith('.png')):
            continue

        # Read the image
        input_image_path = os.path.join(input_dir, file_name)
        img = cv2.imread(input_image_path)

        # Ensure the image is loaded properly
        if img is None:
            print(f"Image {file_name} cannot be loaded. Skipping...")
            continue

        # Iterate over the categories and apply the processing
        for category_name, processing_function in processing_categories:
            # Create a directory for the category if it doesn't exist
            category_output_dir = os.path.join(base_output_dir, os.path.splitext(file_name)[0], category_name)
            ensure_dir(category_output_dir)

            # Apply the processing function and save the images for each level
            for i in range(1, 11):
                processed_img = processing_function(img, i)

                # Construct the output filename
                filename = os.path.join(category_output_dir, f'{i}.jpg')

                # Save the image
                cv2.imwrite(filename, processed_img)
                print("finished processing image: ", file_name, " category: ", category_name, " level: ", i, " saved as: ", filename)
            print("finished processing image: ", file_name, " category: ", category_name)

        # Save the original image as level 10 for each category
        for category_name, _ in processing_categories:
            category_output_dir = os.path.join(base_output_dir, os.path.splitext(file_name)[0], category_name)
            filename = os.path.join(category_output_dir, '10.jpg')
            cv2.imwrite(filename, img)

# Input directory containing the images
input_dir = 'input'

# Base output directory
base_output_dir = 'output'
ensure_dir(base_output_dir)

# Define a list of processing categories
processing_categories = [
    ('intensity_gain', lambda img, i: intensity_gain(img, 1 + (10-i) * 0.1)),
    ('high_frequency_noise', lambda img, i: add_high_freq_noise(img, 0.5 * (10 - i))),
    ('mean_shift', lambda img, i: mean_shift(img, (10 - i) * 10)),
    ('contrast_change', lambda img, i: contrast_change(img, 1 + (10 - i) * 0.2)),
    ('color_saturation', lambda img, i: change_color_saturation(img, 1 + (10 - i) * 0.2)),
    ('masked_noise', lambda img, i: masked_noise(img, (10 - i) * 0.05)),
    ('color_quantization_with_dither', lambda img, i: color_quantization_with_dither(img, round(1.4**i)**2)),
    ('chromatic_aberrations', lambda img, i: chromatic_aberrations(img, (10 - i))),
    ('color_inversion', lambda img, i: color_inversion(img, (10 - i) - 1)),
    ('white_balance_distortion', lambda img, i: white_balance_distortion(img, (10 - i) - 1)),
    ('vibrance_distortion', lambda img, i: vibrance_distortion(img, 1 + (10 - i) * 0.2)),
    ('gaussian_blur', lambda img, i: gaussian_blur(img, 2 * (10 - i) + 1)),
    ('jpeg_compression', lambda img, i: jpeg_compression(img, 10 - 10 * math.log10((10-i)+0.00000001))),
    ('histogram_equalization', lambda img, i: histogram_equalization(img, 10 - ((10 - i) - 1))),
    ('gamma_curve_distortion', lambda img, i: gamma_curve_distortion(img, 1 + (10 - i) * 0.2))
]

# Process all images in the input directory
process_images(input_dir, base_output_dir, processing_categories)

print("All images have been processed and saved successfully.")