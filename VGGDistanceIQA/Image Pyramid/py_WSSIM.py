import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.transform import pyramid_gaussian

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr



def calculate_ssim(img1, img2):
    data_range = img1.max() - img1.min()  # Calculate data range
    # If the images are 3D, indicate that the last axis is the channel axis
    channel_axis = -1 if img1.ndim == 3 else None
    # Calculate SSIM
    ssim_val = ssim(img1, img2, data_range=data_range, multichannel=True, channel_axis=channel_axis)
    return ssim_val

def process_image(img, level):
    pyramid = tuple(pyramid_gaussian(img, max_layer=level))
    processed_image = cv2.resize(pyramid[-1], (img.shape[1], img.shape[0]))
    return processed_image

def main(raw_image_path, processed_image_path):
    raw_img = cv2.imread(raw_image_path)
    processed_img = cv2.imread(processed_image_path)

    if raw_img is None:
        print(f"Could not open/read the image at {raw_image_path}")
        return

    if processed_img is None:
        print(f"Could not open/read the image at {processed_image_path}")
        return
    psnr = calculate_psnr(raw_img, processed_img)
    ssim = calculate_ssim(raw_img, processed_img)

    print(f"Initial PSNR: {psnr}")
    print(f"Initial SSIM: {ssim}")

    for i in range(1, 5):
        downsampled_raw_img = process_image(raw_img, i)
        downsampled_processed_img = process_image(processed_img, i)

        psnr = calculate_psnr(downsampled_raw_img, downsampled_processed_img)
        ssim = calculate_ssim(downsampled_raw_img, downsampled_processed_img)

        print(f"\nLevel {i} PSNR: {psnr}")
        print(f"Level {i} SSIM: {ssim}")

if __name__ == "__main__":
    raw_image_path = 'source/img/2/a0002-dgw_005.dng'
    processed_image_path = 'source/img/2/a0002-dgw_005 (1).tif'
    main(raw_image_path, processed_image_path)