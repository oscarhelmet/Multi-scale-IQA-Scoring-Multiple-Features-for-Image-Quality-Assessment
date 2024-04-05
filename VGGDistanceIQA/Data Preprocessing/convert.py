import os
from PIL import Image
from rawpy import imread, ColorSpace
import imageio
import numpy as np

'''
This scripts convert images from tif or dng to jpg format with RAW colour space  
'''



def convert_image(file_path, output_path, crop_margins=None):
    filename, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    new_size = (1994, 1296)

    if file_extension == '.tif':
        img = Image.open(file_path)
        img = img.convert('RGB')
        img = img.resize(new_size, Image.LANCZOS)
        img.save(output_path, 'JPEG')
        print(f"Sucessfully handled tif to jpg {output_path}")

    elif file_extension == '.dng':
        with imread(file_path) as raw:
            rgb = raw.postprocess(output_color=ColorSpace.raw)
            img = Image.fromarray(rgb)

            if crop_margins:  
                new_left = crop_margins['left']
                new_upper = crop_margins['top']
                new_right = img.width - crop_margins['right']
                new_lower = img.height - crop_margins['bottom']

                img = img.crop((new_left, new_upper, new_right, new_lower))
            
            img = img.resize(new_size, Image.LANCZOS)
            img.save(output_path, 'JPEG')
            print(f"Sucessfully handled dng to jpg {output_path}")

start_dir = '.'

for dirpath, dirnames, filenames in os.walk(start_dir):
    if dirpath == start_dir:
        continue

    directory_name = os.path.basename(dirpath)

    tif_file = next((f for f in filenames if f.lower().endswith('.tif')), None)
    dng_file = next((f for f in filenames if f.lower().endswith('.dng')), None)

    if tif_file and dng_file:
        tif_img = Image.open(os.path.join(dirpath, tif_file))
        with imread(os.path.join(dirpath, dng_file)) as raw:
            dng_img = Image.fromarray(raw.raw_image_visible)

        diff_width = dng_img.width - tif_img.width
        diff_height = dng_img.height - tif_img.height

        crop_margins = {
            'left': diff_width // 2,
            'right': diff_width // 2,
            'top': diff_height // 2,
            'bottom': diff_height // 2
        }

        for i, filename in enumerate(filenames):
            extension = os.path.splitext(filename)[1]

            if extension.lower() == '.dng':
                new_filename = f"{directory_name}_0.jpg"
            elif extension.lower() == '.tif':
                new_filename = f"{directory_name}_{i}.jpg"
            else:
                continue

            old_file_path = os.path.join(dirpath, filename)
            new_file_path = os.path.join(dirpath, new_filename)

            convert_image(old_file_path, new_file_path, crop_margins)
