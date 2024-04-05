import os

start_dir = '.'

for dirpath, dirnames, filenames in os.walk(start_dir):
    if dirpath == start_dir:
        continue

    directory_name = os.path.basename(dirpath)

    for i, filename in enumerate(filenames):
        extension = os.path.splitext(filename)[1]

        if extension.lower() == '.dng':
            new_filename = f"{directory_name}_0{extension}"
        elif extension.lower() == '.tif':
            new_filename = f"{directory_name}_{i+1}{extension}"
        else:
            continue

        old_file_path = os.path.join(dirpath, filename)
        new_file_path = os.path.join(dirpath, new_filename)

        os.rename(old_file_path, new_file_path)