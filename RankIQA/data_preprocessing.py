import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import argparse


def load_dataset(base_dir):
    dataset = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".jpg"):
                score = int(file.split(".")[0])
                file_path = os.path.join(root, file)
                dataset.append((file_path, score))
                print(f"Loaded: {file_path} with score {score}")
    return dataset

def split_dataset(dataset, split_ratio=0.88):
    random.shuffle(dataset)
    split_index = int(len(dataset) * split_ratio)
    train_set = dataset[:split_index]
    test_set = dataset[split_index:]
    print(f"Train set size: {len(train_set)}")
    return train_set, test_set


class ImageDataset(Dataset):
    def __init__(self, dataset, batch_size, im_shape=(224, 224)):
        self.dataset = dataset
        self.batch_size = batch_size
        self.im_shape = im_shape
        self.num_distortions = 15
        self.num_levels = 10
        self.num_images_per_level = 1
        self.num_images_per_distortion = self.num_levels * self.num_images_per_level

    def __len__(self):
        return len(self.dataset) // self.num_distortions

    def __getitem__(self, index):
        db_inds = self._get_next_minibatch_inds(index)
        images = []
        scores = []
        for i in db_inds:
            file_path, score = self.dataset[i]
            image = cv2.imread(file_path)
            image = self._preprocess(image)
            images.append(image)
            scores.append(score)
        images = np.stack(images)
        scores = np.asarray(scores)
        return torch.from_numpy(images), torch.from_numpy(scores)

    def _get_next_minibatch_inds(self, index):
        db_inds = []
        for i in range(self.num_distortions):
            start_index = index * self.num_images_per_distortion + i * self.num_images_per_level
            db_inds.extend(range(start_index, start_index + self.num_images_per_level))
        return db_inds

    def _preprocess(self, image):
        # Perform preprocessing similar to the 'preprocess' function in the original code
        h, w, _ = image.shape
        x = np.random.randint(0, h - self.im_shape[0])
        y = np.random.randint(0, w - self.im_shape[1])
        image = image[x:x+self.im_shape[0], y:y+self.im_shape[1], :]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        return image
    
def main(opt):
    with open("test_data.txt", 'w') as f:  # Open the file in write mode
        for root, dirs, files in os.walk(opt.base_dir):  # os.walk returns a tuple of (root, dirs, files)
            for file in files:
                if file.endswith(".jpg"):
                    score = int(file.split(".")[0])
                    file_path = os.path.join(root, file)
                    f.write(f"{file_path} {score}\n")  # Write to the file within the 'with open' block
                    print(f"Loaded: {file_path} with score {score}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # "/your_own_datasets_path/TID2013/"
    # "/your_own_datasets_path/LIVE2/"
    parser.add_argument("--base_dir", type=str, default=None, help="test set path")

    main(parser.parse_args())
 