import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import ImageDataset  
from model import Vgg16
from utils import plcc, srocc

def main(opt):
    print(opt)

    # Set the random seed for reproducibility
    np.random.seed(1)

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda state: {'available' if torch.cuda.is_available() else 'unavailable'}")

    # Prepare the model
    net = Vgg16()
    net.load_model(opt.model_file)
    net.to(device)
    net.eval()

    # Load the dataset
    lines = [line.rstrip('\n') for line in open(opt.test_file)]
    files, mos = zip(*[line.split() for line in lines])
    mos = np.array(mos, dtype=float)
    print(f"There're {len(files)} images in the test set.")

    # Define transform for resizing and converting images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create an instance of ImageDataset with transformations
    test_dataset = ImageDataset(list(zip(files, mos)), transform=transform)

    # Create a DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Testing loop
    pred = np.zeros(len(test_dataset))
    medn = np.zeros(len(test_dataset))

    with torch.no_grad():
        for i, (images, scores) in enumerate(test_loader):
            images = images.to(device)
            score_pred = net(images)
            batch_start = i * test_loader.batch_size
            batch_end = batch_start + images.size(0)

            pred[batch_start:batch_end] = torch.mean(score_pred, dim=1).cpu().numpy()
            medn[batch_start:batch_end] = torch.median(score_pred, dim=1).values.cpu().numpy()

            print(f"{i}: {pred[batch_start:batch_end]} | {medn[batch_start:batch_end]}")

    # Performance metrics
    PLCC = plcc(pred, mos)
    SROCC = srocc(pred, mos)
    print(f"PLCC = {PLCC:.4f}, SROCC = {SROCC:.4f}")

    if opt.res_file:
        with open(opt.res_file, 'w') as f:
            print(f"img_file,mos,pred,plcc,{PLCC:.4f},srcc,{SROCC:.4f}", file=f)
            for im, gt, pd in zip(files, mos, pred):
                print(f"{im},{gt},{pd}", file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True, help="trained model file")
    parser.add_argument("--test_file", type=str, required=True, help="file to store MOS and image filenames")
    parser.add_argument("--res_file", type=str, help="csv file to save the pred scores")
    args = parser.parse_args()
    main(args)