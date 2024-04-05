import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from evaluate import evaluate
from data_preprocessing import load_dataset, split_dataset, ImageDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


class EfficientSiameseLoss(nn.Module):
    def __init__(self, margin=10):
        super(EfficientSiameseLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2):
        dis = torch.abs(output1 - output2)
        loss = torch.clamp(self.margin - dis, min=0)
        return torch.mean(loss)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    for images, scores in progress_bar:
        images = images.to(device)
        scores = scores.to(device)
        
        batch_size = images.size(0)
        num_distortions = 15
        num_levels = 10
        num_images_per_level = 2
        
        dis_loss = 0.0
        dis_count = 0
        
        for k in range(num_distortions):
            for i in range(num_levels - 1):
                for j in range(i + 1, num_levels):
                    idx1 = k * num_levels * num_images_per_level + i * num_images_per_level
                    idx2 = k * num_levels * num_images_per_level + j * num_images_per_level
                    
                    x1 = images[idx1:idx1+num_images_per_level]
                    x2 = images[idx2:idx2+num_images_per_level]
                    
                    output1, output2 = model(x1,x2)

                    
                    loss = criterion(output1, output2)
                    dis_loss += loss
                    dis_count += 1
        
        dis_loss /= dis_count
        
        optimizer.zero_grad()
        dis_loss.backward()
        optimizer.step()
        
        train_loss += dis_loss.item() * batch_size
    
    train_loss /= len(train_loader.dataset)
    return train_loss



def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    progress_bar = tqdm(val_loader, desc="Validation", unit="batch")
    with torch.no_grad():
        for images, scores in progress_bar:
            images = images.to(device)
            scores = scores.to(device)
            
            batch_size = images.size(0)
            num_distortions = 15
            num_levels = 10
            num_images_per_level = 1
            
            dis_loss = 0.0
            dis_count = 0
            
            for k in range(num_distortions):
                for i in range(num_levels - 1):
                    for j in range(i + 1, num_levels):
                        idx1 = k * num_levels * num_images_per_level + i * num_images_per_level
                        idx2 = k * num_levels * num_images_per_level + j * num_images_per_level
                        
                        x1 = images[idx1:idx1+num_images_per_level]
                        x2 = images[idx2:idx2+num_images_per_level]
                        
                        output1 = model(x1)
                        output2 = model(x2)
                        
                        loss = criterion(output1, output2)
                        dis_loss += loss
                        dis_count += 1
            
            dis_loss /= dis_count
            val_loss += dis_loss.item() * batch_size
    
    val_loss /= len(val_loader.dataset)
    return val_loss

def main():
    try:
        base_dir = "dataset/output/"
        dataset = load_dataset(base_dir)
        train_set, test_set = split_dataset(dataset)
        print(f"Train set size: {len(train_set)}")
    except Exception as e:
        print(f"An error occurred: {e}")

    train_dataset = ImageDataset(train_set,2)
    test_dataset = ImageDataset(test_set,2)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training on ranking data
    rank_model = SiameseNet().to(device)
    rank_optimizer = optim.Adamax(rank_model.parameters(), lr=3e-4)

    num_epochs = 50
    train_losses = []
    validate_losses = []
    for epoch in range(num_epochs):      

        train_loss = train(rank_model, train_loader, EfficientSiameseLoss(margin=10), rank_optimizer, device)
        train_losses.append(train_loss)
        test_loss = validate(rank_model, test_loader, EfficientSiameseLoss(margin=10), device)
        validate_losses.append(test_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}", f"Test Loss: {test_loss:.4f}")

    torch.save(rank_model.state_dict(), "ranking_model.pt")

    # # Fine-tuning on IQA data  
    # ft_model = Vgg16().to(device)
    # ft_model.load_state_dict(rank_model.base_net.state_dict())

    # criterion = nn.MSELoss()
    # ft_optimizer = optim.Adam(ft_model.parameters(), lr=1e-6)

    # num_epochs = 50
    # train_losses = []
    # test_losses = []

    # for epoch in range(num_epochs):
    #     train_loss = train(ft_model, train_loader, criterion, ft_optimizer, device) 
    #     test_loss = evaluate(ft_model, test_loader, criterion, device)
    #     train_losses.append(train_loss)
    #     test_losses.append(test_loss)
    #     print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
    # torch.save(ft_model.state_dict(), "ft_model.pt")

    plt.figure()
    plt.plot(range(num_epochs), train_losses, label="Train Loss")
    plt.plot(range(num_epochs), validate_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.jpg")

    plt.figure()
    plt.plot(range(num_epochs), [1 - loss for loss in train_losses], label="Train Accuracy")
    plt.plot(range(num_epochs), [1 - loss for loss in validate_losses], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy.jpg")

if __name__ == "__main__":  
    main()




# def train(model, train_loader, criterion, optimizer, device):
#     model.train()
#     train_loss = 0.0
#     progress_bar = tqdm(train_loader, desc="Training", unit="batch")
#     for image1, image2, label in train_loader:
#         image1 = image1.to(device)
#         image2 = image2.to(device)
#         optimizer.zero_grad()
#         output1 = model(image1)
#         output2 = model(image2)
#         loss = criterion(output1, output2)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item() * image1.size(0)
#     train_loss /= len(train_loader.dataset)
#     return train_loss

# def train(model, train_loader, criterion, optimizer, device):
    # model.train()
    # train_loss = 0.0
    # start_time = time.time()
    # progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    # for images, scores in progress_bar:
    #     images = images.to(device).float()
    #     scores = scores.to(device).float()
    #     optimizer.zero_grad()
    #     outputs = model(images)
    #     loss = criterion(outputs, scores.unsqueeze(1).float())
    #     loss.backward()
    #     optimizer.step()
    #     train_loss += loss.item() * images.size(0)
    #     progress_bar.set_postfix(loss=loss.item())
    # train_loss /= len(train_loader.dataset)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Training time: {elapsed_time:.2f} seconds")
    # return train_loss