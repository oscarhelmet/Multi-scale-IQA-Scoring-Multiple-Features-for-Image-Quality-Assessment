import torch
from torch.utils.data import DataLoader
from model import Vgg16
from data_preprocessing import load_dataset, ImageDataset

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, scores in test_loader:
            images = images.to(device)
            scores = scores.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += scores.size(0)
            correct += (predicted == scores).sum().item()
    accuracy = correct / total
    return accuracy

def main():
    base_dir = "dataset/output"
    dataset = load_dataset(base_dir)
    test_dataset = ImageDataset(dataset)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Vgg16().to(device)
    model.load_state_dict(torch.load("model_weights.pth"))

    accuracy = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()