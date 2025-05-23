import time
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import Tensor
from torchvision.datasets import CIFAR10
import os

import utils

DATA_ROOT = "./data/"
NUM_EPOCHS = 10

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_data() -> Tuple[
    torch.utils.data.DataLoader, 
    torch.utils.data.DataLoader, 
    Dict
]:
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
              (0.5, 0.5, 0.5), 
              (0.5, 0.5, 0.5)
         )
        ]
    )
    trainset = CIFAR10(DATA_ROOT, 
                       train=True, 
                       download=True, 
                       transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=32, 
                                              shuffle=True)
    testset = CIFAR10(DATA_ROOT, 
                      train=False, 
                      download=True, 
                      transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=32, 
                                             shuffle=False)
    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples

def train(
        net: Net,
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
        lr: float, 
        device: torch.device,
    ) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, 
                                                i + 1, 
                                                running_loss / 2000))
                running_loss = 0.0

def test(
        net: Net,
        testloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy

def main():
    data_dir = "data/cifar-10-batches-py"
    # This is where CIFAR-10 unpacks its files
    expected_file = os.path.join(data_dir, "data_batch_1")
    if not os.path.exists(expected_file):
        print("Dataset not found. Downloading CIFAR-10...")
        CIFAR10(root="data", train=True, download=True)
        CIFAR10(root="data", train=False, download=True)
    else:
        print("CIFAR-10 already exists. Skipping download.")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("#"*50)
    print("Centralized PyTorch training (CIFAR)")
    print("#"*50)
    print("Load data")
    trainloader, testloader, _ = load_data()
    print(f"Start training(via {DEVICE.type.upper()})")
    start_time = time.time()
    net=Net().to(DEVICE)
    train(net=net, trainloader=trainloader, epochs=NUM_EPOCHS, lr=0.01, device=DEVICE)
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print(f"Loss: {loss:.3f} ")
    print(f"Accuracy: {accuracy:.3f} ")
    output_file = "./output/cifar_centralized.json"
    metadata = {
            "run_time": f"{round(time.time() - start_time, 2)} seconds",
            "final_loss": round(loss, 2),
            "final_accuracy": accuracy,
            "batch_size": 32,
            "learning_rate": 0.01,
            "num_epochs": NUM_EPOCHS,
    }
    utils.store_result(metadata, output_file)

if __name__ == "__main__":
    main()