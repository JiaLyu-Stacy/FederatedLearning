import json
import os
import torch
import utils
from datasets import load_dataset
import time
import torch.nn as nn
import torch.nn.functional as F

NUM_EPOCHS = 3

class Net(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train(net, trainloader, optimizer, device="cpu"):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    for batch in trainloader:
        images, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()

def test(net, testloader, device):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.to(device)
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def run_centralised(
            trainloader, testloader, epochs: int, lr: float, momentum: float = 0.9
    ):
    """A minimal (but complete) training loop"""
    # Discover device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("#"*50)
    print("Centralized PyTorch training (MNIST)")
    print("#"*50)
    print(f"Start training(via {DEVICE.type.upper()})")
    # instantiate the model
    model = Net(num_classes=10)
    model.to(DEVICE)
    # define optimiser with hyperparameters supplied
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    # train for the specified number of epochs
    for e in range(epochs):
        print(f"Training epoch {e} ...")
        train(model, trainloader, optim, DEVICE)    
    # training is completed, then evaluate model on the test set
    print("Evaluate model")
    loss, accuracy = test(model, testloader, DEVICE)
    print(f"Loss: {loss:.3f} ")
    print(f"Accuracy: {accuracy:.3f} ")
    return loss, accuracy

def main():
    # Construct dataloaders
    # Download dataset
    print("Load data")
    mnist = load_dataset("ylecun/mnist")  # 60000 training samples, 10000 test samples
    trainloader, testloader = utils.get_mnist_dataloaders(mnist, batch_size=32)
    start_time = time.time()
    # Run the centralised training
    loss, accuracy = run_centralised(trainloader, testloader, epochs=NUM_EPOCHS, lr=0.01)
    elapsed_time = time.time() - start_time
    print(f"Total running time: {elapsed_time:.2f} seconds")
    print("Finished training")
    # Save NUM_EPOCHS and elapsed_time to a file
    metadata = {
        "run_time": f"{round(elapsed_time, 2)} seconds",
        "final_loss": round(loss, 2),
        "final_accuracy": accuracy,
        "batch_size": 32,
        "learning_rate": 0.01,
        "num_epochs": NUM_EPOCHS,
    }
    # Check if the file exists
    output_file = "./output/mnist_centralized.json"
    if os.path.exists(output_file):
        # If the file exists, read the existing data, append, and save
        with open(output_file, "r") as f:
            existing_data = json.load(f)
        # Append new data
        existing_data.append(metadata)
        # Save the combined data back to the file
        with open(output_file, "w") as f:
            json.dump(existing_data, f, indent=2)
    else:
        # If the file doesn't exist, create a new one with the combined data
        with open(output_file, "w") as f:
            json.dump([metadata], f, indent=2)


if __name__ == "__main__":
    main()

