import json
import os
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader

def get_mnist_dataloaders(mnist_dataset, batch_size: int):
      pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

      # Prepare transformation functions
      def apply_transforms(batch):
            batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
            return batch

      mnist_train = mnist_dataset["train"].with_transform(apply_transforms)
      mnist_test = mnist_dataset["test"].with_transform(apply_transforms)

      # Construct PyTorch dataloaders
      trainloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
      testloader = DataLoader(mnist_test, batch_size=batch_size)
      return trainloader, testloader

def store_result(metadata, output_file):

      # Check if the file exists
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


def combined_result(file_1, file_2, result_name):
    # Load both JSON files
    with open(file_1, "r") as f1, open(file_2, "r") as f2:
        data_1 = json.load(f1)
        data_2 = json.load(f2)

    # Combine items by index
    combined = []
    for item_1, item_2 in zip(data_1, data_2):
        combined.append({**item_1, **item_2})

    # Save to a new file
    with open(result_name, "w") as fout:
        json.dump(combined, fout, indent=2)
