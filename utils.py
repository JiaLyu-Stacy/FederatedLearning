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
