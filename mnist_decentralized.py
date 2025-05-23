import time
import utils
from flwr.client import ClientApp
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import NDArrays, Scalar, Metrics, ndarrays_to_parameters, Parameters, Context
from flwr.client import NumPyClient
from collections import OrderedDict
from typing import Dict, Tuple, List
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
import json
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt
from datasets import load_dataset
from flwr.simulation import run_simulation
import os
import warnings
warnings.filterwarnings('ignore')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_rounds", type=int, default=10)
parser.add_argument("--num_partitions", type=int, default=5)
parser.add_argument("--fraction_fit", type=float, default=0.1)
parser.add_argument("--fraction_evaluation", type=float, default=0.25)
parser.add_argument("--fedavgcustom_file", type=str, default="./output/result_1.json")
parser.add_argument("--performance_file", type=str, default="./output/result_2.json")
parser.add_argument("--result_file", type=str, default="./output/result.json")
args = parser.parse_args()

NUM_ROUNDS = args.num_rounds
NUM_PARTITIONS = args.num_partitions
FRACTION_FIT = args.fraction_fit
FRACTION_EVALUATION = args.fraction_evaluation

performance_file = args.fedavgcustom_file
fedavgcustom_file = args.performance_file
result_file = args.result_file
#OUTPUT_FILE = args.output

#NUM_ROUNDS = 5 # Number of rounds of training
#NUM_PARTITIONS = 100 # Number of clients
#FRACTION_FIT = 0.1  # 10% clients sampled each round to do fit()
#FRACTION_EVALUATION = 0.25  # 25% clients sampled each round to do evaluate()


# This caches MNIST locally
mnist = load_dataset("ylecun/mnist", cache_dir="./data/huggingface")
os.environ["HF_DATASETS_CACHE"] = "./data/huggingface"
os.environ["HF_DATASETS_OFFLINE"] = "1"

partitioner = IidPartitioner(num_partitions=NUM_PARTITIONS)
# Let's partition the "train" split of the MNIST dataset
# The MNIST dataset will be downloaded if it hasn't been already
fds = FederatedDataset(dataset="ylecun/mnist", partitioners={"train": partitioner})

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

class FedAvgCustom(FedAvg):
    def __init__(self, file_name: str, num_rounds: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_name = file_name
        self.num_rounds = num_rounds
        self.loss_list = []
        self.metrics_list = [] 
    
    def _make_plot(self):
        """Makes a plot with the results recorded"""
        round = list(range(1, len(self.loss_list) + 1))
        acc = [100.0 * metrics["accuracy"] for metrics in self.metrics_list]
        plt.plot(round, acc)
        plt.grid()
        plt.title("MNIST") 
        plt.ylabel("Test accuracy (%)")
        plt.xlabel("Communication rounds")
        plt.savefig("./output/mnist_accuracy_plot.pdf")  # Or any other filename/path
        plt.close()   

    def evaluate(self, server_round: int, parameters: Parameters):
        """Evaluate model parameters using an evaluation function."""
        loss, metrics = super().evaluate(server_round, parameters)
        # Record results
        self.loss_list.append(loss)
        self.metrics_list.append(metrics)
        # If last round, save results and make a plot
        if server_round == self.num_rounds:

            metadata = {
                "loss": round(loss, 3),
                "metrics": metrics
            }
            #performance_file = './output/mnist_decentralized_performance.json'
            # Save to JSON
            utils.store_result(metadata, performance_file)
            # Generate plot
            self._make_plot()

def get_evaluate_fn(testloader):
    """Return a function that can be called to do global evaluation."""
    def evaluate_fn(server_round: int, parameters, config):
        """Evaluate global model on the whole test set."""
        model = Net(num_classes=10)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # set parameters to the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # call test (evaluate model as in centralised setting)
        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}
    return evaluate_fn

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
        
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes=10)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        """This method trains the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""
        # copy parameters sent by the server into client's local model
        set_params(self.model, parameters)
        # Define the optimizer
        optim = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # do local training (call same function as centralised setting)
        train(self.model, self.trainloader, optim, self.device)
        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return get_params(self.model), len(self.trainloader), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""
        set_params(self.model, parameters)
        # do local evaluation (call same function as centralised setting)
        loss, accuracy = test(self.model, self.valloader, self.device)
        # send statistics back to the server
        return float(loss), len(self.valloader), {"accuracy": accuracy}

# Two auxhiliary functions to set and extract parameters of a model
def set_params(model, parameters):
    """Replace model parameters with those passed as `parameters`."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    # now replace the parameters
    model.load_state_dict(state_dict, strict=True)

def get_params(model):
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def client_fn(context: Context):
    """Returns a FlowerClient containing its data partition."""
    partition_id = int(context.node_config["partition-id"])
    partition = fds.load_partition(partition_id, "train")
    # partition into train/validation
    partition_train_val = partition.train_test_split(test_size=0.2, seed=42)
    trainloader, testloader = utils.get_mnist_dataloaders(partition_train_val, batch_size=32)

    return FlowerClient(trainloader=trainloader, valloader=testloader).to_client()

# ##################   Server ########################
# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context):
    # instantiate the model
    model = Net(num_classes=10)
    ndarrays = get_params(model)
    # Convert model parameters to flwr.common.Parameters
    global_model_init = ndarrays_to_parameters(ndarrays)
    _, testloader = utils.get_mnist_dataloaders(mnist, batch_size=32)
    # Define the strategy
    strategy = FedAvgCustom(
        file_name="./output/results_fedavgcustom",
        num_rounds=NUM_ROUNDS,
        fraction_fit= FRACTION_FIT,  # 10% clients sampled each round to do fit()
        fraction_evaluate=FRACTION_EVALUATION,  # 25% clients sample each round to do evaluate()
        evaluate_metrics_aggregation_fn=weighted_average,  # callback defined earlier
        initial_parameters=global_model_init,  # initialised global model
        evaluate_fn=get_evaluate_fn(
            testloader  # gloabl evaluation (here we can pass the same testset as used in centralised)
        ),  
    )     
    # Construct ServerConfig
    config = ServerConfig(num_rounds=NUM_ROUNDS)    
    # Wrap everything into a `ServerAppComponents` object
    return ServerAppComponents(strategy=strategy, config=config)

###################################################################
# Concstruct the ClientApp
client_app = ClientApp(client_fn=client_fn)
# Create your ServerApp
server_app = ServerApp(server_fn=server_fn)

start_time = time.time()
run_simulation(
    server_app=server_app, client_app=client_app, num_supernodes=NUM_PARTITIONS
)

# Save the results to json
# Define metadata
metadata = {
    "run_time": f"{round(time.time() - start_time, 2)} seconds",
    "num_clients": NUM_PARTITIONS,
    "num_rounds": NUM_ROUNDS,
    "fraction_fit": FRACTION_FIT,
    "fraction_evaluate": FRACTION_EVALUATION,
}

utils.store_result(metadata, fedavgcustom_file)
utils.combined_result(fedavgcustom_file, performance_file, result_file)



