# Introduction
Defines a CNN model architecture designed for both centralized training and federated learning, where multiple clients train locally and a central server aggregates the updates.

Using the MNIST dataset +  Pytorch Framework + Flower for federated learning.

# How to use
The following steps are necessary to reproduce the results

```bash
# Inside the project's root directory
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/active

# Install dependencies
pip install -r requirements.txt

# Create the output directories
mkdir -p output
mkdir -p data

# Run centralized machine learning
python mnist_centralized.py

# Run vertical federated learning
python mnist_decentralized.py

# Run a set of configuration strategies together on federated learning
python run_experiments.py --select_config=1 && \
python run_experiments.py --select_config=2 && \
python run_experiments.py --select_config=3 && \
python run_experiments.py --select_config=4 && \
python run_experiments.py --select_config=5

# Visualize the results
# jupyter notebook

# Deactivate virtual environment
deactivate
```
