# Introduction
Provides a CNN model architecture used for both centralized training and distributed training across multiple clients, aggregated by a single server following the traditional Federated Learning approach.

Using MNIST dataset + Flower + Pytorch Framework

# How to use
## run centralized machine learning
`python mnist_centralized.py`

## run traditional federated learning
`python mnist_decentralized.py`

## run a set of configuration strategies together on traditional federated learning
```bash
python run_experiments.py --select_config=1 && \
python run_experiments.py --select_config=2 && \
python run_experiments.py --select_config=3 && \
python run_experiments.py --select_config=4 && \
python run_experiments.py --select_config=5
```