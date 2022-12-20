# Test forward-forward algorithm in pytorch

This project aims at testing the performance of the Forward-Forward algorithm in PyTorch. The Forward-Forward algorithm is a method for training deep neural networks that is based on the idea of using two forward passes on positive and negative data for training the network. Differently respect to the backpropagation approach the Forward-Forward do not need to compute the gradient of the loss function with respect to the parameters of the network. On the contrary each optimization step can be performed locally, meaning that each layer weights can be updated just after the layer has performed its own forward pass.

This project uses the opensource implementation of the forward-forward algorithm developed by [nebuly-ai](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/forward_forward).

## Installation
It is necessary to install both Pytorch and the forward-forward app. The forward-forward app can be installed from source code. First you have to clone the repository and navigate to the app directory:

```bash
git clone https://github.com/nebuly-ai/nebullvm.git
cd nebullvm/apps/accelerate/forward_forward
```

Then you can install the app with the following command:

```bash 
pip install .
```

Pytorch can be installed following the command line instructions on the [official website](https://pytorch.org/get-started/locally/).

## Usage

The main script is `profile_memory.py`. It can be used to profile the memory usage of the forward-forward algorithm. The script can be run with the following command:

```bash
python profile_memory.py
```

The results will be stored in a json file that can be analysed using the attached notebook.

## Results

As shown in the figure below the memory usage of the forward-forward algorithm is significantly lower than the one of the backpropagation algorithm for deeper models. The memory usage of the forward-forward algorithm is still increases respect to the number of layers, but significantly less respect to the backpropagation algorithm. This is due to the fact that the increase in memory usage for forward-forward algorithm is related just to the number of parameters of the network, while for the backpropagation algorithm the memory usage is related to the number of parameters and the number of layers (since activations must be saved for computing the gradients).

![histogram](https://user-images.githubusercontent.com/38586138/208696596-45a8d0e2-c682-4f69-8e89-399bb1fb8bbf.png)

Actually, the Forward-Forward algorithm could in practice be further optimized, since it is not necessary to load the full network while training. In fact, the Forward-Forward algorithm can be used to train each layer of the network separately, meaning that the memory usage of the algorithm would be related just to the number of parameters of the layer being trained.
