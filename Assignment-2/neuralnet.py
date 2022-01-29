################################################################################
# CSE 251B: Programming Assignment 2
# Winter 2022
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os
import yaml
import numpy as np
import pickle
import random
from matplotlib import pyplot as plt
import time


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open(path + 'config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(inp):
    """
    TODO: Normalize your inputs here to have 0 mean and unit variance.
    """
    return (inp - inp.mean()) / inp.std()


def one_hot_encoding(labels, num_classes=10):
    """
    TODO: Encode labels using one hot encoding and return them.
    """
    encoded_labels = np.array([[1 if label == i else 0 for i in range(num_classes)] for label in labels])
    return encoded_labels


def load_data(path, mode='train'):
    """
    Load CIFAR-10 data.
    """
    def unpickle(file):
        with open(file, 'rb') as fo:
            u_dict = pickle.load(fo, encoding='bytes')
        return u_dict

    cifar_path = os.path.join(path, "cifar-10-batches-py")

    if mode == "train":
        images = []
        labels = []
        for i in range(1, 6):
            images_dict = unpickle(os.path.join(cifar_path, f"data_batch_{i}"))
            data = images_dict[b'data']
            label = images_dict[b'labels']
            labels.extend(label)
            images.extend(data)
        normalized_images = normalize_data(np.array(images))
        one_hot_labels = one_hot_encoding(labels, num_classes=10)
        return np.array(normalized_images), np.array(one_hot_labels)
    elif mode == "test":
        test_images_dict = unpickle(os.path.join(cifar_path, f"test_batch"))
        test_data = test_images_dict[b'data']
        test_labels = test_images_dict[b'labels']
        normalized_images = normalize_data(np.array(test_data))
        one_hot_labels = one_hot_encoding(test_labels, num_classes=10)
        return np.array(normalized_images), np.array(one_hot_labels)
    else:
        val_images_dict = unpickle(os.path.join(cifar_path, f"data_batch_{6}"))
        val_images = val_images_dict[b'data']
        val_labels = val_images_dict[b'labels']
        normalized_images = normalize_data(np.array(val_images))
        one_hot_labels = one_hot_encoding(val_labels, num_classes=10)
        return np.array(normalized_images), np.array(one_hot_labels)


def softmax(x):
    """
    TODO: Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        # >>> sigmoid_layer = Activation("sigmoid")
        # >>> z = sigmoid_layer(a)
        # >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "leakyReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        self.x = a
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

        elif self.activation_type == "leakyReLU":
            return self.leakyReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        grad = None

        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        elif self.activation_type == "leakyReLU":
            grad = self.grad_leakyReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        TODO: Implement the sigmoid activation here.
        """
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        TODO: Implement tanh here.
        """
        return np.tanh(x)

    def ReLU(self, x):
        """
        TODO: Implement ReLU here.
        """
        return np.maximum(0, x)

    def leakyReLU(self, x):
        """
        TODO: Implement leaky ReLU here.
        """
        return np.maximum(0.1 * x, x)

    def grad_sigmoid(self):
        """
        TODO: Compute the gradient for sigmoid here.
        """
        return self.sigmoid(self.x) * (1 - self.sigmoid(self.x))

    def grad_tanh(self):
        """
        TODO: Compute the gradient for tanh here.
        """
        return 1 - np.power(self.tanh(self.x), 2)

    def grad_ReLU(self):
        """
        TODO: Compute the gradient for ReLU here.
        """
        gradient = np.zeros_like(self.x)
        gradient[self.x > 0] = 1
        return gradient

    def grad_leakyReLU(self):
        """
        TODO: Compute the gradient for leaky ReLU here.
        """
        gradient = np.full_like(self.x, 0.1)
        gradient[self.x > 0] = 1
        return gradient


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        # >>> fully_connected_layer = Layer(784, 100)
        # >>> output = fully_connected_layer(input)
        # >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)    # Declare the Weight matrix
        self.b = np.random.randn(1, out_units)    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

        self.delta_w_old = 0  # Save delta w
        self.delta_b_old = 0  # Save delta b

        self.w_min = self.w  # Store the weight matrix
        self.b_min = self.b  # Store the bias

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass through the layer here.
        DO NOT apply activation here.
        Return self.a
        """
        self.x = x
        self.a = np.dot(x, self.w) + self.b
        return self.a
        
    def backward(self, delta):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        scale_size = self.x.shape[0] * 10

        self.d_x = delta.dot(self.w.T)
        self.d_w = -self.x.T.dot(delta) / scale_size
        self.d_b = -delta.sum(axis=0) / scale_size

        return self.d_x

    def update_parameters(self, lr, l2_penalty=0, momentum=None):

        self.d_w = self.d_w + l2_penalty * self.w

        if momentum:
            w_delta = self.d_w + momentum * self.delta_w_old
            b_delta = self.d_b + momentum * self.delta_b_old

            self.w -= lr * w_delta
            self.b -= lr * b_delta

            self.delta_w_old = w_delta
            self.delta_b_old = b_delta

        else:
            self.w += lr * self.d_w
            self.b += lr * self.d_b

    def store_parameters(self):
        self.w_min = self.w
        self.b_min = self.b

    def load_parameters(self):
        self.w = self.w_min
        self.b = self.b_min


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        # >>> net = NeuralNetwork(config)
        # >>> output = net(input)
        # >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable
        self.l2_penalty = config['L2_penalty']
        self.momentum = config['momentum_gamma'] if config['momentum'] else None
        self.lr = config['learning_rate']

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.x = x
        self.targets = targets

        # Forward Path
        Input = x
        for layer in self.layers:
            Input = layer.forward(Input)

        # Softmax Activation
        self.y = softmax(Input)

        if targets is None:
            return self.y
        else:
            loss = self.loss(self.y, targets)
            return self.y, loss

    def loss(self, logits, targets):
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        '''
        scale_size = targets.shape[0]

        loss = -np.sum(np.multiply(targets, np.log(logits))) / scale_size

        # l2 penalty
        if self.l2_penalty:
            for layer in self.layers:
                if isinstance(layer, Layer):
                    loss += (np.sum(layer.w ** 2)) * self.l2_penalty / 2

        return loss

    def backward(self):
        '''
        TODO: Implement backpropagation here.
        Call backward methods of individual layers.
        '''
        delta = self.targets - self.y
        for layer in self.layers[::-1]:
            if isinstance(layer, Layer):
                delta = layer.backward(delta)
            else:
                delta = layer.backward(delta)

    def updata_parameters(self):
        for layer in self.layers[::-1]:
            if isinstance(layer, Layer):
                layer.update_parameters(self.lr, l2_penalty=self.l2_penalty, momentum=self.momentum)

    def store_parameters(self):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.store_parameters()

    def load_parameters(self):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.load_parameters()

    def predict(self, x, targets):
        y = self.forward(x)
        predictions = np.argmax(y, axis=1)
        targets = np.argmax(targets, axis=1)

        return np.mean(predictions == targets)


def generate_batch(x, y, bs=1, shuffle_En=True):
    if shuffle_En:
        index = np.random.permutation(len(x))
    else:
        index = list(range(len(x)))
    for idx in range(0, len(x) - bs + 1, bs):
        index_final = index[idx:idx + bs]
        yield x[index_final], y[index_final]


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """

    epochs = config['epochs']
    bs = config['batch_size']
    early_stop_En = config['early_stop']
    epoch_threshold = config['early_stop_epoch']

    valid_accuracy_max = -float('inf')
    valid_accuracy_decrease = 0
    train_metric = {'epochs': [], 'train_loss': [], 'train_accuracy': [], 'valid_loss': [], 'valid_accuracy': []}

    start_time = time.time()
    for epoch in range(epochs):
        train_loss_batch, train_accuracy_batch = [], []
        for x, y in generate_batch(x_train, y_train, bs=bs, shuffle_En=True):
            train_loss_batch.append(model.forward(x, targets=y)[1])
            model.backward()
            model.updata_parameters()
            train_accuracy_batch.append(model.predict(x, targets=y))

        train_loss = np.mean(np.array(train_loss_batch))
        train_accuracy = np.mean(np.array(train_accuracy_batch))
        valid_loss = model.forward(x_valid, targets=y_valid)[1]
        valid_accuracy = model.predict(x_valid, targets=y_valid)

        if epoch % 10 == 0:
            print('Epoch {}, Time {} seconds'.format(epoch + 1, time.time() - start_time))
            print('Train_loss = {:.4f}, Valid_loss = {:.4f}, Valid_accuracy = {:.4f}'.format(train_loss, valid_loss,
                                                                                             valid_accuracy))

        train_metric['epochs'].append(epoch + 1)
        train_metric['train_loss'].append(train_loss)
        train_metric['train_accuracy'].append(train_accuracy)
        train_metric['valid_loss'].append(valid_loss)
        train_metric['valid_accuracy'].append(valid_accuracy)

        if valid_accuracy > valid_accuracy_max:
            model.store_parameters()
            valid_accuracy_max = valid_accuracy
            valid_accuracy_decrease = 0
        else:
            valid_accuracy_decrease += 1

        if early_stop_En:
            if valid_accuracy_decrease > epoch_threshold:
                break

    return train_metric


def test(model, x_test, y_test):
    """
    TODO: Calculate and return the accuracy on the test set.
    """

    return model.predict(x_test, y_test)


def data_split(x, y, ratio=0.1):
    val_num = int(np.floor(x.shape[0] * ratio))

    alllist = list(range(x.shape[0]))
    
    val_list = random.sample(alllist, val_num)
    
    train_list = [idx for idx in alllist if (idx not in val_list)]
    
    x_train = x[train_list]
    y_train = y[train_list]
    x_val = x[val_list]
    y_val = y[val_list]

    return x_train, y_train, x_val, y_val


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("")

    # Create the model
    model = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./data", mode="train")
    x_test,  y_test = load_data(path="./data", mode="test")
    
    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)

    # TODO: Create splits for validation data here.
    # x_val, y_val = ...
    x_train, y_train, x_valid, y_valid = data_split(x_train, y_train, 0.2)

    # TODO: train the model
    train_metrics = train(model, x_train, y_train, x_valid, y_valid, config)

    # Load parameters with least validation loss
    model.load_parameters()
    
    test_acc = test(model, x_test, y_test)
    print(f'Test_accuracy: {test_acc}')

    # TODO: Plots
    # plt.plot(...)

    plt.figure(1)
    plt.plot(train_metrics['epoches'], train_metrics['train_loss'], label='train')
    plt.plot(train_metrics['epoches'], train_metrics['valid_loss'], label='validation')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(train_metrics['epoches'], train_metrics['train_accuracy'], label='train')
    plt.plot(train_metrics['epoches'], train_metrics['valid_accuracy'], label='validation')
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    