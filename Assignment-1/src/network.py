import numpy as np
import data
import time
import tqdm
import math

"""
NOTE
----
Start by implementing your methods in non-vectorized format - use loops and other basic programming constructs.
Once you're sure everything works, use NumPy's vector operations (dot products, etc.) to speed up your network.
"""
epsilon = 1e-7

def sigmoid(a):
    return (1-epsilon) / (1 + np.exp(-1 * a))


def softmax(a):
    e = np.exp(a)
    return e / e.sum()


def binary_cross_entropy_cost(w, b, X, Y):
#     print("yooooooo",w.shape)
    sigmoid_coeff = (np.dot(X, w)) + b
    val = sigmoid(sigmoid_coeff)

    J = -1 * (np.multiply(Y, np.log(val)) + np.multiply((1 - Y), np.log(1 - val)))
    cost = np.sum(J) / X.shape[0]
    cost = np.squeeze(cost)
    return cost


def multiclass_cross_entropy_cost(w, b, X, y):

    sigmoid_coeff = (np.dot(X, w)) + b
    val = sigmoid(sigmoid_coeff)

    return -np.mean(np.log(val[np.arange(len(y)), y]))


def gradient_computation_logistic(w, b, X, Y):

    sigmoid_coeff = (np.dot(X, w)) + b
    A = sigmoid(sigmoid_coeff)

    difference = np.reshape(Y,(len(Y),1)) - A
    dw = np.dot(X.transpose(), difference) / X.shape[0]
    db = np.squeeze(np.sum(difference, axis=0)) / X.shape[0]
    return [dw, db]


class Network:
    global w_best

    def __init__(self, hyperparameters, activation, loss):
        self.hyperparameters = hyperparameters
        self.activation = activation
        self.loss = loss
        self.learning_rate = hyperparameters.learning_rate
        self.epsilon = epsilon
        self.weights = np.ones((hyperparameters.in_dim, hyperparameters.out_dim))
        self.bias = 0

    def forward(self, X):
        sigmoid_coeff = (np.dot(X, self.weights)) + self.bias
        return np.where(sigmoid(sigmoid_coeff) > 0.5, 1, 0)

    def __call__(self, X):
        return self.forward(X)

    def train(self, train_data, val_data, test_data):
        X, y = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        w_best = self.weights
        b_best = self.bias
    
        total_cost = 0
        training_costs = []
        validation_costs = []
        testing_costs = []
        prev_val_cost = float("inf")
        
        training_iterations = 0
        
        for epoch in range(self.hyperparameters.epochs):
            [dw, db] = gradient_computation_logistic(self.weights, self.bias, X, y)
            training_cost = binary_cross_entropy_cost(self.weights, self.bias, X, y)
#             print("training",dw.shape)
            self.weights =  self.weights+ self.learning_rate * dw
            self.bias = self.bias + self.learning_rate * db
#             print("training",self.weights.shape)
            validation_cost = binary_cross_entropy_cost(self.weights, self.bias, X_val, y_val)
            testing_cost = binary_cross_entropy_cost(self.weights, self.bias, X_test, y_test)
#             if(epoch%10 == 9):
#                 print(validation_cost)
 
            training_costs.append(training_cost)
            validation_costs.append(validation_cost)
            testing_costs.append(testing_cost)

            if validation_cost < prev_val_cost:
                prev_val_cost = validation_cost
                w_best = self.weights
                b_best = self.bias
                training_iterations += 1
            else:
                print('EARLY STOPPING: at iteration = {}'.format(epoch))
                training_iterations = epoch
                break

        self.weights =  w_best
        self.bias = b_best 
        
        val_accuracy = self.test(val_data)
        test_accuracy = self.test(test_data)

    
        return test_accuracy,validation_cost

    def test(self, minibatch):
        """
    Test the network on the given minibatch

    Use `self.weights` and `self.activation` to compute the network's output
    Use `self.loss` to compute the loss.
    Do NOT update the weights in this method!

    Parameters
    ----------
    minibatch
        The minibatch to iterate over
    """
        X, y = minibatch
        y = y.reshape(y.shape[0],1)
        y_test = self.forward(X)
        return  sum(y == y_test) / len(y) * 100

        pass

