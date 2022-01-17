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
min_epsilon = -20
max_epsilon = 20
def sigmoid(X,w,b):
    a = (np.dot(X, w)) + b

    a[a>max_epsilon] = max_epsilon
    a[a<min_epsilon] = min_epsilon
    return (1-epsilon)/(1 + np.exp(-1 * a))


def softmax(a):
    e = np.exp(a)
    return e / e.sum()


def binary_cross_entropy(w, b, X, Y):

    val = sigmoid(X,w,b)
    J = -1 * (np.multiply(Y, np.log(val)) + np.multiply((1 - Y), np.log(1 - val)))
    cost = np.sum(J) / X.shape[0]
    return cost


def multiclass_cross_entropy_cost(w, b, X, y):
    val = sigmoid(X,w,b)
    return -np.mean(np.log(val[np.arange(len(y)), y]))


def logistic_gradient(w, b, X, Y):
    A = sigmoid(X,w,b)
    difference = np.reshape(Y,(len(Y),1)) - A
    dw = -( np.dot(X.transpose(), difference) / X.shape[0])
    db = -( np.sum(difference)/X.shape[0])
    return [dw, db]


class Network:
    global w_best
    def __init__(self, hyperparameters, activation, loss,gradient):
        self.hyperparameters = hyperparameters
        self.activation = activation
        self.loss = loss
        self.gradient = gradient
        self.learning_rate = hyperparameters.learning_rate
        self.epsilon = epsilon
        self.weights = np.random.normal(0,5,size=(hyperparameters.in_dim, hyperparameters.out_dim))
        self.bias = np.random.normal(0,5)

    def forward(self, X):
        return np.where(self.activation(X,self.weights,self.bias) > 0.5, 1, 0)

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
        

        for epoch in range(self.hyperparameters.epochs):
            [dw,db] = self.gradient(self.weights, self.bias, X, y,)
            self.weights -= dw*self.learning_rate
            self.bias -= db*self.learning_rate

            training_costs.append(self.loss(self.weights, self.bias, X, y))
            validation_costs.append(self.loss(self.weights, self.bias, X_val, y_val))
            testing_costs.append(self.loss(self.weights, self.bias, X_test, y_test))

            validation_cost = self.loss(self.weights, self.bias, X_val, y_val)

            if validation_cost < prev_val_cost:
                prev_val_cost = validation_cost
                w_best = self.weights
                b_best = self.bias
            else:
                print('EARLY STOPPING: at iteration = {}'.format(epoch))
                print(validation_cost)
                break

        self.weights = w_best
        self.bias = b_best 
        train_accuracy = self.test(train_data)
        val_accuracy = self.test(val_data)
        test_accuracy = self.test(test_data)
        print(train_accuracy,val_accuracy,test_accuracy)

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
        y_pred = self.forward(X)
        return sum(y == y_pred) / len(y) * 100

        pass

