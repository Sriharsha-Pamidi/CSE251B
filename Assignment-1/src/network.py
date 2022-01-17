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
min_epsilon = -25
max_epsilon = 25
init_var = 1
init_mue = 0


def sigmoid(X, w):
	a = np.dot(X, w)
	a[a > max_epsilon] = max_epsilon
	a[a < min_epsilon] = min_epsilon
	return (1 - epsilon) / (1 + np.exp(-1 * a))


def softmax(X, w):
	a = np.dot(X, w)
	e = np.exp(a)
	return e / np.sum(e, axis=1).reshape(X.shape[0], 1)


def binary_cross_entropy(w, X, Y):
	#     sigmoid_coeff = (np.dot(X, w))
	val = sigmoid(X, w)

	J = -1 * (np.multiply(Y, np.log(val)) + np.multiply((1 - Y), np.log(1 - val)))
	cost = np.sum(J) / X.shape[0]

	return cost


def multiclass_cross_entropy(w, X, y):
	val = softmax(X, w)

	return -np.sum(np.multiply(y,np.log(val)))/X.shape[0]


def logistic_gradient(w, X, Y):
	A = sigmoid(X, w)
	difference = np.reshape(Y, (len(Y), 1)) - A
	dw = -(np.dot(X.transpose(), difference) / X.shape[0])
	return dw


def softmax_gradient(w, X, Y):
	A = softmax(X, w)
	difference = Y - A
	dw = -(np.dot(X.transpose(), difference) / X.shape[0])

	return dw


class Network:
    global w_best

    def __init__(self, hyperparameters, activation, loss,gradient):
        self.hyperparameters = hyperparameters
        self.activation = activation
        self.loss = loss
        self.gradient = gradient
        self.learning_rate = hyperparameters.learning_rate
        self.epsilon = epsilon
        self.weights = np.random.normal(init_mue,init_var,size=(hyperparameters.in_dim+1, hyperparameters.out_dim))

    def forward(self, X):
        if(self.activation==sigmoid):
            return np.where(self.activation(X,self.weights)>0.5,1,0)
        
        probabilties = self.activation(X,self.weights)
        probabilties[probabilties!=(np.max(probabilties,axis=1)).reshape(X.shape[0],1)] = 0
        probabilties[probabilties==(np.max(probabilties,axis=1)).reshape(X.shape[0],1)] = 1
        return probabilties

    def __call__(self, X):
        return self.forward(X)

    def train(self, train_data, val_data, test_data):
        X, y = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        w_best = self.weights

    
        total_cost = 0
        training_costs = []
        validation_costs = []
        testing_costs = []
        prev_val_cost = float("inf")

        for epoch in range(self.hyperparameters.epochs):

            ##gradient val and updation
            dw = self.gradient(self.weights, X, y)
            self.weights =  self.weights- self.learning_rate * dw


            ##loss caculation
            validation_cost = self.loss(self.weights, X_val, y_val)

            training_costs.append(self.loss(self.weights, X, y))
            validation_costs.append(validation_cost)
            testing_costs.append(self.loss(self.weights, X_test, y_test))
            if(epoch%1000 == 999):
                print(validation_cost)
            if validation_cost < prev_val_cost:
                prev_val_cost = validation_cost
                w_best = self.weights
            else:
                print('EARLY STOPPING: at iteration = {}'.format(epoch))
                print(validation_cost)
                break

        self.weights =  w_best
        
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
        if(self.activation == sigmoid):
            y = y.reshape(y.shape[0],1)
        y_pred = self.forward(X)
        if(self.activation == sigmoid):
            return  np.sum(y == y_pred) / len(y) * 100
        return  (np.sum(y*y_pred)) / y.shape[0] * 100

        pass

