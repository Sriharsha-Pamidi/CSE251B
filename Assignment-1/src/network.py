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


def sigmoid(a):
	return 1 / (1 + np.exp(-1 * a))


def softmax(a):
	e = np.exp(a)
	return e / e.sum()


def binary_cross_entropy_cost(w, b, X, Y):
	epsilon = 1e-15
	sigmoid_coeff = (np.dot(X, w)) + b
	val = sigmoid(sigmoid_coeff) + epsilon
	
	J = -1 * (np.multiply(Y, np.log(val)) + np.multiply((1 - Y), np.log(1 - val)))
	cost = np.sum(J, axis=0) / X.shape[0]
	cost = np.squeeze(cost)
	return cost


def multiclass_cross_entropy_cost(w, b, X, y):
	epsilon = 1e-15
	sigmoid_coeff = (np.dot(X, w)) + b
	val = sigmoid(sigmoid_coeff) + epsilon
	
	return -np.mean(np.log(val[np.arange(len(y)), y]))


def gradient_computation_logistic(w, b, X, Y):
	sigmoid_coeff = (np.dot(X, w)) + b
	A = sigmoid(sigmoid_coeff)
	difference = Y - A
	dw = np.dot(X.transpose(), difference) / X.shape[0]
	db = np.squeeze(np.sum(difference, axis=0)) / X.shape[0]
	return [dw, db]


class Network:
	global w_best
    
    def __init__(self, hyperparameters, activation, loss):
		self.hyperparameters = hyperparameters
		self.activation = activation
		self.loss = loss
		
		self.weights = np.ones((hyperparameters.in_dim + 1, hyperparameters.out_dim))
		self.b = 0
	
	def forward(self, X):
		sigmoid_coeff = (np.dot(X, self.weights)) + self.b
		return np.where(sigmoid(sigmoid_coeff) > 0.5, 1, 0)
	
	def __call__(self, X):
		return self.forward(X)
	
	def train(self, train_data, val_data, test_data):
		X, y = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        learning_rate = 1

        w_best = w
        b_best = b
    
        total_cost = 0
        training_costs = []
        validation_costs = []
        testing_costs = []

        training_iterations = 0
        for epoch in range(self.hyperparameters.epochs):
            [dw, db] = gradient_computation_logistic(w, b, X, y)
            training_cost = binary_cross_entropy_cost(w, b, X, y)
            w = w + learning_rate * dw
            b = b + learning_rate * db
            validation_cost = binary_cross_entropy_cost(w, b, X_val, y_val)
            testing_cost = binary_cross_entropy_cost(w, b, X_test, y_test)
            
            training_costs.append(training_cost)
            validation_costs.append(validation_cost)
            testing_costs.append(testing_cost)
            
            if validation_cost < prev_val_cost:
                prev_val_cost = validation_cost
                w_best = w
                b_best = b
                training_iterations += 1
            else:
                print('EARLY STOPPING: at iteration = {}'.format(epoch))
                training_iterations = epoch
                break
            
        Y_prediction_test = forward(w_best, b_best, X_test)
        Y_prediction_val = forward(w_best, b_best, X_val)

        test_accuracy = sum(Y_prediction_test == y_test) / len(Y_prediction_test) * 100
        val_accuracy = sum(Y_prediction_val == y_val) / len(Y_prediction_val) * 100
    
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
			
			pass
