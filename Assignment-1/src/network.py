import numpy as np
import data
import pickle
from sklearn.metrics import confusion_matrix

epsilon = 1e-7
min_epsilon = -20
max_epsilon = 20
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
	val = sigmoid(X, w)
	J = -1 * (np.multiply(Y, np.clip(np.log(val), -100, 100)) + np.multiply((1 - Y),
	                                                                        np.clip(np.log(1 - val), -100, 100)))
	cost = np.sum(J) / X.shape[0]
	return cost


def multiclass_cross_entropy(w, X, y):
	val = softmax(X, w)
	return -np.sum(np.multiply(y, np.clip(np.log(val), -100, 100))) / X.shape[0]


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


def stochastic_softmax_gradient(w, x, y):
	A = np.exp(np.dot(x, w)) / np.sum(np.exp(np.dot(x, w)))
	difference = y - A
	dw = -(np.multiply(x.reshape(x.shape[0], 1), difference) / x.shape[0])
	return dw


class Network:
	
	def __init__(self, hyperparameters, activation, loss, gradient):
		self.hyperparameters = hyperparameters
		self.activation = activation
		self.loss = loss
		self.gradient = gradient
		self.learning_rate = hyperparameters.learning_rate
		self.epsilon = epsilon
		self.weights = np.zeros((self.hyperparameters.in_dim + 1, self.hyperparameters.out_dim))
	
	def forward(self, X):
		if self.activation == sigmoid:
			return np.where(self.activation(X, self.weights) > 0.5, 1, 0)
		
		probabilties = self.activation(X, self.weights)
		probabilties[probabilties != (np.max(probabilties, axis=1)).reshape(X.shape[0], 1)] = 0
		probabilties[probabilties == (np.max(probabilties, axis=1)).reshape(X.shape[0], 1)] = 1
		return probabilties
	
	def __call__(self, X):
		return self.forward(X)
	
	def train_stochastic(self, dataset, aligned):
		train_k_cost = []
		validation_k_cost = []
		# test_k_cost = []
		
		# train_k_best_acc = []
		# validation_k_best_acc = []
		test_k_best_acc = []
		
		train_k_acc = []
		validation_k_acc = []
		# test_k_acc = []
		best_weights = []
		w_best = self.weights
		for k_fold_num in range(self.hyperparameters.k_folds):
			self.weights = np.zeros((self.hyperparameters.in_dim + 1, self.hyperparameters.out_dim))
			train, valid, test = next(data.generate_k_fold_set(dataset, self.hyperparameters.k_folds))
			
			X, y = train
			X_val, y_val = valid
			X_test, y_test = test
			
			train_epoch_costs = []
			validation_epoch_costs = []
			# test_epoch_costs = []
			
			train_epoch_acc = []
			validation_epoch_acc = []
			# test_epoch_acc = []
			
			train_epoch_best_acc = 0
			validation_best_epoch_acc = 0
			test_best_epoch_acc = 0
			w_best = self.weights
			
			for epoch in range(self.hyperparameters.epochs):
				print("epoch", epoch)
				w_best = self.weights
				# gradient calculation and update
				count = 0
				prev_val_cost = float('inf')
				for k in np.random.permutation(X.shape[0]):
					dw = stochastic_softmax_gradient(self.weights, X[k], y[k])
					self.weights = self.weights - self.learning_rate * dw
					count += 1
					if count % 500 == 0:
						validation_cost = self.loss(self.weights, X_val, y_val)
						print(validation_cost)
						if validation_cost < prev_val_cost:
							prev_val_cost = validation_cost
							w_best = self.weights
							train_epoch_best_acc = self.test((X, y))
							validation_best_epoch_acc = self.test((X_val, y_val))
							test_best_epoch_acc = self.test((X_test, y_test))
						else:
							print('early stopping')
						# break
			
				validation_cost = self.loss(self.weights, X_val, y_val)
				
				train_epoch_costs.append(self.loss(self.weights, X, y))
				validation_epoch_costs.append(validation_cost)
				# test_epoch_costs.append(self.loss(self.weights, X_test, y_test))
				
				train_epoch_acc.append(self.test((X, y)))
				validation_epoch_acc.append(self.test((X_val, y_val)))
		# test_epoch_acc.append(self.test((X_test, y_test)))
			best_weights.append(w_best)
			print("best values for the fold:", train_epoch_best_acc, validation_best_epoch_acc, test_best_epoch_acc)
			train_k_cost.append(train_epoch_costs)
			validation_k_cost.append(validation_epoch_costs)
			# test_k_cost.append(test_epoch_costs)
			
			# train_k_best_acc.append(train_epoch_best_acc)
			# validation_k_best_acc.append(validation_best_epoch_acc)
			test_k_best_acc.append(test_best_epoch_acc)
			
			train_k_acc.append(train_epoch_acc)
			validation_k_acc.append(validation_epoch_acc)
		# test_k_acc.append(test_epoch_acc)
		
		test_k = None
		best_model = test_k_best_acc.index(min(test_k_best_acc))
		for p in range(best_model):
			train, valid, test_k = next(data.generate_k_fold_set(dataset, self.hyperparameters.k_folds))
		X_test, y_test = test_k
		confusion_m = test_cm(self.activation, best_weights[best_model], X_test, y_test)
		
		filename = f"file_stochastic_{self.activation}_{self.hyperparameters.in_dim}_" \
		           f"{self.hyperparameters.out_dim}_{self.hyperparameters.epochs}_{True}_{aligned}.pkl"
		
		with open(filename, 'w') as f:
			pickle.dump([self.weights, self.hyperparameters.epochs, train_k_cost, validation_k_cost, train_k_acc,
			             validation_k_acc, confusion_m], f)
		
		return filename, sum(test_k_best_acc) / len(test_k_best_acc)
	
	def train(self, dataset, k_fold = True, aligned = True):
		train_k_cost = []
		validation_k_cost = []
		# test_k_cost = []
		
		# train_k_best_acc = []
		# validation_k_best_acc = []
		test_k_best_acc = []
		
		train_k_acc = []
		validation_k_acc = []
		# test_k_acc = []
		
		best_weights = []
		w_best = self.weights
		for k_fold_num in range(self.hyperparameters.k_folds):
			self.weights = np.zeros((self.hyperparameters.in_dim + 1, self.hyperparameters.out_dim))
			train, valid, test = data.generate_split_dataset(dataset, 0.8)
			if k_fold:
				train, valid, test = next(data.generate_k_fold_set(dataset, self.hyperparameters.k_folds))
			
			X, y = train
			X_val, y_val = valid
			X_test, y_test = test
			
			train_epoch_costs = []
			validation_epoch_costs = []
			# test_epoch_costs = []
			
			train_epoch_acc = []
			validation_epoch_acc = []
			# test_epoch_acc = []
			
			train_epoch_best_acc = 0
			validation_best_epoch_acc = 0
			test_best_epoch_acc = 0
			
			w_best = self.weights
			prev_val_cost = float("inf")
			for epoch in range(self.hyperparameters.epochs):
				# gradient calculation and update
				dw = self.gradient(self.weights, X, y)
				self.weights = self.weights - self.learning_rate * dw
				
				# loss calculation
				validation_cost = self.loss(self.weights, X_val, y_val)
				
				train_epoch_costs.append(self.loss(self.weights, X, y))
				validation_epoch_costs.append(validation_cost)
				# test_epoch_costs.append(self.loss(self.weights, X_test, y_test))
				
				train_epoch_acc.append(self.test((X, y)))
				validation_epoch_acc.append(self.test((X_val, y_val)))
				# test_epoch_acc.append(self.test((X_test, y_test)))
				
				if validation_cost < prev_val_cost:
					prev_val_cost = validation_cost
					w_best = self.weights
					train_epoch_best_acc = self.test((X, y))
					validation_best_epoch_acc = self.test((X_val, y_val))
					test_best_epoch_acc = self.test((X_test, y_test))
				else:
					print('early stopping')
				# break
				print(validation_cost)
			# self.weights = w_best
			
			best_weights.append(w_best)
			print("best values for the fold:", train_epoch_best_acc, validation_best_epoch_acc, test_best_epoch_acc)
			train_k_cost.append(train_epoch_costs)
			validation_k_cost.append(validation_epoch_costs)
			# test_k_cost.append(test_best_epoch_acc)
			
			# train_k_best_acc.append(train_epoch_best_acc)
			# validation_k_best_acc.append(validation_best_epoch_acc)
			test_k_best_acc.append(test_best_epoch_acc)
			
			train_k_acc.append(train_epoch_acc)
			validation_k_acc.append(validation_epoch_acc)
			# test_k_acc.append(test_epoch_acc)
			
			if not k_fold:
				break
		
		test_k = None
		best_model = test_k_best_acc.index(min(test_k_best_acc))
		for p in range(best_model):
			train, valid, test_k = next(data.generate_k_fold_set(dataset, self.hyperparameters.k_folds))
		X_test, y_test = test_k
		confusion_m = test_cm(self.activation, best_weights[best_model] ,X_test, y_test)
		
		filename = f"file_{self.activation}_{self.hyperparameters.in_dim}_" \
		           f"{self.hyperparameters.out_dim}_{self.hyperparameters.epochs}_{k_fold}_{aligned}.pkl"
		
		with open(filename, 'wb') as f:
			pickle.dump([self.weights, self.hyperparameters.epochs, train_k_cost, validation_k_cost, train_k_acc, validation_k_acc, confusion_m], f)
		
		return filename, sum(test_k_best_acc) / len(test_k_best_acc)
	
	def test(self, minibatch):
		X, y = minibatch
		if self.activation == sigmoid:
			y = y.reshape(y.shape[0], 1)
		y_pred = self.forward(X)
		if self.activation == sigmoid:
			return np.sum(y == y_pred) / len(y) * 100
		return (np.sum(y * y_pred)) / y.shape[0] * 100


def test_cm(activation, weights, X, y):
	if activation == sigmoid:
		return np.where(sigmoid(X, weights) > 0.5, 1, 0)
	
	probabilties = softmax(X, weights)
	probabilties[probabilties != (np.max(probabilties, axis=1)).reshape(X.shape[0], 1)] = 0
	probabilties[probabilties == (np.max(probabilties, axis=1)).reshape(X.shape[0], 1)] = 1
	y_pred = probabilties
	
	cm = confusion_matrix(y.argmax(axis=1), y_pred.argmax(axis=1))
	return cm
