import numpy as np
import matplotlib.pyplot as plt
import data

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
	val = sigmoid(X, w)
	J = -1 * (np.multiply(Y, np.log(val)) + np.multiply((1 - Y), np.log(1 - val)))
	cost = np.sum(J) / X.shape[0]
	
	return cost


def multiclass_cross_entropy(w, X, y):
	val = softmax(X, w)
	return -np.sum(np.multiply(y, np.log(val))) / X.shape[0]


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


def plot_Q2(train_k_cost, validation_k_cost):
	train_k_cost = np.array(train_k_cost)
	mean_train = train_k_cost.mean(axis=1)
	std_train = train_k_cost.std(axis=1)
	
	validation_k_cost = np.array(validation_k_cost)
	mean_validation = validation_k_cost.mean(axis=1)
	std_validation = validation_k_cost.std(axis=1)
	
	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	ax1.set_title('average of costs at each epoch')
	tc_plt, = ax1.plot(mean_train, label='Training Cost')
	vc_plt, = ax1.plot(mean_validation, label='Validation Cost')
	ax1.legend(handles=[tc_plt, vc_plt])
	plt.show()
	
	fig2 = plt.figure()
	ax1 = fig2.add_subplot(111)
	ax1.set_title('standard deviation of costs at each epoch')
	tc_plt, = ax1.plot(std_train, label='Training Cost')
	vc_plt, = ax1.plot(std_validation, label='Validation Cost')
	ax1.legend(handles=[tc_plt, vc_plt])
	plt.show()
	
	# fig3 = plt.figure()
	# ax1 = fig3.add_subplot(111)
	# ax1.set_title('error bar check')
	# tc_plt, = ax1.errorbar([50, 100, 150, 200, 250, 300], mean_train, , label='Training Cost')
	# vc_plt, = ax1.errorbar([50, 100, 150, 200, 250, 300], mean_validation, label='Validation Cost')
	# ax1.legend(handles=[tc_plt, vc_plt])
	# plt.show()


def plot_Q4(train_k_acc,validation_k_acc):

	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	ax1.set_title('accuracy over the k folds')
	tc_plt, = ax1.plot(train_k_acc, label='Training Accuracy')
	vc_plt, = ax1.plot(validation_k_acc, label='Validation Accuracy')
	ax1.legend(handles=[tc_plt, vc_plt])
	plt.show()
	pass


def plot_Q5b(train_epoch_costs,validation_epoch_costs):
	train_epoch_costs = np.array(train_epoch_costs)
	validation_epoch_costs = np.array(validation_epoch_costs)

	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	ax1.set_title('cost for 80-10-10 split')
	tc_plt, = ax1.plot(train_epoch_costs, label='Training cost')
	vc_plt, = ax1.plot(validation_epoch_costs, label='Validation cost')
	ax1.legend(handles=[tc_plt, vc_plt])
	plt.show()
	pass


class Network:
	global w_best
	
	def __init__(self, hyperparameters, activation, loss, gradient):
		self.hyperparameters = hyperparameters
		self.activation = activation
		self.loss = loss
		self.gradient = gradient
		self.learning_rate = hyperparameters.learning_rate
		self.epsilon = epsilon
		self.weights = np.random.normal(init_mue, init_var, size=(hyperparameters.in_dim + 1, hyperparameters.out_dim))
	
	def forward(self, X):
		if self.activation == sigmoid:
			return np.where(self.activation(X, self.weights) > 0.5, 1, 0)
		
		probabilties = self.activation(X, self.weights)
		probabilties[probabilties != (np.max(probabilties, axis=1)).reshape(X.shape[0], 1)] = 0
		probabilties[probabilties == (np.max(probabilties, axis=1)).reshape(X.shape[0], 1)] = 1
		return probabilties
	
	def __call__(self, X):
		return self.forward(X)
	
	def train(self, dataset, k_fold = True):
		
		train_k_cost = []
		validation_k_cost = []
		test_k_cost = []
		
		train_k_best_acc = []
		validation_k_best_acc = []
		test_k_best_acc = []
		
		train_k_acc = []
		validation_k_acc = []
		test_k_acc = []
		
		for k in range(self.hyperparameters.k_folds):
			train, valid, test = data.generate_split_dataset(dataset,0.8)
			if k_fold:
				train, valid, test = next(data.generate_k_fold_set(dataset, self.hyperparameters.k_folds))
			X, y = train
			X_val, y_val = valid
			X_test, y_test = test
			train_epoch_costs = []
			validation_epoch_costs = []
			test_epoch_costs = []
			
			train_epoch_acc = []
			validation_epoch_acc = []
			test_epoch_acc = []
			
			train_epoch_best_acc = 0
			validation_best_epoch_acc = 0
			test_best_epoch_acc = 0

			for epoch in range(self.hyperparameters.epochs):
				w_best = self.weights
				prev_val_cost = float("inf")
				# gradient calculation and update
				dw = self.gradient(self.weights, X, y)
				self.weights = self.weights - self.learning_rate * dw
				
				# loss calculation
				validation_cost = self.loss(self.weights, X_val, y_val)
				
				train_epoch_costs.append(self.loss(self.weights, X, y))
				validation_epoch_costs.append(validation_cost)
				test_epoch_costs.append(self.loss(self.weights, X_test, y_test))
				
				train_epoch_acc.append(self.test((X, y)))
				validation_epoch_acc.append(self.test((X_val, y_val)))
				test_epoch_acc.append(self.test((X_test, y_test)))
				
				if round(validation_cost, 5) < round(prev_val_cost, 5):
					prev_val_cost = validation_cost
					w_best = self.weights
					train_epoch_best_acc = self.test((X, y))
					validation_best_epoch_acc = self.test((X_val, y_val))
					test_best_epoch_acc = self.test((X_test, y_test))
				else:
					break
				
				self.weights = w_best
			
			train_k_cost.append(train_epoch_costs)
			validation_k_cost.append(validation_epoch_costs)
			test_k_cost.append(test_best_epoch_acc)
			
			train_k_best_acc.append(train_epoch_best_acc)
			validation_k_best_acc.append(validation_best_epoch_acc)
			test_k_best_acc.append(test_epoch_acc)
			
			train_k_acc.append(train_epoch_acc)
			validation_k_acc.append(validation_epoch_acc)
			test_k_acc.append(test_epoch_acc)
			
			if not k_fold:
				break
		if k_fold:
			# plot_Q2(train_k_cost, validation_k_cost)
			plot_Q4(train_k_best_acc,validation_k_best_acc)
		else:
			plot_Q5b(train_k_cost[0],validation_k_cost[0])
			
		return sum(train_k_best_acc) / len(train_k_best_acc)
	
	def test(self, minibatch):
		X, y = minibatch
		if self.activation == sigmoid:
			y = y.reshape(y.shape[0], 1)
		y_pred = self.forward(X)
		if self.activation == sigmoid:
			return np.sum(y == y_pred) / len(y) * 100
		return (np.sum(y * y_pred)) / y.shape[0] * 100
		
		pass
