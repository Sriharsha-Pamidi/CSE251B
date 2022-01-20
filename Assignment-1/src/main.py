import argparse
import network
from network import Network
import data
import pickle
from pca import PCA
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


def logistic_regression_classes(hyperparameters, classes, k_fold, alinged):
	X, y = data.load_traffic_data(alinged, classes)
	scaler = preprocessing.StandardScaler().fit(X)
	dataset = (scaler.transform(X), np.array([max(classes) - x for x in y]))
	# PCA
	pca_instace = PCA(hyperparameters.in_dim)
	pca_instace.fit(dataset[0])
	pca_instace.plot_pca_components(4)
	dataset = (data.append_bias(pca_instace.transform(dataset[0])), dataset[1])
	# training
	model = Network(hyperparameters, network.sigmoid, network.binary_cross_entropy, network.logistic_gradient)
	filename, test_accuracy = model.train(dataset, k_fold,alinged)
	print("test accuracy-", test_accuracy)
	return filename


def softmax_regression(hyperparameters, k_fold, alinged, stochastic):
	X, y = data.load_traffic_data(alinged)
	scaler = preprocessing.StandardScaler().fit(X)
	dataset = (scaler.transform(X), data.onehot_encode(y))
	# PCA
	pca_instace = PCA(hyperparameters.in_dim)
	pca_instace.fit(dataset[0])
	dataset = (data.append_bias(pca_instace.transform(dataset[0])), dataset[1])
	# training
	model = Network(hyperparameters, network.softmax, network.multiclass_cross_entropy, network.softmax_gradient)
	if stochastic:
		filename, test_accuracy = model.train_stochastic(dataset, alinged)
	else:
		filename, test_accuracy = model.train(dataset, k_fold, alinged)
	print("test accuracy-", test_accuracy)
	return filename


def plot_k_cost_stddev(train_k_cost, validation_k_cost, epochs):
	train_k_cost = np.array(train_k_cost)
	mean_train = train_k_cost.mean(axis=0)
	std_train = train_k_cost.std(axis=0)
	
	validation_k_cost = np.array(validation_k_cost)
	mean_validation = validation_k_cost.mean(axis=0)
	std_validation = validation_k_cost.std(axis=0)
	
	fig3 = plt.figure()
	ax1 = fig3.add_subplot(111)
	ax1.set_title('Cost with std deviations')
	ax1.set_xlabel('epochs')
	ax1.set_ylabel('cost')
	arr = [y * epochs / 6 for y in range(6)]
	tc_plt = ax1.errorbar([y * epochs / 6 for y in range(6)], [list(mean_train)[int(x)] for x in arr],
	                      [list(std_train)[int(x)] for x in arr], label='Training Cost')
	vc_plt = ax1.errorbar(arr, [list(mean_validation)[int(x)] for x in arr],
	                      [list(std_validation)[int(x)] for x in arr], label='Validation Cost')
	ax1.legend(handles=[tc_plt, vc_plt])
	plt.show()


def plot_k_acc_stddev(train_k_acc, validation_k_acc, epochs):
	train_k_acc = np.array(train_k_acc)
	mean_train = train_k_acc.mean(axis=0)
	std_train = train_k_acc.std(axis=0)
	
	validation_k_acc = np.array(validation_k_acc)
	mean_validation = validation_k_acc.mean(axis=0)
	std_validation = validation_k_acc.std(axis=0)
	
	fig2 = plt.figure()
	ax1 = fig2.add_subplot(111)
	ax1.set_title('Accuracy with std deviations')
	ax1.set_xlabel('epochs')
	ax1.set_ylabel('accuracy')
	arr = [y * epochs / 6 for y in range(6)]
	tc_plt = ax1.errorbar([y * epochs / 6 for y in range(6)], [list(mean_train)[int(x)] for x in arr],
	                      [list(std_train)[int(x)] for x in arr], label='Training Accuracy')
	vc_plt = ax1.errorbar(arr, [list(mean_validation)[int(x)] for x in arr],
	                      [list(std_validation)[int(x)] for x in arr], label='Validation Accuracy')
	ax1.legend(handles=[tc_plt, vc_plt])
	plt.show()


def plot_performance(train_epoch_costs, validation_epoch_costs, train_epoch_acc, validation_epoch_acc):
	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	ax1.set_title('cost for 80-10-10 split')
	ax1.set_xlabel('epochs')
	ax1.set_ylabel('cost')
	tc_plt, = ax1.plot(train_epoch_costs, label='Training cost')
	vc_plt, = ax1.plot(validation_epoch_costs, label='Validation cost')
	ax1.legend(handles=[tc_plt, vc_plt])
	plt.show()
	
	fig2 = plt.figure()
	ax1 = fig2.add_subplot(111)
	ax1.set_title('accuracy for 80-10-10 split')
	ax1.set_xlabel('epochs')
	ax1.set_ylabel('accuracy')
	tc_plt, = ax1.plot(train_epoch_acc, label='Training accuracy')
	vc_plt, = ax1.plot(validation_epoch_acc, label='Validation accuracy')
	ax1.legend(handles=[tc_plt, vc_plt])
	plt.show()


def plot_confusion_matrix(cm):
	import seaborn as sns
	ax = sns.heatmap(cm, annot=True, cmap='Blues')
	plt.show()


def make_plots(file,k_fold):
	with open(file,'rb') as f:
		[weights,epochs,train_k_cost,validation_k_cost,train_k_acc,validation_k_acc,confusion_m] = pickle.load(f)
	
	if k_fold:
		plot_k_cost_stddev(train_k_cost, validation_k_cost, epochs)
		plot_k_acc_stddev(train_k_acc, validation_k_acc, epochs)
	else:
		plot_performance(train_k_cost[0], validation_k_cost[0], train_k_acc[0], validation_k_acc[0])
	plot_confusion_matrix(confusion_m)
	

def main(hyperparameters):
	k_fold = True
	alinged = True
	stochastic = True
	# Q1
	# file1 = logistic_regression_classes(hyperparameters, [2, 3], k_fold, alinged)
	# file2 = logistic_regression_classes(hyperparameters, [19, 20], k_fold, alinged)
	
	# Q2
	file3 = softmax_regression(hyperparameters, k_fold, alinged, stochastic)
	
	# make_plots(file1, k_fold)
	# make_plots(file2, k_fold)
	make_plots(file3, k_fold)
	
	pass


parser = argparse.ArgumentParser(description='CSE251B PA1')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train (default: 150)')
parser.add_argument('--learning-rate', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--z-score', dest='normalization', action='store_const',
                    default=data.min_max_normalize, const=data.z_score_normalize,
                    help='use z-score normalization on the dataset, default is min-max normalization')
parser.add_argument('--in-dim', type=int, default=32 * 32,
                    help='number of principal components to use')
parser.add_argument('--out-dim', type=int, default=43,
                    help='number of outputs')
parser.add_argument('--k-folds', type=int, default=10,
                    help='number of folds for cross-validation')

hyperparameters = parser.parse_args()
main(hyperparameters)
