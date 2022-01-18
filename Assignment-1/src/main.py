import argparse
import network
from network import Network
import data
from pca import PCA
import numpy as np
# from sklearn.decomposition import PCA
from sklearn import preprocessing


def logistic_regression_classes(hyperparameters, classes):
	X, y = data.load_traffic_data(True, classes)
	
	scaler = preprocessing.StandardScaler().fit(X)
	dataset = (scaler.transform(X), np.array([8-x for x in y]))
	
	# PCA
	pca_instace = PCA(hyperparameters.in_dim)
	pca_instace.fit(dataset[0])
	dataset = (data.append_bias(pca_instace.transform(dataset[0])), dataset[1])
	
	# training
	model = Network(hyperparameters, network.sigmoid, network.binary_cross_entropy, network.logistic_gradient)
	test_accuracy = model.train(dataset)
	# test_accuracy = model.train(dataset, False)
	print("test accuracy-", test_accuracy)
	
	return test_accuracy


def softmax_regression(hyperparameters):
	X, y = data.load_traffic_data(True)
	
	scaler = preprocessing.StandardScaler().fit(X)
	dataset = (scaler.transform(X), data.onehot_encode(y))
	
	# PCA
	pca_instace = PCA(hyperparameters.in_dim)
	pca_instace.fit(dataset[0])
	dataset = (data.append_bias(pca_instace.transform(dataset[0])), dataset[1])
	
	# training
	model = Network(hyperparameters, network.softmax, network.multiclass_cross_entropy, network.softmax_gradient)
	test_accuracy = model.train(dataset)
	print("test accuracy-", test_accuracy)
	
	return test_accuracy


def main(hyperparameters):
	# Q1
	logistic_regression_classes(hyperparameters, [7, 8])
	
	# Q2
	# softmax_regression(hyperparameters)
	
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
parser.add_argument('--k-folds', type=int, default=5,
                    help='number of folds for cross-validation')


hyperparameters = parser.parse_args()
main(hyperparameters)
