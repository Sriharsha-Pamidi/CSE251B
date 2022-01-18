import random
import numpy as np

from traffic_reader import load_traffic


def load_traffic_data(aligned = True, subclass = None):
	if aligned:
		return load_traffic('data', kind='aligned', subclass=subclass)
	return load_traffic('data', kind='unaligned', subclass=subclass)


def z_score_normalize(X, u = None, sd = None):
	return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def min_max_normalize(X, _min = None, _max = None):
	return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))


def onehot_encode(y):
	return np.eye(np.max(y) + 1)[y]


def onehot_decode(y):
	return np.argmax(y, axis=1)


def shuffle(dataset):
	return random.shuffle(dataset)


def append_bias(X):
	ones_matrix = np.ones((X.shape[0], 1))
	X = np.concatenate((X, ones_matrix), axis=1)
	return X


def generate_minibatches(dataset, batch_size = 64):
	X, y = dataset
	l_idx, r_idx = 0, batch_size
	while r_idx < len(X):
		yield X[l_idx:r_idx], y[l_idx:r_idx]
		l_idx, r_idx = r_idx, r_idx + batch_size
	
	yield X[l_idx:], y[l_idx:]


def generate_k_fold_set(dataset, k = 5):
	X, y = dataset
	order = np.random.permutation(len(X))
	fold_width = len(X) // k
	l_idx, r_idx = 0, 2 * fold_width
	for i in range(k):
		train = np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]), np.concatenate(
			[y[order[:l_idx]], y[order[r_idx:]]])
		validation = X[order[l_idx:r_idx]], y[order[l_idx:r_idx]]
		test = X[order[l_idx + fold_width:r_idx]], y[order[l_idx + fold_width:r_idx]]
		yield train, validation, test
		l_idx, r_idx = r_idx, r_idx + 2 * fold_width


def generate_split_dataset(dataset, split):
	X, Y = dataset
	l_idx, r_idx = int(0.8 * len(X)), int(0.9 * len(X))
	
	train = np.array(X)[:l_idx], np.array(Y)[:l_idx]
	validation = X[l_idx:r_idx], Y[l_idx:r_idx]
	test = X[r_idx:], Y[r_idx:]
	
	train_X, train_Y = train
	holdout_X, holdout_Y = validation
	test_X, test_Y = test
	
	return (train_X, train_Y), (holdout_X, holdout_Y), (test_X, test_Y)
