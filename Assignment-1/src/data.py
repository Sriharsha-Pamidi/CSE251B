import random
import numpy as np

from traffic_reader import load_traffic


def traffic_sign(aligned = True):
	if aligned:
		return load_traffic('data', kind='aligned')
	return load_traffic('data', kind='unaligned')


load_data = traffic_sign


def z_score_normalize(X, u = None, sd = None):
	return (X - np.mean(X)) / np.std(X)


def min_max_normalize(X, _min = None, _max = None):
	return (X - np.min(X)) / (np.max(X) - np.min(X))


def onehot_encode(y):
	return np.eye(np.max(y) + 1)[y]


def onehot_decode(y):
	return np.argmax(y, axis=1)


def shuffle(dataset):
	return random.shuffle(dataset)


def append_bias(X):
	pass


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
	
	l_idx, r_idx = 0, 2*fold_width
	
	for i in range(k):
		train = np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]), np.concatenate(
			[y[order[:l_idx]], y[order[r_idx:]]])
		validation = X[order[l_idx:r_idx]], y[order[l_idx:r_idx]]
		test = X[order[l_idx + fold_width:r_idx]], y[order[l_idx + fold_width:r_idx]]
		yield train, validation, test
		l_idx, r_idx = r_idx, r_idx + 2 * fold_width


# def generate_k_distributed_fold_set(dataset, k = 5):
# 	X, y = dataset
#
# 	new_dataset =
# 	for p in set(y):
# 		dataset_n = filter(lambda f: f == p, list(zip(X, y)))
#
# 	pass

#
# if __name__ == '__main__':
# 	pp = load_data(False)
# 	dataset = generate_k_distributed_fold_set(pp)
# 	print(dataset)
#
# 	poi = 0
