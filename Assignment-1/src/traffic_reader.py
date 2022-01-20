import pickle
import numpy as np


def load_traffic(path, kind = 'train', subclass = None):
	t_file = f"../{path}/train" + "_wb_" + kind + ".p"
	
	"""Load traffic data from `path`"""
	with open(t_file, mode='rb') as f:
		train = pickle.load(f)
	
	images, labels = train['features'], train['labels']
	
	if subclass is not None:
		output = filter(lambda s: s[1] in subclass, list(zip(images, labels)))
		images = []
		labels = []
		for x in list(output):
			images += [x[0]]
			labels += [x[1]]
		
		images = np.array(images)
		labels = np.array(labels)
	
	images = images.reshape((images.shape[0], -1))
	
	return images, labels
