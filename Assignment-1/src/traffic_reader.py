import pickle


def load_traffic(path, kind='train', subclass=None):
    t_file = f"../{path}/train" + "_wb_" + kind + ".p"

    """Load traffic data from `path`"""
    with open(t_file, mode='rb') as f:
        train = pickle.load(f)

    images, labels = train['features'], train['labels']
    images = images.reshape((images.shape[0], -1))

    return images, labels
