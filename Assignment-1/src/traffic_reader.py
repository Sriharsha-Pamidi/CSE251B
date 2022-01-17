import pickle
import numpy as np

def load_traffic(path, kind = 'train', subclass = None):
    t_file = f"../{path}/train" + "_wb_" + kind + ".p"
    
    """Load traffic data from `path`"""
    with open(t_file, mode='rb') as f:
        train = pickle.load(f)
    
    images, labels = train['features'], train['labels']
    images = images.reshape((images.shape[0], -1))

#     ####added
#     images, labels = train['features'], train['labels']
#     output = filter(lambda x : x[1] == 7 or x[1] == 8 , list(zip(images,labels)))
#     i=-1
#     images=[]
#     labels = []
#     for x in list(output):
#         images += [x[0]]
#         labels += [0] if (x[1] == 7) else [1]
    
#     images = np.array(images)
#     labels = np.array(labels)
    
#     images = images.reshape((images.shape[0], -1))
    
#     ####added
    
    return images, labels
