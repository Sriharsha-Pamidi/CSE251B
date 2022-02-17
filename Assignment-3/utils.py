import numpy as np
# import tensorflow as tf
# from keras import *

def iou(pred, target, n_classes = 10):
    ious = []
    #pred = pred.view(-1)
    #target = target.view(-1)

  # Ignore IoU for undefined class ("9")
    for cls in range(n_classes-1):  # last class is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        #intersection = keras.backend.sum(keras.backend.abs(pred_inds*target_inds), axis = [1,2,3]) #complete this
        #union = keras.backend.sum(pred_inds, [1,2,3]) + keras.backend.sum(target_inds, [1,2,3]) - intersection #complete this
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
            #ious = keras.backend.mean((intersection + 1) / (union + 1), axis=0) #complete this

    return np.array(ious)

def pixel_acc(pred, target):
    #TODO complete this function, make sure you don't calculate the accuracy for undefined class ("9")
    correct = ((pred == target) & (target != 9)).sum()
    total   = (target != 9).sum()
    return correct / total
    