import argparse
import network
from network import Network
import data
from pca import PCA
import numpy as np


def run_an_epoch(X_train,y_train,X_valid,y_valid,my_NN):
    train_loss = my_NN.train((X_train,y_train))
    valid_loss = my_NN.test((X_valid,y_valid))
    return train_loss, valid_loss

def runLogisticRegresssion(dataset,hyperparameters):
    my_NN = Network(hyperparameters,network.sigmoid,network.binary_cross_entropy)
    curr_train_loss = float("inf")
    curr_valid_loss = float("inf")
    for i in range(hyperparameters.epochs):
        train_loss,valid_loss = run_an_epoch(X_train,y_train,X_valid,y_valid,my_NN)
        

def main(hyperparameters):
    dataset = data.load_data(False)
    pca_instace = PCA(hyperparameters.in_dim)
    X,y = dataset
    pca_instace.fit(X[:20])
    X_1= pca_instace.fit_transform(X[:10])
    print(hyperparameters.epochs)
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





