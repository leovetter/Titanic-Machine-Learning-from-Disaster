from logisitic_regression import predict_logistic_regression
from utils import normalize_features, label_encoding
from sklearn.decomposition import PCA
from predict import get_testing_data
from train import get_training_data
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

DATA_DIR = './data/'

def visualize_data():

    X_train, Y_train = get_training_data()

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_train)
    X = X_reduced[:, 0]
    Y = X_reduced[:, 1]

    plt.figure()
    plt.scatter(X, Y, c=Y_train);
    plt.show()

if __name__ == '__main__':

    visualize_data()



