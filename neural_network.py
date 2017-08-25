from utils import label_encoding, normalize_features
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import random
import pickle
import sys

def backward(a1, a2, a3, z1, z2, weights_2, weights_3, X, Y):
    """
    Compute the backward pass of a 3 hidden layer neural network

    :param a1: the activation output for the first layer
    :param a2: the activation output for the second layer
    :param a3: the activation output for the third layer
    :param z1: the cache output for the first layer
    :param z2: the cache output for the second layer
    :param weights_2: the weights for the second layer
    :param weights_3: the weights for the third layer
    :param X: the training inputs
    :param Y: the training labels
    :return:
    """

    m = X.shape[0]

    dz3 = (a3 - Y)
    dw3 = np.dot(a2.T, dz3) / m
    db3 = np.sum(dz3) / m

    da2 = np.dot(dz3, weights_3.T)
    der_g2 = np.full([Y.shape[0], 20], 0.01)[z2 > 0] = 1
    dz2 = da2 * der_g2
    dw2 = np.dot(a1.T, dz2) / m

    db2 = np.sum(da2, 0) / m

    da1 = np.dot(dz2, weights_2.T)
    der_g1 = np.full([Y.shape[0], 20], 0.01)
    der_g1[z1 > 0] = 1
    dz1 = da1 * der_g1
    dw1 = np.dot(X.T, dz1) / m

    db1 = np.sum(da1, 0) / m

    return dw1, db1, dw2, db2, dw3, db3


def forward(X, weights_1, weights_2, weights_3, biais_1, biais_2, biais_3):
    """
    Compute the forward pass of a 3 hidden layer neural network

    :param X: the training inputs
    :param weights_1: the weights for the first layer
    :param weights_2: the weights for the second layer
    :param weights_3: the weights for the third layer
    :param biais_1: the bias term for the first layer
    :param biais_2: the bias term for the second layer
    :param biais_3: the bias term for the third layer
    :return:
    """

    z1 = np.dot(X, weights_1) + biais_1
    a1 = np.maximum(np.full(z1.shape, 0.01), z1)

    z2 = np.dot(a1, weights_2) + biais_2
    a2 = np.maximum(np.full(z2.shape, 0.01), z2)

    z3 = np.dot(a2, weights_3) + biais_3
    z3 = np.clip(z3, -100, 100)
    a3 = 1 / (1 + np.exp(-z3))

    return a1, a2, a3, z1, z2, z3

def predict_ann(X):
    """
    Make predictions on the inputs

    :param X: the inputs of the networks
    :return:
    """

    weights_1 = pickle.load(open('./models/ann_weights_1.pkl', 'r'))
    biais_1 = pickle.load(open('./models/ann_biais_1.pkl', 'r'))

    weights_2 = pickle.load(open('./models/ann_weights_2.pkl', 'r'))
    biais_2 = pickle.load(open('./models/ann_biais_2.pkl', 'r'))

    weights_3 = pickle.load(open('./models/ann_weights_3.pkl', 'r'))
    biais_3 = pickle.load(open('./models/ann_biais_3.pkl', 'r'))

    _, preds, _, _, _, _ = forward(X, weights_1, weights_2, weights_3, biais_1, biais_2, biais_3)

    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    preds = preds[:, 0]

    return preds

def fit_ann(X, Y):
    """
    Fit a 3 hidden layer neural network

    :param X: the training inputs
    :param Y: the training labels
    :return:
    """

    lr = 0.2
    m = X.shape[0]

    weights_1 = np.random.normal(0, 0.1, size=(9, 20))
    biais_1 = np.random.normal(0, 0.1, 20)
    weights_2 = np.random.normal(0, 0.1, size=(20, 20))
    biais_2 = np.random.normal(0, 0.1, 20)
    weights_3 = np.random.normal(0, 0.1, size=(20, 1))
    biais_3 = np.random.normal(0, 0.1, 1)

    for epoch in range(5000):

        a1, a2, a3, z1, z2, z3 = forward(X, weights_1, weights_2, weights_3, biais_1, biais_2, biais_3)
        dw1, db1, dw2, db2, dw3, db3 = backward(a1, a2, a3, z1, z2, weights_2, weights_3, X, Y)

        cost = np.sum(- (Y * np.log(a3) + (1 - Y) * np.log(1 - a3))) / m

        weights_1 -= lr * dw1
        biais_1 -= lr * db1
        weights_2 -= lr * dw2
        biais_2 -= lr * db2
        weights_3 -= lr * dw3
        biais_3 -= lr * db3

        print("epoch %s - loss %s" % (epoch, cost))

    weights_1_file = open('./models/ann_weights_1.pkl', 'w')
    biais_1_file = open('./models/ann_biais_1.pkl', 'w')
    pickle.dump(weights_1, weights_1_file)
    pickle.dump(biais_1, biais_1_file)

    weights_2_file = open('./models/ann_weights_2.pkl', 'w')
    biais_2_file = open('./models/ann_biais_2.pkl', 'w')
    pickle.dump(weights_2, weights_2_file)
    pickle.dump(biais_2, biais_2_file)

    weights_3_file = open('./models/ann_weights_3.pkl', 'w')
    biais_3_file = open('./models/ann_biais_3.pkl', 'w')
    pickle.dump(weights_3, weights_3_file)
    pickle.dump(biais_3, biais_3_file)