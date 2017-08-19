from utils import label_encoding, normalize_features
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import random
import pickle
import sys

DATA_DIR = './data/'

def backward(a1, a2, z1, z2, weights_1, weights_2, X_train, Y_train):

    m = X_train.shape[0]

    dw2 = np.dot(a1.T, (a2 - Y_train))
    db2= np.sum(a2 - Y_train) / m

    dw1 = np.dot((a2 - Y_train), weights_2.T)
    da1dz1 = np.full([Y_train.shape[0], 20], 0.01)
    da1dz1[z1 > 0] = 1
    dw1 = dw1 * da1dz1
    dw1 = np.dot(X_train.T, dw1)

    db1 = np.dot((a2 - Y_train), weights_2.T)
    db1 = np.sum(db1 * da1dz1, 0) / m

    return dw1, db1, dw2, db2


def forward(X, weights_1, weights_2, biais_1, biais_2):

        z1 = np.dot(X, weights_1) + biais_1
        a1 = np.maximum(np.full(z1.shape, 0.01), z1)

        z2 = np.dot(a1, weights_2) + biais_2
        z2 = np.clip(z2, -100, 100)
        a2 = 1 / (1 + np.exp(-z2))

        # z3 = np.dot(a2, weights_3) + biais_3
        # a3 = 1 / (1 + np.exp(-z3))

        return a1, a2, z1, z2

def predict_ann(X):

    weights_1 = pickle.load(open('./models/ann_weights_1.pkl', 'r'))
    biais_1 = pickle.load(open('./models/ann_biais_1.pkl', 'r'))

    weights_2 = pickle.load(open('./models/ann_weights_2.pkl', 'r'))
    biais_2 = pickle.load(open('./models/ann_biais_2.pkl', 'r'))

    _, preds, _, _ = forward(X, weights_1, weights_2, biais_1, biais_2)

    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    preds = preds[:, 0]

    return preds

def fit_ann(X, Y):

    lr = 0.0002
    m = X.shape[0]

    weights_1 = np.random.normal(0, 0.1, size=(9, 20))
    biais_1 = np.random.normal(0, 0.1, 20)
    weights_2 = np.random.normal(0, 0.1, size=(20, 20))
    biais_2 = np.random.normal(0, 0.1, 20)
    # weights_3 = np.random.normal(0, 0.1, size=(20, 1))
    # biais_3 = np.random.normal(0, 0.1, 1)

    for epoch in range(5000):

        a1, a2, z1, z2 = forward(X, weights_1, weights_2, biais_1, biais_2)
        dw1, db1, dw2, db2 = backward(a1, a2, z1, z2, weights_1, weights_2, X, Y)

        cost = np.sum(- (Y * np.log(a2) + (1 - Y) * np.log(1 - a2))) / m

        weights_1 -= lr * dw1
        biais_1 -= lr * db1
        weights_2 -= lr * dw2
        biais_2 -= lr * db2
        # weights_3 -= lr * dw3
        # biais_3 -= lr * db3

        print("epoch %s - loss %s" % (epoch, cost))

    weights_1_file = open('./models/ann_weights_1.pkl', 'w')
    biais_1_file = open('./models/ann_biais_1.pkl', 'w')
    pickle.dump(weights_1, weights_1_file)
    pickle.dump(biais_1, biais_1_file)

    weights_2_file = open('./models/ann_weights_2.pkl', 'w')
    biais_2_file = open('./models/ann_biais_2.pkl', 'w')
    pickle.dump(weights_2, weights_2_file)
    pickle.dump(biais_2, biais_2_file)