from utils import label_encoding, normalize_features
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import random
import pickle
import sys

DATA_DIR = './data/'

def backward(a1, a2, a3, z1, z2, z3, weights_1, weights_2, weights_3, X, Y):

    m = X.shape[0]

    dw3 = np.dot(a2.T, (a3 - Y))
    db3 = np.sum(a3 - Y) / m

    dw2 = np.dot((a3 - Y), weights_3.T)
    da2dz2 = np.full([Y.shape[0], 20], 0.01)
    da2dz2[z2 > 0] = 1
    dw2 = dw2 * da2dz2
    dw2 = np.dot(a1.T, dw2)

    db2 = np.dot((a3 - Y), weights_3.T)
    db2 = np.sum(db2 * da2dz2, 0) / m

    dw1 = np.dot((a3 - Y), weights_3.T)
    da2dz2 = np.full([Y.shape[0], 20], 0.01)
    da2dz2[z2 > 0] = 1
    dw1 = dw1 * da2dz2
    dw1 = np.dot(dw1, weights_2.T)
    da1dz1 = np.full([Y.shape[0], 20], 0.01)
    da1dz1[z1 > 0] = 1
    dw1 = dw1 * da1dz1
    dw1 = np.dot(X.T, dw1)

    db1 = np.dot((a3 - Y), weights_3.T)
    db1 = db1 * da2dz2
    db1 = np.dot(db1, weights_2.T)
    db1 = np.sum(db1 * da1dz1, 0) / m

    return dw1, db1, dw2, db2, dw3, db3


def forward(X, weights_1, weights_2, weights_3, biais_1, biais_2, biais_3):

        z1 = np.dot(X, weights_1) + biais_1
        a1 = np.maximum(np.full(z1.shape, 0.01), z1)

        z2 = np.dot(a1, weights_2) + biais_2
        a2 = np.maximum(np.full(z2.shape, 0.01), z2)

        z3 = np.dot(a2, weights_3) + biais_3
        z3 = np.clip(z3, -100, 100)
        a3 = 1 / (1 + np.exp(-z3))

        return a1, a2, a3, z1, z2, z3

def predict_ann(X):

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

    lr = 0.00003
    m = X.shape[0]

    weights_1 = np.random.normal(0, 0.1, size=(9, 20))
    biais_1 = np.random.normal(0, 0.1, 20)
    weights_2 = np.random.normal(0, 0.1, size=(20, 20))
    biais_2 = np.random.normal(0, 0.1, 20)
    weights_3 = np.random.normal(0, 0.1, size=(20, 1))
    biais_3 = np.random.normal(0, 0.1, 1)

    for epoch in range(3000):

        a1, a2, a3, z1, z2, z3 = forward(X, weights_1, weights_2, weights_3, biais_1, biais_2, biais_3)
        dw1, db1, dw2, db2, dw3, db3 = backward(a1, a2, a3, z1, z2, z3, weights_1, weights_2, weights_3, X, Y)

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