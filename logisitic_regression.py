import numpy as np
import random
import pickle

def fit_logistic_regression(X_train, Y_train):
    """
    Implement non vectorized logistic regression -
    Forward and backward pass

    :param X_train: the training features set
    :param Y_train: the label training set
    :return:
    """

    # Hyperparameters initialization
    lr = 0.05

    # Parameters initialization
    weights = np.random.normal(0, 0.1, [9, 1])
    biais = random.normalvariate(0, 0.1)

    m = X_train.shape[0]
    for epoch in range(3000):

        # Forward pass
        Z = np.dot(X_train, weights) + biais
        A = 1 / (1 + np.exp(-Z))

        # Loss Computation
        J = np.sum(-(Y_train * np.log(A) + (1 - Y_train) * np.log(1 - A))) / m
        # Gradient computation
        dZ = A - Y_train

        dw = np.dot(X_train.T, dZ) / m
        db = np.sum(dZ) / m

        # Update weights
        weights = weights - lr * dw
        biais = biais - lr * db

        if epoch % 10 == 0:
            print("epoch %s - loss %s" % (epoch, J))

    weights_file = open('./models/logistic_weights.pkl', 'w')
    biais_file = open('./models/logistic_biais.pkl', 'w')
    pickle.dump(weights, weights_file)
    pickle.dump(biais, biais_file)

def predict_logistic_regression(X_test):
    """
    Implement forward logistic regression

    :param X_test: the training test
    :return: the predictions for the test set
    """

    weights = pickle.load(open('./models/logistic_weights.pkl', 'r'))
    biais = pickle.load(open('./models/logistic_biais.pkl', 'r'))

    Z = np.dot(X_test, weights) + biais
    A = 1 / (1 + np.exp(-Z))

    A[A > 0.5] = 1
    A[A <= 0.5] = 0

    return A
