from utils import label_encoding, normalize_features
import pandas as pd
import numpy as np
import random
import pickle
import sys

DATA_DIR = './data/'

def logistic_regression(X_train, Y_train):
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
    weights = np.random.normal(0, 0.1, 9)
    biais = random.normalvariate(0, 0.1)

    m = X_train.shape[0]
    for epoch in range(300):

        # Forward pass
        Z = np.dot(X_train, weights) + biais
        A = 1 / (1 + np.exp(-Z))

        # Loss Computation
        J = np.sum(-(Y_train * np.log(A) + (1 - Y_train) * np.log(1 - A))) / m

        # Gradient computation
        dZ = A - Y_train
        dw = np.dot(dZ, X_train) / m
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




def get_training_data():
    """
    Get the training data from the csv file and clean nan values

    :return:
    """

    train_csv = pd.read_csv(DATA_DIR + 'train.csv')

    train_csv['Cabin'] = train_csv['Cabin'].fillna('C0')
    train_csv['Embarked'] = train_csv['Embarked'].fillna('0')
    train_csv['Age'] = train_csv['Age'].fillna(train_csv['Age'].mean())
    train_csv = label_encoding(train_csv, ['Sex', 'Ticket', 'Cabin', 'Embarked'])

    X_train = train_csv[['Pclass', 'Sex', 'Age',  'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
    Y_train = train_csv['Survived']

    normalize_features(X_train)

    return X_train.as_matrix(), Y_train.as_matrix()

def train():

    X_train, Y_train = get_training_data()
    logistic_regression(X_train, Y_train)

if __name__ == "__main__":

    train()