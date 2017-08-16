import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import pickle
import timeit
import sys

DATA_DIR = './data/'

def logistic_regression(X_train, Y_train):

    lr = 0.5
    J = 0
    dw = np.zeros(9)
    db = 0
    m = X_train.shape[0]

    weights = np.random.normal(0, 0.1, 9)
    biais = random.normalvariate(0,0.1)

    for epoch in range(1000):

        for id, (feats, y) in enumerate(zip(X_train, Y_train)):

            z = np.dot(feats,weights) + biais
            a = 1 / (1 + np.exp(-z))
            J = -(y*np.log(a) + (1-a)*np.log(1-a))
            J = np.sum(-(y * np.log(a) + (1 - y) * np.log(1 - a)))
            dz = a - y

            for i, x in enumerate(feats):
                dw[i] = dw[i] + dz*x
                db += dz

        J /= m
        dw /= m
        db /= m

        weights = weights - lr*dw
        biais = biais - lr*db

        print("epoch %s - loss %s" % (epoch, J))

    weights_file = open('./models/logistic_weights.pkl', 'w')
    biais_file = open('./models/logistic_biais.pkl', 'w')
    pickle.dump(weights, weights_file)
    pickle.dump(biais, biais_file)


def label_encoding(dataframe, labels):

    le = LabelEncoder()
    for label in labels:
        le.fit(dataframe[label])
        dataframe[label] = le.transform(dataframe[label])

    return dataframe

def normalize_features(X_train):

    for features in X_train:
        feats = X_train[features].tolist()
        mean = np.mean(feats)
        std = np.std(feats)
        feats = (feats - mean)/std
        X_train[features] = feats

    return X_train

def get_training_data():

    train_csv = pd.read_csv(DATA_DIR + 'train.csv')

    print(train_csv['Ticket'].value_counts())
    sys.exit('')

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

    start = timeit.timeit()

    logistic_regression(X_train, Y_train)

if __name__ == "__main__":

    train()