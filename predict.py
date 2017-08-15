from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle
import csv
import sys

DATA_DIR = './data/'

def label_encoding(dataframe, labels):

    le = LabelEncoder()
    for label in labels:
        le.fit(dataframe[label])
        dataframe[label] = le.transform(dataframe[label])

    return dataframe

def normalize_features(X):

    for features in X:
        feats = X[features].tolist()
        mean = np.mean(feats)
        std = np.std(feats)
        feats = (feats - mean)/std
        X[features] = feats

    return X

def logistic_regression(X_test):

    weights = pickle.load(open('./models/logistic_weights.pkl', 'r'))
    biais = pickle.load(open('./models/logistic_biais.pkl', 'r'))

    preds = []
    for feats in X_test:

        z = np.dot(feats, weights) + biais
        a = 1 / (1 + np.exp(-z))

        if a > 0.5:
            preds.append(1)
        elif a <= 0.5:
            preds.append(0)

    return preds


def get_testing_data():

    test_csv = pd.read_csv(DATA_DIR + 'test.csv')

    print(test_csv.info())
    print(test_csv.head())

    test_csv['Cabin'] = test_csv['Cabin'].fillna('C0')
    test_csv['Embarked'] = test_csv['Embarked'].fillna('0')
    test_csv['Age'] = test_csv['Age'].fillna(test_csv['Age'].mean())
    test_csv['Fare'] = test_csv['Fare'].fillna(test_csv['Fare'].mean())
    test_csv = label_encoding(test_csv, ['Sex', 'Ticket', 'Cabin', 'Embarked'])

    X_test = test_csv[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]

    normalize_features(X_test)

    return X_test.as_matrix(), test_csv['PassengerId']

def predict():

    X_test, PassengerId = get_testing_data()
    preds = logistic_regression(X_test)

    with open('./predictions.csv', 'w') as csvfile:
        csvfile.write('PassengerId,Survived\n')
        for pred, id in zip(preds, PassengerId):
            csvfile.write(str(id) + ',' + str(pred) + '\n')


if __name__ == "__main__":

    predict()