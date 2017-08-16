from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from utils import label_encoding, normalize_features
import pandas as pd
import numpy as np
import pickle
import csv
import sys

DATA_DIR = './data/'


def logistic_regression(X_test):
    """
    Implement forward logistic regression

    :param X_test: the training test
    :return: the predictions for the test set
    """

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
    """
    Get the testing data from the csv file and clean nan values

    :return:
    """

    test_csv = pd.read_csv(DATA_DIR + 'test.csv')

    test_csv['Cabin'] = test_csv['Cabin'].fillna('C0')
    test_csv['Embarked'] = test_csv['Embarked'].fillna('0')
    test_csv['Age'] = test_csv['Age'].fillna(test_csv['Age'].mean())
    test_csv['Fare'] = test_csv['Fare'].fillna(test_csv['Fare'].mean())
    test_csv = label_encoding(test_csv, ['Sex', 'Ticket', 'Cabin', 'Embarked'])

    X_test = test_csv[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]

    normalize_features(X_test)

    return X_test.as_matrix(), test_csv['PassengerId']

def predict():
    """
    Predict the ouptut values from the test set and write
    the predictions in a csv files. Also compute the accuracy score

    :return:
    """

    X_test, PassengerId = get_testing_data()
    preds = logistic_regression(X_test)

    gender_submission_csv = pd.read_csv(DATA_DIR + 'gender_submission.csv')

    print(accuracy_score(list(gender_submission_csv['Survived']), preds))
    with open('./predictions.csv', 'w') as csvfile:
        csvfile.write('PassengerId,Survived\n')
        for pred, id in zip(preds, PassengerId):
            csvfile.write(str(id) + ',' + str(pred) + '\n')


if __name__ == "__main__":

    predict()
