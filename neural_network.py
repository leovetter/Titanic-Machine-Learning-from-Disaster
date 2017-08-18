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

    dw2 = np.dot(a2.T, (a2 - Y_train))
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
        a2 = 1 / (1 + np.exp(-z2))

        # z3 = np.dot(a2, weights_3) + biais_3
        # a3 = 1 / (1 + np.exp(-z3))

        return a1, a2, z1, z2

def ann(X, Y):

    lr = 0.00002
    m = X.shape[0]

    weights_1 = np.random.normal(0, 0.1, size=(9, 20))
    biais_1 = np.random.normal(0, 0.1, 20)
    weights_2 = np.random.normal(0, 0.1, size=(20, 20))
    biais_2 = np.random.normal(0, 0.1, 20)
    # weights_3 = np.random.normal(0, 0.1, size=(20, 1))
    # biais_3 = np.random.normal(0, 0.1, 1)

    for epoch in range(3000):

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

    X_train = train_csv[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
    Y_train = train_csv['Survived']

    normalize_features(X_train)

    return X_train.as_matrix().reshape(891, 9), Y_train.as_matrix().reshape(891, 1)

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

def train():

    X_train, Y_train = get_training_data()

    ann(X_train, Y_train)

def predict():

    weights_1 = pickle.load(open('./models/ann_weights_1.pkl', 'r'))
    biais_1 = pickle.load(open('./models/ann_biais_1.pkl', 'r'))

    weights_2 = pickle.load(open('./models/ann_weights_2.pkl', 'r'))
    biais_2 = pickle.load(open('./models/ann_biais_2.pkl', 'r'))

    X_test, PassengerId = get_testing_data()
    _, preds, _, _ = forward(X_test, weights_1, weights_2, biais_1, biais_2)
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    preds = preds[:,0]

    # preds.reshape(418)
    print(preds.shape)

    gender_submission_csv = pd.read_csv(DATA_DIR + 'gender_submission.csv')

    print(accuracy_score(list(gender_submission_csv['Survived']), preds))
    with open('./predictions.csv', 'w') as csvfile:
        csvfile.write('PassengerId,Survived\n')
        for pred, id in zip(preds, PassengerId):
            csvfile.write(str(id) + ',' + str(int(pred)) + '\n')

if __name__ == "__main__":

    # train()
    predict()