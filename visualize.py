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

    # train_csv = pd.read_csv(DATA_DIR + 'train.csv')
    #
    # train_csv['Cabin'] = train_csv['Cabin'].fillna('C0')
    # train_csv['Embarked'] = train_csv['Embarked'].fillna('0')
    # train_csv['Age'] = train_csv['Age'].fillna(train_csv['Age'].mean())
    # train_csv = label_encoding(train_csv, ['Sex', 'Ticket', 'Cabin', 'Embarked'])
    #
    # X_train = train_csv[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
    # Y_train = train_csv['Survived']
    #
    # X_train = normalize_features(X_train)

    X_train, Y_train = get_training_data()

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_train)
    X = X_reduced[:, 0]
    Y = X_reduced[:, 1]

    plt.figure()
    # plt.scatter(X_reduced[:, 0], X_train[:, 1], c=Y_train, s=40, cmap=plt.cm.Spectral);
    plt.scatter(X, Y, c=Y_train);
    # plt.scatter([3, 1, 4, 5], [-2, 7, 3, -1]);
    plt.show()

if __name__ == '__main__':

    visualize_data()



