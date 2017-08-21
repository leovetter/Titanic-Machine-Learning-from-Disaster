from logisitic_regression import fit_logistic_regression
from neural_network import fit_ann
from utils import label_encoding, normalize_features
import pandas as pd
import argparse

DATA_DIR = './data/'

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

    X_train = normalize_features(X_train)

    return X_train.as_matrix().reshape(891, 9), Y_train.as_matrix().reshape(891, 1)

def train(model):
    """
    Train a model on the titanic training set

    :param model: the model to use for training on the titanic training set
    :return:
    """

    X_train, Y_train = get_training_data()

    if model == 'log':
        fit_logistic_regression(X_train, Y_train)
    elif model == 'ann':
        fit_ann(X_train, Y_train)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train machine learning and deep learning models on the titanic dataset')
    parser.add_argument('-m', dest='model', default='ann',
                        help='the model to use for training')
    args = parser.parse_args()

    train(args.model)
