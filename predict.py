from logisitic_regression import predict_logistic_regression
from neural_network import predict_ann
from sklearn.metrics import accuracy_score
from utils import label_encoding, normalize_features
import pandas as pd
import argparse

DATA_DIR = './data/'

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

def predict(model):
    """
    Predict the ouptut values from the test set and write
    the predictions in a csv files. Also compute the accuracy score

    :param model: the model to use for predicting on the titanic test set
    :return:
    """

    X_test, PassengerId = get_testing_data()

    if model == 'log':
        preds = predict_logistic_regression(X_test)
    elif model == 'ann':
        preds = predict_ann(X_test)

    gender_submission_csv = pd.read_csv(DATA_DIR + 'gender_submission.csv')

    print(accuracy_score(list(gender_submission_csv['Survived']), preds))
    with open('./predictions.csv', 'w') as csvfile:
        csvfile.write('PassengerId,Survived\n')
        for pred, id in zip(preds, PassengerId):
            csvfile.write(str(id) + ',' + str(pred) + '\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train machine learning and deep learning models on the titanic dataset')
    parser.add_argument('-m', dest='model', default='ann',
                        help='the model to use for training')
    args = parser.parse_args()

    predict(args.model)
