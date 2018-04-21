from logisitic_regression import predict_logistic_regression
from neural_network import predict_ann
from sklearn.metrics import accuracy_score
from utils import label_encoding, normalize_features
import argparse


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
