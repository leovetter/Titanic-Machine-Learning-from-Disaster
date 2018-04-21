from logisitic_regression import fit_logistic_regression
from neural_network import fit_ann
from features import get_original_training_data
import argparse

def train(model):
    """
    Train a model on the titanic training set

    :param model: the model to use for training on the titanic training set
    :return:
    """

    X_train, Y_train = get_original_training_data()

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
