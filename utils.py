from sklearn.preprocessing import LabelEncoder
import numpy as np

def label_encoding(dataframe, labels):
    """
    Encode the categorical variables into numerical variables

    :param dataframe: a dataframe with categorical variables
    :param labels: labels we want to encode
    :return:
    """

    le = LabelEncoder()
    for label in labels:
        le.fit(dataframe[label])
        dataframe[label] = le.transform(dataframe[label])

    return dataframe

def normalize_features(X_train):
    """
    Normalize the features by substracting the mean and
    dividing by the standard deviation

    :param X_train: the training features
    :return:
    """

    for features in X_train:
        feats = X_train[features].tolist()
        mean = np.mean(feats)
        std = np.std(feats)
        feats = (feats - mean)/std
        X_train[features] = feats

    return X_train