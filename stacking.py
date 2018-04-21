from features import get_engineered_training_data, get_engineered_testing_data, get_eng_training_data, get_eng_testing_data
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
import xgboost as xgb
import numpy as np
import pandas as pd
import sys

# Put in our parameters for said classifiers
# Random Forest parameters
# rf_gini_params = {
#     'n_jobs': -1,
#     'n_estimators': 575,
#      'warm_start': True,
#      #'max_features': 0.2,
#     'max_depth': 5,
#     'min_samples_leaf': 2,
#     'max_features': 'sqrt',
#     'verbose': 3,
#     'criterion': 'gini',
# }

rf_params = {
    'n_jobs': [-1],
    'n_estimators': [10, 50, 100, 300, 700, 1000],
    'warm_start': [True, False],
    'max_features': [1, 3, 5, 7, 9, 'auto', 'sqrt', 'log2', None],
    'max_depth': [5, 10, 20, 30, None],
    'min_samples_leaf': [1, 3, 7, 10, 15],
    'criterion': ['gini', 'entropy'],
}

# rf_entropy_params = {
#     'n_jobs': -1,
#     'n_estimators': 575,
#      'warm_start': True,
#      #'max_features': 0.2,
#     'max_depth': 6,
#     'min_samples_leaf': 2,
#     'max_features': 'sqrt',
#     'verbose': 0,
#     'criterion': 'entropy',
# }

# # Extra Trees Parameters
# et_gini_params = {
#     'n_jobs': -1,
#     'n_estimators':575,
#     #'max_features': 0.5,
#     'max_depth': 5,
#     'min_samples_leaf': 3,
#     'verbose': 3,
#     'criterion': 'gini',
# }

et_params = {
    'n_jobs': [-1],
    'n_estimators': [10, 50, 100, 300, 700, 1000],
    'max_features': [1, 3, 5, 7, 9, 'auto', 'sqrt', 'log2', None],
    'max_depth': [5, 10, 20, 30, None],
    'min_samples_leaf': [1, 3, 7, 10, 15],
    'criterion': ['gini', 'entropy'],
}

# Extra Trees Parameters
# et_entropy_params = {
#     'n_jobs': -1,
#     'n_estimators':500,
#     #'max_features': 0.5,
#     'max_depth': 8,
#     'min_samples_leaf': 2,
#     'verbose': 0,
#     'criterion': 'entropy',
# }

# # AdaBoost parameters
# ada_params = {
#     'n_estimators': 575,
#     'learning_rate' : 0.95
# }

# AdaBoost parameters
ada_params = {
    'n_estimators': [10, 50, 100, 300, 700, 1000],
    'learning_rate' : [0.25, 0.5, 0.75, 1],
    'algorithm': ['SAMME', 'SAMME.R']
}

# # Gradient Boosting parameters
# gb_params = {
#     'n_estimators': 575,
#      #'max_features': 0.2,
#     'max_depth': 5,
#     'min_samples_leaf': 3,
#     'verbose': 3
# }

# Gradient Boosting parameters
gb_params = {
    'n_estimators': [10, 50, 100, 300, 700, 1000],
    'max_features': [1, 3, 5, 7, 9, 'auto', 'sqrt', 'log2', None],
    'max_depth': [5, 10, 20, 30, None],
    'min_samples_leaf': [1, 3, 7, 10, 15],
    'criterion': ['friedman_mse', 'mse', 'mae'],
}

# # Support Vector Classifier parameters
# svc_params = {
#     'kernel' : 'linear',
#     'C' : 0.025
# }

# Support Vector Classifier parameters
svc_params = {
    'C': [0.3, 0.7, 1.0, 1.3, 1.7, 2],
    'kernel': ['rbf', 'linear', 'sigmoid', 'poly'],
    'degree': [1, 2, 3, 4, 5],
}

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self ,x ,y):
        return self.clf.fit(x ,y)

    def feature_importances(self ,x ,y):
        return self.clf.fit(x ,y).feature_importances_

# oof = Out-of-Folds
def get_oof(clf, x_train, y_train, x_test, ntrain, ntest, NFOLDS, kf):

    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):

        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

def stacking():

    X_train, Y_train = get_eng_training_data()
    X_test, PassengerId = get_eng_testing_data()

    # Some useful parameters which will come in handy later on
    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]
    SEED = 0  # for reproducibility
    NFOLDS = 5  # set folds for out-of-fold prediction
    kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)

    acc_scorer = make_scorer(accuracy_score)

    # Run the grid search on random forest
    random_forest = RandomForestClassifier()
    print('Random Forest grid search...')
    grid_obj = GridSearchCV(random_forest, rf_params, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, Y_train)
    clf_random_forest = grid_obj.best_estimator_
    clf_random_forest.fit(X_train, Y_train)

    # Run the grid search on extra trees
    extra_trees = ExtraTreesClassifier()
    print('Extra Trees grid search...')
    grid_obj = GridSearchCV(extra_trees, et_params, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, Y_train)
    clf_extra_trees = grid_obj.best_estimator_
    clf_extra_trees.fit(X_train, Y_train)

    # Run the grid search on adaboost
    adaboost = AdaBoostClassifier()
    print('Adaboost grid search...')
    grid_obj = GridSearchCV(adaboost, ada_params, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, Y_train)
    clf_adaboost = grid_obj.best_estimator_
    clf_adaboost.fit(X_train, Y_train)

    # Run the grid search on gradient_boosting
    gradient_boosting = GradientBoostingClassifier()
    print('Gradient Boosting grid search...')
    grid_obj = GridSearchCV(gradient_boosting, gb_params, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, Y_train)
    clf_gradient_boosting = grid_obj.best_estimator_
    clf_gradient_boosting.fit(X_train, Y_train)

    # Run the grid search on support vector
    support_vector = SVC()
    print('Support Vector grid search...')
    grid_obj = GridSearchCV(support_vector, svc_params, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, Y_train)
    clf_support_vector = grid_obj.best_estimator_
    clf_support_vector.fit(X_train, Y_train)


    # rf_gini = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_gini_params)
    # # rf_entropy = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_entropy_params)
    # et_gini = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_gini_params)
    # # et_entropy = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_entropy_params)
    # ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
    # gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
    # svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
    #

    print('Running KFOLD')
    rf_oof_train, rf_oof_test = get_oof(clf_random_forest, X_train, Y_train, X_test, ntrain, ntest, NFOLDS, kf)  # Random Forest gini
    # rf_entropy_oof_train, rf_entropy_oof_test = get_oof(rf_entropy, X_train, Y_train, X_test, ntrain, ntest, NFOLDS, kf)  # Random Forest entropy
    et_oof_train, et_oof_test = get_oof(clf_extra_trees, X_train, Y_train, X_test, ntrain, ntest, NFOLDS, kf)  # Extra Trees gini
    # et_entropy_oof_train, et_entropy_oof_test = get_oof(et_entropy, X_train, Y_train, X_test, ntrain, ntest, NFOLDS, kf)  # Extra Trees entropy
    ada_oof_train, ada_oof_test = get_oof(clf_adaboost, X_train, Y_train, X_test, ntrain, ntest, NFOLDS, kf)  # AdaBoost
    gb_oof_train, gb_oof_test = get_oof(clf_gradient_boosting, X_train, Y_train, X_test, ntrain, ntest, NFOLDS, kf)  # Gradient Boost
    svc_oof_train, svc_oof_test = get_oof(clf_support_vector, X_train, Y_train, X_test, ntrain, ntest, NFOLDS, kf)  # Support Vector Classifier

    # rf_feature = rf.feature_importances(X_train, Y_train)
    # et_feature = et.feature_importances(X_train, Y_train)
    # ada_feature = ada.feature_importances(X_train, Y_train)
    # gb_feature = gb.feature_importances(X_train, Y_train)
    #
    # cols = pd.read_csv('./data/train.csv').columns.values
    # # Create a dataframe with features
    # # feature_dataframe = pd.DataFrame({'features': cols,
    # #                                   'Random Forest feature importances': rf_feature,
    # #                                   'Extra Trees  feature importances': et_feature,
    # #                                   'AdaBoost feature importances': ada_feature,
    # #                                   'Gradient Boost feature importances': gb_feature
    # #                                   })
    #
    # # feature_dataframe['mean'] = feature_dataframe.mean(axis=1)  # axis = 1 computes the mean row-wise
    #
    # base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
    #                                        'ExtraTrees': et_oof_train.ravel(),
    #                                        'AdaBoost': ada_oof_train.ravel(),
    #                                        'GradientBoost': gb_oof_train.ravel()
    #                                        })

    # X_train = np.concatenate((et_gini_oof_train, et_entropy_oof_train,  rf_gini_oof_train, rf_entropy_oof_train, ada_oof_train,
    #                           gb_oof_train, svc_oof_train), axis=1)
    X_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
    # X_test = np.concatenate((et_gini_oof_test, et_entropy_oof_test,  rf_gini_oof_test, rf_entropy_oof_test, ada_oof_test,
    #                          gb_oof_test, svc_oof_test), axis=1)
    X_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

    gbm = xgb.XGBClassifier(
        # learning_rate = 0.02,
        n_estimators=5000,
        max_depth=4,
        min_child_weight=2,
        # gamma=1,
        gamma=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=-1,
        scale_pos_weight=1).fit(X_train, Y_train)
    predictions = gbm.predict(X_test)

    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId,
                                       'Survived': predictions})
    StackingSubmission.to_csv("StackingSubmission.csv", index=False)

if __name__ == '__main__':

    stacking()