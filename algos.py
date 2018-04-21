from sklearn.ensemble import RandomForestClassifier
from features import get_original_training_data, get_original_testing_data, \
                     get_engineered_training_data, get_engineered_testing_data


def random_forest():

    # X_train, Y_train = get_original_training_data()
    # X_test, PassengerId = get_original_testing_data()
    X_train, Y_train = get_engineered_training_data()
    X_test, PassengerId = get_engineered_testing_data()

    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    with open('./predictions.csv', 'w') as csvfile:
        csvfile.write('PassengerId,Survived\n')
        for pred, id in zip(Y_pred, PassengerId):
            csvfile.write(str(id) + ',' + str(pred) + '\n')



if __name__ == '__main__':

    random_forest()


