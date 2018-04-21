from utils import label_encoding, normalize_features
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import re as re

DATA_DIR = './data/'

# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def get_original_training_data():
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

def get_original_testing_data():
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

    X_test = normalize_features(X_test)

    return X_test.as_matrix(), test_csv['PassengerId']

def get_engineered_training_data():

    train_df = pd.read_csv(DATA_DIR + 'train.csv')

    # Fill the missing values with median or mode
    train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
    train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

    # Build the Family features
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

    # Build the Title features
    train_df['Title'] = train_df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    # Convert categorical variables Sex, Embarked and Title into dummy variables
    label = LabelEncoder()
    train_df['Sex'] = label.fit_transform(train_df['Sex'])
    train_df['Embarked'] = label.fit_transform(train_df['Embarked'])
    train_df['Title'] = label.fit_transform(train_df['Title'])

    # Convert continuous variables Age and Fare to categorical variables
    train_df['Age'] = pd.cut(train_df['Age'], 10)
    train_df['Fare'] = pd.cut(train_df['Fare'], 10)
    train_df['Age'] = label.fit_transform(train_df['Age'])
    train_df['Fare'] = label.fit_transform(train_df['Fare'])

    # Drop non relevant features
    train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)

    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]

    X_train = normalize_features(X_train)

    return X_train.as_matrix(), Y_train.as_matrix()


def get_engineered_testing_data():

    test_df = pd.read_csv(DATA_DIR + 'test.csv')

    # Fill the missing values with median or mode
    test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
    test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

    # Build the Family features
    test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

    # Build the Title features
    test_df['Title'] = test_df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    # Convert categorical variables Sex, Embarked and Title into dummy variables
    label = LabelEncoder()
    test_df['Sex'] = label.fit_transform(test_df['Sex'])
    test_df['Embarked'] = label.fit_transform(test_df['Embarked'])
    test_df['Title'] = label.fit_transform(test_df['Title'])

    # Convert continuous variables Age and Fare to categorical variables
    test_df['Age'] = pd.cut(test_df['Age'], 10)
    test_df['Fare'] = pd.cut(test_df['Fare'], 10)
    test_df['Age'] = label.fit_transform(test_df['Age'])
    test_df['Fare'] = label.fit_transform(test_df['Fare'])

    # Drop non relevant features
    test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)

    test_df = normalize_features(test_df)

    X_test = test_df.as_matrix()

    return X_test, pd.read_csv(DATA_DIR + 'test.csv')['PassengerId']

def get_eng_training_data():

    train_df = pd.read_csv(DATA_DIR + 'train.csv')

    # Create new feature FamilySize as a combination of SibSp and Parch
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

    # Create new feature IsAlone from FamilySize
    train_df['IsAlone'] = 0
    train_df.loc[train_df['FamilySize'] == 1, 'IsAlone'] = 1

    # Remove all NULLS in the Embarked column
    train_df['Embarked'] = train_df['Embarked'].fillna('S')

    # Remove all NULLS in the Fare column and create a new feature CategoricalFare
    train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
    train_df['CategoricalFare'] = pd.qcut(train_df['Fare'], 4)

    # Create a New feature CategoricalAge
    age_avg = train_df['Age'].mean()
    age_std = train_df['Age'].std()
    age_null_count = train_df['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    train_df['Age'][np.isnan(train_df['Age'])] = age_null_random_list
    train_df['Age'] = train_df['Age'].astype(int)
    train_df['CategoricalAge'] = pd.cut(train_df['Age'], 5)

    # Create a new feature Title, containing the titles of passenger names
    train_df['Title'] = train_df['Name'].apply(get_title)

    # Group all non-common titles into one single grouping "Rare"
    train_df['Title'] = train_df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
    train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
    train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')

    # Encoding our features
    # Mapping Sex
    train_df['Sex'] = train_df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    train_df['Title'] = train_df['Title'].map(title_mapping)
    train_df['Title'] = train_df['Title'].fillna(0)

    # Mapping Embarked
    train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Mapping Fare
    train_df.loc[train_df['Fare'] <= 7.91, 'Fare'] = 0
    train_df.loc[(train_df['Fare'] > 7.91) & (train_df['Fare'] <= 14.454), 'Fare'] = 1
    train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare'] <= 31), 'Fare'] = 2
    train_df.loc[train_df['Fare'] > 31, 'Fare'] = 3
    train_df['Fare'] = train_df['Fare'].astype(int)

    # Mapping Age
    train_df.loc[train_df['Age'] <= 16, 'Age'] = 0
    train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 32), 'Age'] = 1
    train_df.loc[(train_df['Age'] > 32) & (train_df['Age'] <= 48), 'Age'] = 2
    train_df.loc[(train_df['Age'] > 48) & (train_df['Age'] <= 64), 'Age'] = 3
    train_df.loc[train_df['Age'] > 64, 'Age'] = 4

    # Feature Selection
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    # drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch', 'FamilySize']
    train_df = train_df.drop(drop_elements, axis=1)
    train_df = train_df.drop(['CategoricalAge', 'CategoricalFare'], axis=1)

    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]

    # X_train = normalize_features(X_train)

    return X_train.as_matrix(), Y_train.as_matrix()

def get_eng_testing_data():

    test_df = pd.read_csv("./data/test.csv")

    # Create new feature FamilySize as a combination of SibSp and Parch
    test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

    # Create new feature IsAlone from FamilySize
    test_df['IsAlone'] = 0
    test_df.loc[test_df['FamilySize'] == 1, 'IsAlone'] = 1

    # Remove all NULLS in the Embarked column
    test_df['Embarked'] = test_df['Embarked'].fillna('S')

    # Remove all NULLS in the Fare column and create a new feature CategoricalFare
    test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

    # Create a New feature CategoricalAge
    age_avg = test_df['Age'].mean()
    age_std = test_df['Age'].std()
    age_null_count = test_df['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    test_df['Age'][np.isnan(test_df['Age'])] = age_null_random_list
    test_df['Age'] = test_df['Age'].astype(int)

    # Create a new feature Title, containing the titles of passenger names
    test_df['Title'] = test_df['Name'].apply(get_title)

    # Group all non-common titles into one single grouping "Rare"
    test_df['Title'] = test_df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
    test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
    test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')

    # Encoding our features
    # Mapping Sex
    test_df['Sex'] = test_df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    test_df['Title'] = test_df['Title'].map(title_mapping)
    test_df['Title'] = test_df['Title'].fillna(0)

    # Mapping Embarked
    test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Mapping Fare
    test_df.loc[test_df['Fare'] <= 7.91, 'Fare'] = 0
    test_df.loc[(test_df['Fare'] > 7.91) & (test_df['Fare'] <= 14.454), 'Fare'] = 1
    test_df.loc[(test_df['Fare'] > 14.454) & (test_df['Fare'] <= 31), 'Fare'] = 2
    test_df.loc[test_df['Fare'] > 31, 'Fare'] = 3
    test_df['Fare'] = test_df['Fare'].astype(int)

    # Mapping Age
    test_df.loc[test_df['Age'] <= 16, 'Age'] = 0
    test_df.loc[(test_df['Age'] > 16) & (test_df['Age'] <= 32), 'Age'] = 1
    test_df.loc[(test_df['Age'] > 32) & (test_df['Age'] <= 48), 'Age'] = 2
    test_df.loc[(test_df['Age'] > 48) & (test_df['Age'] <= 64), 'Age'] = 3
    test_df.loc[test_df['Age'] > 64, 'Age'] = 4

    # Feature Selection
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    # drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch', 'FamilySize']
    test_df = test_df.drop(drop_elements, axis=1)

    # test_df = normalize_features(test_df)

    X_test = test_df.as_matrix()

    return X_test, pd.read_csv(DATA_DIR + 'test.csv')['PassengerId']



