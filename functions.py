import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

# ---------------- Fill missing values ---------------- #


def fill_age(df):
    sex = ['male', 'female']
    mean_ages = np.zeros((2, 3))
    for i, gender in enumerate(sex):
        for j in range(0, 3):
            mean_ages[i, j] = df[(df['Sex'] == gender) & (df['Pclass'] == j + 1)]['Age'].dropna().mean()

    for i, gender in enumerate(sex):
        for j in range(0, 3):
            df.loc[(df['Age'].isnull()) & (df['Sex'] == gender) & (df['Pclass'] == j + 1), 'Age'] = mean_ages[i, j]
    return df

# ---------------- Create New Features ---------------- #


def create_familysize(df):
    df['FamilySize'] = df["Parch"] + df["SibSp"] + 1
    return df


def create_isalone(df):
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    return df


def __find_titles__(words):
    for word in words.split():
        if word[0].isupper() and word.endswith('.'):
            return word


def create_title(df):
    df['Title'] = df.Name.apply(__find_titles__)
    df['Title'] = df.groupby('Title')['Title'].transform(
        lambda x: 'Other.' if x.count() < 9 else x)
    return df


def create_primacy(df):
    df['Primacy'] = (df['Fare'] + 1) / df['Pclass']
    return df


# ---------------- Mapping  ---------------- #


def mapping(df):
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    title_mapping = {"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Other.": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)

    df.loc[df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[df['Age'] > 64, 'Age'] = 4
    df['Age'] = df['Age'].astype(int)

    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)
    return df

# ---------------- Create Dummy Values ---------------- #


def dummy_embarked(df):
    dummies = pd.get_dummies(df['Embarked'])
    dummies.columns = ['C', 'Q', 'S']
    df = df.join(dummies)
    df = df.drop('Embarked', axis=1)
    return df

# ---------------- Drop unnecessary columns ---------------- #


def drop_columns(df):
    df = df.drop(['PassengerId', 'Pclass', 'Fare', 'Name', 'SibSp', 'Parch', 'Ticket', 'FamilySize', 'Cabin'], axis=1)
    return df

