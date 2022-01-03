from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm, neighbors
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class titanic:
    def __init__(self, path_csv: Path, path_test_csv: Path) -> None:
        self.model = None
        self.df_train = pd.read_csv(path_csv)
        self.df_test = pd.read_csv(path_test_csv)
        self.df = pd.concat([self.df_train, self.df_test])

    # ==========================================
    def plot_data(self, df: pd.DataFrame, feature: str) -> None:
        plt.figure()
        sns.barplot(x=feature, y='Survived', data=df)
        plt.show()

    # ==========================================

    def impute_missing_values(self, df: pd.DataFrame) -> None:
        # Impute feature  Age
        replace_values = df.groupby('Pclass').median()['Age'].sort_index().values
        df.loc[(df['Age'].isnull() == 1) & (df['Pclass'] == 1), 'Age'] = replace_values[0]
        df.loc[(df['Age'].isnull() == 1) & (df['Pclass'] == 2), 'Age'] = replace_values[1]
        df.loc[(df['Age'].isnull() == 1) & (df['Pclass'] == 3), 'Age'] = replace_values[2]
        # Impute Feature Fare and Embarked
        df.fillna(value={'Fare': df['Fare'].median(), 'Embarked': df['Embarked'].mode()[0]},
                  inplace=True)

    # ==========================================
    def drop_features(self, df: pd.DataFrame) -> None:
        df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # ==========================================
    def transform_features(self, df: pd.DataFrame) -> None:
        # Scale the Features Age and Fare
        std_scaler = StandardScaler()
        df.loc[:, ['Age', 'Fare']] = std_scaler.fit_transform(df[['Age', 'Fare']])

        # Create New Features of cluster of Age and Name
        def transform_name(input_str: str) -> str:

            if input_str == ' Capt' or input_str == ' Col' or input_str == ' Major' or input_str == ' Dr' or input_str == ' Rev':
                return 'Officer'
            elif input_str == ' Don' or input_str == ' Sir' or input_str == ' the Countess' or input_str == ' Dona' or input_str == ' Lady' or input_str == ' Jonkheer':
                return 'Royalty'
            elif input_str in [' Mme', ' Ms', ' Mrs', ' Miss', ' Mlle']:
                return 'Miss'
            elif input_str == ' Mr':
                return 'Mr'
            elif input_str == ' Master':
                return 'Master'

        # df['Name'] = df['Name'].str.split(',').str[1].str.split('.').str[0].astype(str)
        #df['AgeGroup'] = pd.qcut(df['Age'], 4, labels=["child", "adolescent", "adult", "senior"])
        #df['Title'] = df['Name'].str.split(',').str[1].str.split('.').str[0].astype(str)
        #df['Title'] = df['Title'].apply(transform_name)
        # Encoder Feature into Numerical Labels

    # ==========================================
    def get_features(self) -> None:
        self.impute_missing_values(self.df)
        self.transform_features(self.df)
        self.drop_features(self.df)
        df = pd.get_dummies(self.df)
        print(df.info)
        self.features = df.to_numpy()

    # ==========================================
    def train_model(self, algo: str, features: np.ndarray, reference: np.ndarray) -> None:
        if algo == 'ensemble':
            ensemble = [RandomForestClassifier(), svm.NuSVC(probability=True),
                        neighbors.KNeighborsClassifier()]

            classifiers_with_names = []
            _ = [classifiers_with_names.append((clf.__class__.__name__, clf)) for clf in ensemble]
            voting = VotingClassifier(classifiers_with_names, voting='hard')
            voting.fit(features, reference)
            self.model = voting

        if algo == 'rf':
            # RandomForest
            rf = RandomForestClassifier(
                n_estimators=110, max_depth=8, max_features='auto',
                random_state=0, oob_score=False, min_samples_split=2,
                criterion='gini', min_samples_leaf=2, bootstrap=False

            )
            rf.fit(features, reference)
            self.model = rf
        elif algo == 'grgb':
            grdb_clf = GradientBoostingClassifier(max_depth=4, max_features=10, n_estimators=101, random_state=0)
            grdb_clf.fit(features, reference)
            self.model = grdb_clf

    # ==========================================
    def validate_model(self, algo: str) -> None:
        features_train = self.features[0:self.df_train.shape[0], :]
        X_train = features_train[:, 2:]
        y_train = self.df_train["Survived"]

        self.train_model(algo, X_train, y_train)
        print(self.model.score(X_train, y_train))

    # ==========================================
    def predict_survive(self) -> None:
        features_test = self.features[self.df_train.shape[0]:, :]
        self.df_test['Survived'] = self.model.predict(features_test[:, 2:])

    # ===============================================
    def save_result(self) -> None:
        self.df_result = self.df_test[['PassengerId', 'Survived']]
        self.df_result.to_csv('result.csv', index=False)

    # ===============================================
    def __call__(self):

        self.get_features()
        print(self.features)
        self.validate_model('rf')
        self.predict_survive()
        self.save_result()


if __name__ == '__main__':
    tc = titanic(Path(__file__).parent / 'train.csv', Path(__file__).parent / 'test.csv')
    tc()
