from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm, neighbors
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class titanic_features:
    # =================================================================
    def __init__(self, path_csv: Path, path_test_csv: Path) -> None:
        self.df_train = pd.read_csv(path_csv)
        self.df_test = pd.read_csv(path_test_csv)
        self.df = pd.concat([self.df_train, self.df_test])
        self.list_numerical_features = list(self.df.corr().index)
        list_features = list(self.df.columns)
        for feature_nu in self.list_numerical_features:
            list_features.remove((feature_nu))
        self.list_category_features = list_features

        print(self.df_train.corr().to_string())
        print(self.df[self.list_numerical_features].isna().sum())

        print('nu',self.list_numerical_features,'cat',self.list_category_features)


    # ==========================================
    def plot_data(self, df: pd.DataFrame, feature: str) -> None:
        plt.figure()
        sns.barplot(x=feature, y='Survived', data=df)
        plt.show()

    # ===============================================================
    def impute_missing_values(self, df: pd.DataFrame) -> None:
        # Impute feature  Age
        replace_values = df.groupby('Pclass').median()['Age'].sort_index().values
        df.loc[(df['Age'].isnull() == 1) & (df['Pclass'] == 1), 'Age'] = replace_values[0]
        df.loc[(df['Age'].isnull() == 1) & (df['Pclass'] == 2), 'Age'] = replace_values[1]
        df.loc[(df['Age'].isnull() == 1) & (df['Pclass'] == 3), 'Age'] = replace_values[2]
        # Impute Feature Fare and Embarked
        df.fillna(value={'Embarked': df['Embarked'].mode()[0]},
                  inplace=True)

        # Impute feature  Age
        replace_values = df.groupby('Pclass').median()['Fare'].sort_index().values
        df.loc[(df['Fare'].isnull() == 1) & (df['Pclass'] == 1), 'Fare'] = replace_values[0]
        df.loc[(df['Fare'].isnull() == 1) & (df['Pclass'] == 2), 'Fare'] = replace_values[1]
        df.loc[(df['Fare'].isnull() == 1) & (df['Pclass'] == 3), 'Fare'] = replace_values[2]


    # ==========================================
    def drop_features(self, df: pd.DataFrame) -> None:
        df.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1, inplace=True)

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
        df['Title'] = df['Name'].apply(transform_name)
        df['Famly_number']=df['Parch']+df['SibSp']

        df.loc[(df.Title == 'Mr') & (df.Pclass == 1) & (df.Parch == 0) & (
                    (df.SibSp == 0) | (df.SibSp == 1)), 'MPPS'] = 1
        df.loc[(df.Title == 'Mr') & (df.Pclass != 1) & (df.Parch == 0) & (df.SibSp == 0), 'MPPS'] = 2
        df.loc[(df.Title == 'Miss') & (df.Pclass == 3) & (df.Parch == 0) & (df.SibSp == 0), 'MPPS'] = 3
        df['MPPS']=df.loc[df['MPPS'].isna(),'MPPS']=4
        df['MPPS'].astype(int)

    # ==========================================
    def get_features(self) -> pd.DataFrame:
        self.impute_missing_values(self.df)
        self.transform_features(self.df)
        self.drop_features(self.df)
        df = pd.get_dummies(self.df)

        return df

    # =========================================================================
    def get_features_train_test(self) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.features=self.get_features()

        features_train = self.features.iloc[0:self.df_train.shape[0], :],
        features_train=features_train[0].drop(columns=['Survived'])
        reference_train = np.reshape((self.df_train["Survived"].to_numpy()), (-1, 1))
        features_test = self.features.iloc[self.df_train.shape[0]:, :]
        features_test=features_test.drop(columns=['Survived'])
        return features_train, reference_train, features_test

#======================================================================================
if __name__ == '__main__':
    tc = titanic_features(Path(__file__).parent / 'train.csv', Path(__file__).parent / 'test.csv')
    ff=tc.get_features()
    print(ff.dtypes)




