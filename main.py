from dataset import titanic_features
from model import titanic_model
from typing import Callable
import numpy as np
import pandas as pd
from pathlib import Path


class titanic:
    # =====================================================================
    def __init__(self, model: Callable) -> None:
        self.df_result = pd.DataFrame()
        self.model = model

    # =====================================================================
    def train_model(self, features: pd.DataFrame, reference: pd.DataFrame) -> None:
        self.model.gridsearch.fit(features, reference)
        print('best parameter', self.model.gridsearch.best_params_, 'best score',
              self.model.gridsearch.best_score_)

    # =====================================================
    def get_features_importance(self):
        features_importance = pd.DataFrame()
        features_importance['name'] = self.features.columns
        features_importance['importance'] = self.gridsearch.best_estimator_.coef_.reshape(-1, 1)
        self.features_importance = features_importance.sort_values('importance')

    # =====================================================
    def save_result(self, path_test: Path, features_test: pd.DataFrame) -> None:
        df_result = pd.read_csv(path_test)
        print(features_test.isna().sum())
        df_result['Survived'] = self.model(features_test)
        df_result = df_result[['PassengerId', 'Survived']]
        df_result.to_csv(Path(__file__).parent / 'result.csv', index=False)

    # ===============================================
    def __call__(self, features_train: np.ndarray, reference_train: np.ndarray,
                 features_test: pd.DataFrame):
        self.train_model(features_train, reference_train)
        self.save_result(Path(__file__).parent / 'test.csv', features_test)


# =====================================================================================================
if __name__ == '__main__':
    features_train, reference_train, features_test = titanic_features(Path(__file__).parent / 'train.csv',
                                                                      Path(
                                                                          __file__).parent / 'test.csv').get_features_train_test()

    tm = titanic_model('rf', param_grid={'n_estimators': [280,300,320], 'max_depth': [8,10,12]})


    hp = titanic(model=tm)
    hp(features_train=features_train, features_test=features_test, reference_train=reference_train)
