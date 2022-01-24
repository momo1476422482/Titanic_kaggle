from typing import Dict

import numpy as np
import pandas as pd
from sklearn import svm, neighbors
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class titanic_model:
    def __init__(self,algo:str,param_grid:Dict) -> None:

        self.algo = algo
        if algo == 'ensemble':
            ensemble = [RandomForestClassifier(), svm.NuSVC(probability=True),
                        neighbors.KNeighborsClassifier()]
            classifiers_with_names = []
            _ = [classifiers_with_names.append((clf.__class__.__name__, clf)) for clf in ensemble]
            voting = VotingClassifier(classifiers_with_names, voting='hard')

            self.model = voting

        if algo == 'rf':
            # RandomForest
            rf = RandomForestClassifier(
                n_estimators=110, max_depth=8, max_features='auto',
                random_state=0, oob_score=False, min_samples_split=2,
                criterion='gini', min_samples_leaf=2, bootstrap=False

            )
            self.model = rf
        elif algo == 'grgb':
            grdb_clf = GradientBoostingClassifier(max_depth=4, max_features=10, n_estimators=101, random_state=0)
            self.model = grdb_clf
        elif algo =='svc':
            svc = make_pipeline(StandardScaler(), SVC(random_state=1))
            self.model=svc

        self.gridsearch = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy')

    # ==========================================
    def __call__(self,features_test:pd.DataFrame) -> np.ndarray:
        return self.gridsearch.predict(features_test)



