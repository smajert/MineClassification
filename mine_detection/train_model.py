from dataclasses import dataclass
import pickle
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from mine_detection import params
from mine_detection.data_preparation import get_preprocessing_pipeline, load_mine_data

@dataclass()
class ModelScores:
    accuracy: float


def train_and_evaluate_model(
    model_type: ClassifierMixin, param_distribution: dict[str, Any], n_iter: int = 50
) -> None:

    df_train, df_test = load_mine_data(random_train_test_split=True)
    X = df_train[["V", "H", "S_type", "S_wet"]]
    y = df_train["M"]
    X_test = df_test[["V", "H", "S_type", "S_wet"]]
    y_test = df_test["M"]

    pipeline = Pipeline(
        [
            ("preprocessing", get_preprocessing_pipeline()),
            ("classify", model_type)
        ]
    )

    n_splits = 5
    k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    search = RandomizedSearchCV(pipeline, param_distribution, n_iter=n_iter, verbose=1, cv=k_fold, refit=True)
    search.fit(X, y)
    y_pred = search.predict(X_test)
    results = {
        "best_parameters": search.best_params_,
        "model": search.best_estimator_,
        "mu_cv": search.cv_results_['mean_test_score'][search.best_index_],
        "sigma_cv": search.cv_results_['std_test_score'][search.best_index_],
        "training_score": search.best_estimator_.score(X, y=y),
        "test_score": search.best_estimator_.score(X_test, y=y_test),
        "test_confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    model_name = re.sub(r'\W', '', str(type(model_type)).split('.')[-1])
    print(params.REPO_ROOT / f"outs/{model_name}.pkl")
    with open(params.REPO_ROOT / f"outs/{model_name}.pkl", "wb") as file:
        pickle.dump(results, file)

    print(results)


if __name__ == "__main__":

    param_distribution = {
        "classify__max_iter": [1000],
        "classify__learning_rate": ["constant"],
        "classify__tol": [1e-6],
        "classify__alpha": [0],
        "classify__verbose": [True],
        "classify__n_iter_no_change": [100],
        "classify__hidden_layer_sizes": [(1000, 900, 800, 700, 600, 500)]
    }
    print("------- Decision Tree -------")
    param_distribution = {
        "classify__ccp_alpha": [0, 0.0001, 0.001, 0.01, 0.1],
        "classify__max_depth": [None, 5, 3, 10, 20, 40],
        "classify__criterion": ["gini", "entropy", "log_loss"]
    }
    train_and_evaluate_model(DecisionTreeClassifier(), param_distribution)

    print("\n------- MLP -------")
    param_distribution = {
         "classify__max_iter": [10000],
         # "classify__learning_rate": ["constant"],
         # "classify__tol": [1e-4],
         "classify__alpha": [0.0001, 0.001, 0.0002, 0.0003, 0.0],
         # "classify__n_iter_no_change": [10],
         "classify__hidden_layer_sizes": [(100, ), (200, ), (100, 50, 20, 10), (300, 200, 100)]
    }
    train_and_evaluate_model(MLPClassifier(), param_distribution)






