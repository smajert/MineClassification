from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


from mine_detection.data_preparation import get_preprocessing_pipeline, load_mine_data

@dataclass()
class ModelScores:
    accuracy: float


def train_and_evaluate_model(
    model_type: ClassifierMixin, param_distribution: dict[str, Any]
) -> tuple[ClassifierMixin, ModelScores]:

    df_train, df_test = load_mine_data()
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

    # k_fold = StratifiedKFold(n_splits=2, shuffle=True)
    # search = RandomizedSearchCV(pipeline, param_distribution, n_iter=500, verbose=2, scoring="accuracy", cv=k_fold)
    # search.fit(X, y)
    # print(search.best_params_)
    # print(search.best_score_)

    #pipeline.set_params(**search.best_params_)
    pipeline.set_params(**{key: value[0] for key, value in param_distribution.items()})
    pipeline.fit(X, y=y)
    print(pipeline.score(X, y=y))
    print(pipeline.score(X_test, y=y_test))


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
    train_and_evaluate_model(MLPClassifier(), param_distribution)





    # scores = cross_validate(pipeline, X, y, scoring="accuracy", cv=5, verbose=1)
    # print(scores)


    # model.fit(df_train[["V", "H", "S"]], df_train["M"])
    # print(model.predict(np.array([[1, 1, 5], [0, 0, 2]])))
    # print(scores)





