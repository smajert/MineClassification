import pickle
import re
from typing import Any

from interpret.glassbox import ExplainableBoostingClassifier
from lineartree import LinearTreeClassifier
from scipy.stats import uniform
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from mine_classification import params
from mine_classification.data_preparation import make_processing_pipeline, load_mine_data
from mine_classification.simulate_data import load_simulated_mine_data


def train_and_evaluate_model(
    model: ClassifierMixin, param_distribution: dict[str, Any], n_iter: int = 50
) -> None:

    df_train, df_test = load_mine_data(random_train_test_split=True)
    #df_train, df_test = load_simulated_mine_data(n_samples=338)
    X = df_train[["V", "H", "S_type", "S_wet"]]
    y = df_train["M"]
    X_test = df_test[["V", "H", "S_type", "S_wet"]]
    y_test = df_test["M"]

    pipeline = make_processing_pipeline(model)

    n_splits = 5
    k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    search = RandomizedSearchCV(pipeline, param_distribution, n_iter=n_iter, verbose=1, cv=k_fold, refit=True)
    search.fit(X, y)
    y_pred = search.best_estimator_.predict(X_test)

    results = {
        "best_parameters": search.best_params_,
        "pipeline": search.best_estimator_,
        "mu_cv": search.cv_results_['mean_test_score'][search.best_index_],
        "sigma_cv": search.cv_results_['std_test_score'][search.best_index_],
        "training_score": search.best_estimator_.score(X, y=y),
        "test_score": search.best_estimator_.score(X_test, y=y_test),
        "test_confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    model_name = re.sub(r'\W', '', str(type(model)).split('.')[-1])
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
        "classify__ccp_alpha": uniform(0, 0.4),
        "classify__max_depth": [2, 3, 4],
        "classify__criterion": ["gini", "entropy", "log_loss"]
    }
    train_and_evaluate_model(DecisionTreeClassifier(), param_distribution)

    print("------- ExplainableBoostingClassifier -------")
    param_distribution = {
        "classify__estimator__early_stopping_rounds": [50, 100, 200],
        "classify__estimator__greediness": [0, 0.05, 0.1],
        "classify__estimator__interactions": [0, 1, 2, 3],
        "classify__estimator__learning_rate": uniform(0.001, 0.2),
        "classify__estimator__smoothing_rounds": [0, 1, 2, 5],
    }
    train_and_evaluate_model(OneVsRestClassifier(ExplainableBoostingClassifier()), param_distribution, n_iter=10)

    print("------- LinearTree -------")
    param_distribution = {
        "classify__base_estimator": [RidgeClassifier(), LogisticRegression()],
        "classify__max_depth": [2, 3, 4, 5, 6]
    }
    train_and_evaluate_model(LinearTreeClassifier(base_estimator=None), param_distribution)

    print("------- RandomForestClassifier -------")
    param_distribution = {

    }
    train_and_evaluate_model(RandomForestClassifier(), param_distribution)


    print("\n------- MLP -------")
    param_distribution = {
         "classify__max_iter": [10000],
         # "classify__learning_rate": ["constant"],
         # "classify__tol": [1e-4],
         "classify__alpha": [0.0001, 0.001, 0.0002, 0.0003, 0.0],
         # "classify__n_iter_no_change": [10],
         "classify__hidden_layer_sizes": [(100, 50, 20, 10)]  #[(100, ), (200, ), (100, 50, 20, 10), (300, 200, 100)]
    }
    train_and_evaluate_model(MLPClassifier(), param_distribution)






