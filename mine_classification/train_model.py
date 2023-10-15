import pickle
import re
from typing import Any

from interpret.glassbox import ExplainableBoostingClassifier
from prettytable import PrettyTable
from scipy.stats import randint, uniform
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from mine_classification import params
from mine_classification.data_preparation import make_processing_pipeline, load_mine_data
from mine_classification.simulate_data import load_simulated_mine_data


def train_and_evaluate_model(
    model: ClassifierMixin,
    hyper_param_distribution: dict[str, Any],
    n_iter: int = 50,
    preprocess_info: params.Preprocessing = params.Preprocessing,
) -> dict[str, Any]:
    if preprocess_info.simulated_data:
        df_train, df_test = load_simulated_mine_data(n_samples=338)
    else:
        df_train, df_test = load_mine_data(preprocess_info.random_train_test_split, preprocess_info.soil_treatment)

    X = df_train[["V", "H", "S_type", "S_wet"]]
    y = df_train["M"]
    X_test = df_test[["V", "H", "S_type", "S_wet"]]
    y_test = df_test["M"]

    pipeline = make_processing_pipeline(preprocess_info.soil_treatment, classifier=model)

    n_splits = 5
    k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    search = RandomizedSearchCV(pipeline, hyper_param_distribution, n_iter=n_iter, verbose=2, cv=k_fold, refit=True)
    search.fit(X, y)
    y_pred = search.best_estimator_.predict(X_test)

    results = {
        "best_hyperparameters": search.best_params_,
        "pipeline": search.best_estimator_,
        "mean_cv": search.cv_results_['mean_test_score'][search.best_index_],
        "stdv_cv": search.cv_results_['std_test_score'][search.best_index_],
        "training_score": search.best_estimator_.score(X, y=y),
        "test_score": search.best_estimator_.score(X_test, y=y_test),
        "test_confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    if isinstance(model, OneVsRestClassifier):
        model_name = type(model.estimator).__name__
    else:
        model_name = type(model).__name__

    model_name = re.sub(r'\W', '', model_name.split('.')[-1])

    with open(params.REPO_ROOT / f"outs/{model_name}.pkl", "wb") as file:
        pickle.dump(results, file)

    return results


if __name__ == "__main__":
    results = []

    print("------- Decision Tree -------")
    hyper_param_distribution = {
        "classify__ccp_alpha": uniform(6e-3, 8e-3),
        "classify__max_depth": [3, 5],
        "classify__min_samples_split": randint(10, 20),
        "classify__min_samples_leaf": randint(5, 15),
        "classify__min_impurity_decrease": uniform(6e-3, 10e-3),
        "classify__criterion": ["gini", "entropy"],
        "classify__max_features": ["sqrt", "log2", None],
        "classify__splitter": ["best"]
    }
    results.append(train_and_evaluate_model(DecisionTreeClassifier(), hyper_param_distribution, n_iter=50))

    print("------- ExplainableBoostingClassifier -------")
    hyper_param_distribution = {
        "classify__estimator__early_stopping_rounds": [190, 200, 210],
        "classify__estimator__greediness": [0],
        "classify__estimator__interactions": [3],
        "classify__estimator__learning_rate": uniform(1e-3, 4e-3),
        "classify__estimator__max_bins": [16],
        "classify__estimator__max_interaction_bins": [128],
        "classify__estimator__max_rounds": [8000],
        "classify__estimator__smoothing_rounds": [0, 1, 2],
    }
    results.append(
        train_and_evaluate_model(
            OneVsRestClassifier(ExplainableBoostingClassifier()), hyper_param_distribution, n_iter=5
        )
    )

    print("------- RandomForestClassifier -------")
    hyper_param_distribution = {
        "classify__n_estimators": [250, 300, 350],
        "classify__criterion": ["gini"],
        "classify__max_features": [ None],
        "classify__ccp_alpha": uniform(3e-3, 8e-3),

    }
    results.append(train_and_evaluate_model(RandomForestClassifier(), hyper_param_distribution, n_iter=50))

    print("\n------- MLP -------")
    hyper_param_distribution = {
         "classify__max_iter": [10000],
         "classify__early_stopping": [False],
         "classify__learning_rate": ["constant"],
         "classify__tol": uniform(3e-4, 7e-4),
         "classify__alpha": uniform(0.0002, 0.0004),
         "classify__n_iter_no_change": [33, 34, 35, 36, 37],
         "classify__hidden_layer_sizes": [(200, 100, 50, 20, 10)]
    }
    results.append(train_and_evaluate_model(MLPClassifier(), hyper_param_distribution, n_iter=50))

    results_table = PrettyTable(field_names=["Classifier", "acc_CV", "acc_stdv_CV", "acc_test"])
    for result in results:
        classifier = result["pipeline"]["classify"]
        if isinstance(classifier, OneVsRestClassifier):
            classifier_name = type(classifier.estimator).__name__
        else:
            classifier_name = type(classifier).__name__

        results_table.add_row([
            classifier_name,
            f"{result['mean_cv']:.3f}",
            f"{result['stdv_cv']:.3f}",
            f"{result['test_score']:.3f}"
        ])
    print(results_table)






