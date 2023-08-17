import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


from mine_detection.data_preparation import get_preprocessing_pipeline, load_mine_data


if __name__ == "__main__":
    df_train, df_test = load_mine_data()
    X = df_train[["V", "H", "S_type", "S_wet"]]
    y = df_train["M"]
    X_test = df_test[["V", "H", "S_type", "S_wet"]]
    y_test = df_test["M"]



    pipeline = Pipeline(
        [
            ("preprocessing", get_preprocessing_pipeline()),
            ("classify", MLPClassifier(max_iter=1000))
        ]
    )


    param_distribution = {
        #"classify__weights": ["uniform", "distance"],
        #"classify__n_estimators": [100, 200]
        "classify__hidden_layer_sizes": [(100,), (500, ), (50, )]
    }
    k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    search = RandomizedSearchCV(pipeline, param_distribution, n_iter=500, verbose=2, scoring="accuracy", cv=k_fold)
    search.fit(X, y)
    print(search.best_params_)
    print(search.best_score_)

    pipeline.set_params(**search.best_params_)
    pipeline.fit(X, y=y)
    print(pipeline.score(X, y=y))
    print(pipeline.score(X_test, y=y_test))





    # scores = cross_validate(pipeline, X, y, scoring="accuracy", cv=5, verbose=1)
    # print(scores)


    # model.fit(df_train[["V", "H", "S"]], df_train["M"])
    # print(model.predict(np.array([[1, 1, 5], [0, 0, 2]])))
    # print(scores)





