from pathlib import Path
import pickle
from pprint import pprint

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from mine_classification import params
from mine_classification.data_preparation import load_mine_data


def _get_predictions(
    model: Pipeline, soil_wetness: str, soil_type: str, n_increments: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heights = np.linspace(0, 0.2, num=n_increments)
    voltages = np.linspace(0, 10.6, num=n_increments)
    heights_grid, voltages_grid = np.meshgrid(heights, voltages)
    grids_shape = heights_grid.shape
    df_grid = pd.DataFrame(
        {"S_wet": soil_wetness, "S_type": soil_type, "V": voltages_grid.flatten(), "H": heights_grid.flatten()}
    )
    pred_grid = model.predict(df_grid)
    pred_grid = pred_grid.reshape(*grids_shape)
    classification_to_int = {mine_type: idx for idx, mine_type in enumerate(model.classes_)}
    return heights, voltages, np.vectorize(classification_to_int.get)(pred_grid)


def plot_decision_space(
    model: Pipeline, df_train: pd.DataFrame, df_test: pd.DataFrame, mark_test_data_idx: int | None = None
) -> None:
    """
    Plot the whole decision space of the land mine classification model by iterating over
    a grid of the input parameters and plotting the resulting predictions

    :param model: Pipeline used for prediction.
    :param df_train: Training data
    :param df_test: Test data
    :param mark_test_data_idx: Index of a measurement in the test data which to mark with a solid black
        "x" in the resulting plot
    """

    mpl.rcParams.update({"font.size": 14})
    fig, axis = plt.subplots(nrows=2, ncols=3)
    axis = axis.flatten()
    soil_types = [
        ("humid", "sandy"),
        ("humid", "limy"),
        ("humid", "humus"),
        ("dry", "sandy"),
        ("dry", "limy"),
        ("dry", "humus"),
    ]
    for ax_idx, (soil_wetness, soil_type) in enumerate(soil_types):
        heights, voltages, pred_grid = _get_predictions(model, soil_wetness, soil_type)

        soil_train = df_train[(df_train["S_wet"] == soil_wetness) & (df_train["S_type"] == soil_type)]
        soil_test = df_test[(df_test["S_wet"] == soil_wetness) & (df_test["S_type"] == soil_type)]
        classification_to_int = {mine_type: idx for idx, mine_type in enumerate(model.classes_)}
        true_train = np.vectorize(classification_to_int.get)(soil_train["M"])
        true_test = np.vectorize(classification_to_int.get)(soil_test["M"])

        accent = mpl.colormaps["Accent"].resampled(5)
        axis[ax_idx].set_title(f"{soil_type} - {soil_wetness}", fontdict={"fontsize": 10})
        color_mesh = axis[ax_idx].pcolormesh(heights * 1e2, voltages, pred_grid, cmap=accent)
        if ax_idx == (len(soil_types) - 1):
            formatter = plt.FuncFormatter(lambda val, loc: model.classes_[int(val)])
            fig.colorbar(color_mesh, ax=axis[ax_idx], ticks=[0, 1, 2, 3, 4], format=formatter)
        axis[ax_idx].scatter(
            soil_train["H"] * 1e2, soil_train["V"], c=true_train, cmap=accent, edgecolors=["black"], s=30
        )
        axis[ax_idx].scatter(
            soil_test["H"] * 1e2, soil_test["V"], c=true_test, cmap=accent, edgecolors=["black"], marker="X", s=30
        )
        axis[ax_idx].set_xlabel("Height in cm")
        axis[ax_idx].set_ylabel("Voltage in V")
        if mark_test_data_idx is not None:
            datapoint_to_mark = df_test.iloc[mark_test_data_idx]
            if (datapoint_to_mark["S_wet"] == soil_wetness) and (datapoint_to_mark["S_type"] == soil_type):
                axis[ax_idx].scatter(
                    datapoint_to_mark["H"] * 1e2,
                    datapoint_to_mark["V"],
                    marker="x",
                    s=200,
                    facecolor="black",
                    linewidths=3,
                )
    plt.subplots_adjust(top=0.95)
    plt.show()


def explain_model(pipeline_file: Path, processing_info: params.Preprocessing) -> None:
    with open(pipeline_file, "rb") as file:
        training_results = pickle.load(file)

    pprint(training_results)
    observe_test_datapoint_idx = 10
    pipeline = training_results["pipeline"]
    df_train, df_test = load_mine_data(processing_info.random_train_test_split, processing_info.soil_treatment)
    X_train = pipeline[:-1].transform(df_train[["V", "H", "S_type", "S_wet"]])
    X_test = pipeline[:-1].transform(df_test[["V", "H", "S_type", "S_wet"]])

    if isinstance(pipeline["classify"], DecisionTreeClassifier):
        plot_tree(
            pipeline["classify"],
            fontsize=10,
            feature_names=list(X_train.columns),
            class_names=list(pipeline["classify"].classes_),
        )
        plt.show()

    # feature importance on training data
    X100 = shap.utils.sample(X_train, nsamples=100)
    explainer = shap.Explainer(pipeline["classify"].predict_proba, X100)
    shap_values_train = explainer(X_train)
    shap_values_train_by_mine_type = {
        mine_type: shap_values_train[..., idx] for idx, mine_type in enumerate(pipeline["classify"].classes_)
    }
    shap.plots.bar(shap_values_train_by_mine_type)

    plot_decision_space(pipeline, df_train, df_test, mark_test_data_idx=observe_test_datapoint_idx)

    # explain prediction for single point of test data
    shap_values_test = explainer(X_test)
    predicted_probas = np.sum(shap_values_test[observe_test_datapoint_idx, ...].values, axis=0)
    predicted_class_idx = np.argmax(predicted_probas)
    predicted_class = pipeline["classify"].classes_[predicted_class_idx]
    print(f"For selected point {predicted_class} was predicted.")
    shap.plots.waterfall(shap_values_test[observe_test_datapoint_idx, :, predicted_class_idx])


if __name__ == "__main__":
    model_file = params.REPO_ROOT / r"outs\MLPClassifier.pkl"
    explain_model(model_file, params.Preprocessing())
