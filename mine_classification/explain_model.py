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


def plot_decision_space(model: Pipeline, df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    soil_wetness = "humid"
    soil_type = "sandy"
    df_train = df_train[(df_train['S_wet'] == soil_wetness) & (df_train['S_type'] == soil_type)]
    df_test = df_test[(df_test['S_wet'] == soil_wetness) & (df_test['S_type'] == soil_type)]

    n_background_increments = 200
    heights = np.linspace(0, 0.2, num=n_background_increments)
    voltages = np.linspace(0, 10.6, num=n_background_increments)
    heights_grid, voltages_grid = np.meshgrid(heights, voltages)
    grids_shape = heights_grid.shape
    df_grid = pd.DataFrame({
        "S_wet": soil_wetness, "S_type": soil_type, "V": voltages_grid.flatten(), "H": heights_grid.flatten()
    })
    pred_grid = model.predict(df_grid)
    pred_grid = pred_grid.reshape(*grids_shape)
    classification_to_int = {mine_type: idx for idx, mine_type in enumerate(model.classes_)}
    pred_grid = np.vectorize(classification_to_int.get)(pred_grid)

    true_train = np.vectorize(classification_to_int.get)(df_train["M"])
    true_test = np.vectorize(classification_to_int.get)(df_test["M"])

    mpl.rcParams.update({'font.size': 16})
    plt.figure()
    formatter = plt.FuncFormatter(lambda val, loc: model.classes_[int(val)])
    accent = mpl.colormaps['Accent'].resampled(5)
    plt.pcolormesh(heights * 1e2, voltages, pred_grid, cmap=accent)
    plt.colorbar(ticks=[0, 1, 2, 3, 4], format=formatter)
    plt.scatter(df_train["H"] * 1e2, df_train["V"], c=true_train, cmap=accent, edgecolors=["black"], s=30)
    plt.scatter(df_test["H"] * 1e2, df_test["V"], c=true_test, cmap=accent, edgecolors=["black"], marker="X", s=30)
    plt.xlabel("Height in cm")
    plt.ylabel("Voltage in V")

    plt.show()


def explain_model(pipeline_file: Path, processing_info: params.Preprocessing) -> None:
    with open(pipeline_file, "rb") as file:
        training_results = pickle.load(file)

    pprint(training_results)
    pipeline = training_results["pipeline"]
    df_test, df_train = load_mine_data(processing_info.random_train_test_split, processing_info.soil_treatment)
    plot_decision_space(pipeline, df_train, df_test)

    # X_test = pipeline[:-1].transform(df_test[["V", "H", "S_type", "S_wet"]])
    # if isinstance(pipeline["classify"], DecisionTreeClassifier):
    #     plot_tree(
    #         pipeline["classify"], fontsize=10, feature_names=X_test.columns, class_names=pipeline["classify"].classes_
    #     )
    #     plt.show()
    # explainer = shap.KernelExplainer(pipeline["classify"].predict_proba, shap.kmeans(X_test, 5))
    # shap_values = explainer.shap_values(X_test)
    # plot = shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
    # shap.save_html(str(params.OUTS_BASE_DIR / f"{pipeline_file.stem}.html"), plot)



if __name__ == "__main__":
    model_file = Path(r"C:\my_files\Projekte\MineClassification\outs\MLPClassifier.pkl")
    explain_model(model_file, params.Preprocessing())