from pathlib import Path
import pickle
from pprint import pprint

from matplotlib import pyplot as plt
import shap
from sklearn.tree import DecisionTreeClassifier, plot_tree

from mine_classification import params
from mine_classification.data_preparation import load_mine_data


def explain_model(pipeline_file: Path) -> None:
    with open(pipeline_file, "rb") as file:
        training_results = pickle.load(file)

    pprint(training_results)
    pipeline = training_results["pipeline"]
    df_test, _ = load_mine_data()

    X_test = pipeline[:-1].transform(df_test[["V", "H", "S_type", "S_wet"]])
    if isinstance(pipeline["classify"], DecisionTreeClassifier):
        plot_tree(
            pipeline["classify"], fontsize=10, feature_names=X_test.columns, class_names=pipeline["classify"].classes_
        )
        plt.show()
    explainer = shap.KernelExplainer(pipeline["classify"].predict_proba, shap.kmeans(X_test, 5))
    shap_values = explainer.shap_values(X_test)
    plot = shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
    shap.save_html(str(params.OUTS_BASE_DIR / f"{pipeline_file.stem}.html"), plot)



if __name__ == "__main__":
    model_file = Path(r"C:\my_files\Projekte\MineClassification\outs\DecisionTreeClassifier.pkl")
    explain_model(model_file)