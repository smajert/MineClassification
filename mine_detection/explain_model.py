from pathlib import Path
import pickle

import shap

from mine_detection import params
from mine_detection.data_preparation import load_mine_data


def explain_model(model_file: Path) -> None:
    with open(model_file, "rb") as file:
        model = pickle.load(file)["model"]
    df_test, _ = load_mine_data()

    X_test = model["preprocessing"].transform(df_test[["V", "H", "S_type", "S_wet"]])
    explainer = shap.KernelExplainer(model["classify"].predict_proba, shap.kmeans(X_test, 5))
    shap_values = explainer.shap_values(X_test)
    plot = shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
    shap.save_html(str(params.OUTS_BASE_DIR / f"{model_file.stem}.html"), plot)



if __name__ == "__main__":
    model_file = Path(r"C:\my_files\Projekte\MineDetection\outs\MLPClassifier.pkl")
    explain_model(model_file)