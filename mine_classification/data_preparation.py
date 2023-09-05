from interpret.glassbox import ExplainableBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier

from mine_classification import params

SOIL_WETNESS = {
    1: "dry",
    2: "dry",
    3: "dry",
    4: "humid",
    5: "humid",
    6: "humid"
}


SOIL_TYPE = {
    1: "sandy",
    2: "humus",
    3: "limy",
    4: "sandy",
    5: "humus",
    6: "limy"
}


MINE_TYPE = {
    1: "no_mine",
    2: "anti_tank",
    3: "anti_personnel",
    4: "booby_trapped_anti_personnel",
    5: "m14_anti_personnel"
}


def _remove_soil_cols(x: pd.DataFrame) -> pd.DataFrame:
    return x.drop(columns=["S_type", "S_wet"])


def load_mine_data(
    random_train_test_split: bool = params.Preprocessing.random_train_test_split,
    soil: params.SoilTransformation = params.Preprocessing.soil_treatment,
    stdv_voltage_noise_on_test_data: float | None = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(
        params.DATA_BASE_DIR / "Mine_Dataset.xls", sheet_name="Normalized_Data"
    )

    # undo normalization, see [XXX] for details
    max_voltage = 10.6
    max_height = 0.2
    df["V"] = np.array(df["V"] * max_voltage)  # voltage in V
    df["H"] = np.array(df["H"] * max_height)  # height in m
    df["S"] = np.array(df["S"] * 5 + 1).astype(int)  # different soil types as used in [XXX]

    # labels
    df["M"] = np.array(MINE_TYPE[m_type] for m_type in df["M"])

    if soil == params.SoilTransformation.RANDOMIZE:
        df["S_wet"] = np.random.choice(list(SOIL_WETNESS.values()), size=df.shape[0])
        df["S_type"] = np.random.choice(list(SOIL_TYPE.values()), size=df.shape[0])
    else:
        # split "S" into wetness and actual soil type
        df["S_wet"] = np.array(SOIL_WETNESS[s_type] for s_type in df["S"])
        df["S_type"] = np.array(SOIL_TYPE[s_type] for s_type in df["S"])
    df = df.drop(columns="S")

    if random_train_test_split:
        df_train, df_test = train_test_split(df, test_size=1/3, stratify=df["M"])
    else:
        df_train = df.iloc[:225, :]
        df_test = df.iloc[225:, :]

    if stdv_voltage_noise_on_test_data is not None:
        df_test["V"] += np.random.normal(loc=0, scale=stdv_voltage_noise_on_test_data, size=df_test["V"].count())
    return df_train, df_test


def make_processing_pipeline(
        classifier: ClassifierMixin | None = None,
        soil_treatment: params.SoilTransformation = params.Preprocessing.soil_treatment
) -> Pipeline:
    if soil_treatment == params.SoilTransformation.REMOVE:
        encoding = FunctionTransformer(_remove_soil_cols)
    else:
        encoding = ColumnTransformer([
            ("wetness", OrdinalEncoder(categories=[["dry", "humid"]]), ["S_wet"]),
            ("soil_type", OneHotEncoder(sparse_output=False), ["S_type"]),
        ], remainder="passthrough")

    pipeline = Pipeline([("encode", encoding)])
    match classifier:
        case None:
            pass
        case [DecisionTreeClassifier(), ExplainableBoostingClassifier()]:
            pipeline.steps.append(("classify", classifier))
        case _:
            pipeline.steps.append(("scaling", RobustScaler()))
            pipeline.steps.append(("classify", classifier))

    pipeline.set_output(transform="pandas")

    return pipeline




