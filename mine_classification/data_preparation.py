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

SOIL_WETNESS = {1: "dry", 2: "dry", 3: "dry", 4: "humid", 5: "humid", 6: "humid"}


SOIL_TYPE = {1: "sandy", 2: "humus", 3: "limy", 4: "sandy", 5: "humus", 6: "limy"}


MINE_TYPE = {
    1: "no_mine",
    2: "anti_tank",
    3: "anti_personnel",
    4: "booby_trapped_anti_personnel",
    5: "m14_anti_personnel",
}


def _remove_soil_type(x: pd.DataFrame) -> pd.DataFrame:
    return x.drop(columns=["S_type"])


def _remove_soil_wetness(x: pd.DataFrame) -> pd.DataFrame:
    return x.drop(columns=["S_wet"])


def load_mine_data(
    random_train_test_split: bool,
    soil_transformation: params.SoilTransformation = params.SoilTransformation.NORMAL,
    stdv_voltage_noise_on_test_data: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the land mines dataset. The features are converted to their
    original range (e.g. the voltage is rescaled to go from 0 to 10.6V as
    mentioned in [1] instead of just going from 0 to 1).
    Additionally, the soil type is split into soil wetness (humid/dry) and actual type of the soil
    (sandy, humus, limy). This behavior can be adjusted by changing the
    `soil_transformation` input.

    :param random_train_test_split: Whether to split of 1/3 of the dataset as test randomly (`True`) or by
        taking the split as suggested by the land mines Excel file (i.e. the point at which the labels
        repeat for the first time).
    :param soil_transformation: How to treat the soil type, see `SoilTransformation` enum for details.
    :param stdv_voltage_noise_on_test_data: In Volt. If set, adds noise to the measured voltages.
    :return: Train and test data
    """
    df = pd.read_excel(params.DATA_BASE_DIR / "Mine_Dataset.xls", sheet_name="Normalized_Data")

    # undo normalization, see [1] for details
    max_voltage = 10.6
    max_height = 0.2
    df["V"] = np.array(df["V"] * max_voltage)  # voltage in V
    df["H"] = np.array(df["H"] * max_height)  # height in m
    df["S"] = np.array(df["S"] * 5 + 1).astype(int)  # different soil types as used in [1]

    # labels
    df["M"] = np.array(MINE_TYPE[m_type] for m_type in df["M"])

    if soil_transformation == params.SoilTransformation.RANDOMIZE:
        df["S_wet"] = np.random.choice(list(SOIL_WETNESS.values()), size=df.shape[0])
        df["S_type"] = np.random.choice(list(SOIL_TYPE.values()), size=df.shape[0])
    else:
        # split "S" into wetness and actual soil type
        df["S_wet"] = np.array(SOIL_WETNESS[s_type] for s_type in df["S"])
        df["S_type"] = np.array(SOIL_TYPE[s_type] for s_type in df["S"])
    df = df.drop(columns="S")

    if random_train_test_split:
        df_train, df_test = train_test_split(df, test_size=1 / 3, stratify=df["M"])
    else:
        df_train = df.iloc[:225, :]
        df_test = df.iloc[225:, :]

    if stdv_voltage_noise_on_test_data is not None:
        df_test["V"] += np.random.normal(loc=0, scale=stdv_voltage_noise_on_test_data, size=df_test["V"].count())
    return df_train, df_test


def make_processing_pipeline(
    soil_treatment: params.SoilTransformation,
    classifier: ClassifierMixin | None = None,
) -> Pipeline:
    """
    Build a sci-kit learn pipeline that preprocesses the data and serves
    it to the classifier.

    :param soil_treatment: How to treat the soil type, see `SoilTransformation` enum for details.
    :param classifier: Sci-kit learn model to use for classification
    :return: Pipeline; use its fit/predict member functions on the unprocessed data
    """

    if soil_treatment in [params.SoilTransformation.IGNORE_SOIL_TYPE, params.SoilTransformation.IGNORE_SOIL]:
        pipeline = Pipeline([("remove_soil_type", FunctionTransformer(_remove_soil_type))])
    else:
        soil_encoder = ColumnTransformer(
            [("soil_type", OneHotEncoder(sparse_output=False), ["S_type"])],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        soil_encoder.set_output(transform="pandas")
        pipeline = Pipeline([("encode_soil_type", soil_encoder)])

    if soil_treatment in [params.SoilTransformation.IGNORE_SOIL_WET, params.SoilTransformation.IGNORE_SOIL]:
        pipeline.steps.append(("remove_soil_wet", FunctionTransformer(_remove_soil_wetness)))
    else:
        wetness_encoder = ColumnTransformer(
            [("wetness", OrdinalEncoder(categories=[["dry", "humid"]]), ["S_wet"])],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        wetness_encoder.set_output(transform="pandas")
        pipeline.steps.append(("encode_soil_wetness", wetness_encoder))

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
