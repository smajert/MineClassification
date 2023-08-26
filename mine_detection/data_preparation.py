from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, RobustScaler

from mine_detection import params

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


def load_mine_data(random_train_test_split: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(
        params.DATA_BASE_DIR / "Mine_Dataset.xls", sheet_name="Normalized_Data"
    )

    # undo normalization, see [XXX] for details
    df["V"] = np.array(df["V"] * 10.6)  # voltage in V
    df["H"] = np.array(df["H"] * 0.2)  # height in m
    df["S"] = np.array(df["S"] * 5 + 1).astype(int)  # different soil types as used in [XXX]

    df["M"] = np.array(MINE_TYPE[m_type] for m_type in df["M"])

    # split "S" into wetness and actual soil type
    df["S_wet"] = np.array(SOIL_WETNESS[s_type] for s_type in df["S"])
    df["S_type"] = np.array(SOIL_TYPE[s_type] for s_type in df["S"])
    df = df.drop(columns="S")

    if random_train_test_split:
        df_train, df_test = train_test_split(df, test_size=1/3, stratify=df["M"])
    else:
        df_train = df.iloc[:225, :]
        df_test = df.iloc[225:, :]

    return df_train, df_test


def get_preprocessing_pipeline() -> Pipeline:
    encoding = ColumnTransformer([
        ("wetness", OrdinalEncoder(categories=[["dry", "humid"]]), ["S_wet"]),
        ("soil_type", OneHotEncoder(sparse_output=False), ["S_type"]),
    ], remainder="passthrough")
    pipeline = Pipeline([
        ("encode", encoding),
        ("scaling", RobustScaler())
    ])
    pipeline.set_output(transform="pandas")
    return pipeline




