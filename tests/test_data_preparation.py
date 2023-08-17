from copy import deepcopy

import pandas as pd
import plotly.express as px

from mine_detection import data_preparation as dpp


def test_load_mine_data():
    df_train, df_test = dpp.load_mine_data()
    value_counts_train = df_train["M"].value_counts()
    value_counts_test = df_test["M"].value_counts()

    assert value_counts_train["no_mine"] + value_counts_test["no_mine"] == 71
    assert value_counts_train["anti_tank"] + value_counts_test["anti_tank"] == 70
    assert value_counts_train["anti_personnel"] + value_counts_test["anti_personnel"] == 66
    assert value_counts_train["booby_trapped_anti_personnel"] + value_counts_test["booby_trapped_anti_personnel"] == 66
    assert value_counts_train["m14_anti_personnel"] + value_counts_test["m14_anti_personnel"] == 65

    do_plots = True
    if do_plots:
        # warning: px.scatter_matrix seems to be quite buggy at the moment,
        #   meaning that the argument `color="M"` does not work on the test
        #   data split via `iloc` from the rest of the dataframe (which is
        #   why the plots are generated in such an obtuse manner) and apparently,
        #   labels are sometimes changed when plotting with `color="M", symbol="is_train"`.
        df_train["is_train"] = True
        df_test["is_train"] = False
        df = pd.concat([df_test, df_train])
        fig = px.scatter_matrix(df[df["is_train"] == True].drop(columns=["is_train"]))
        fig.show()

        fig = px.scatter_matrix(df[df["is_train"] == False].drop(columns=["is_train"]))
        fig.show()



def test_preprocessing_pipeline():
    df_train, _ = dpp.load_mine_data()
    pipeline = dpp.get_preprocessing_pipeline()
    blup = pipeline.fit_transform(df_train[["V", "H", "S_wet", "S_type"]])
    print(pipeline.get_feature_names_out())
    print(blup)