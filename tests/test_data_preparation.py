from copy import deepcopy

import pandas as pd
import plotly.express as px

from mine_classification import data_preparation as dpp


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
        df = pd.concat([df_test, df_train])
        pipeline = dpp.get_preprocessing_pipeline()
        df_train_transformed = pipeline.fit_transform(df_train[["V", "H", "S_type", "S_wet"]])
        df_train_transformed["M"] = df_train["M"]
        fig = px.scatter_matrix(df_train_transformed, color="M", dimensions=['wetness__S_wet', 'soil_type__S_type_humus', 'soil_type__S_type_limy', 'soil_type__S_type_sandy', 'remainder__V', 'remainder__H'])
        #fig = px.scatter_matrix(df, color="M", dimensions=["is_train", "V", "H", "S_type", "S_wet"])
        fig.show()




def test_preprocessing_pipeline():
    df_train, _ = dpp.load_mine_data()
    pipeline = dpp.get_preprocessing_pipeline()
    blup = pipeline.fit_transform(df_train[["V", "H", "S_wet", "S_type"]])
    print(pipeline.get_feature_names_out())
    print(blup)