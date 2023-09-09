import pandas as pd
import plotly.express as px

from mine_classification import simulate_data


def test_load_simulated_data():
    df_train, df_test = simulate_data.load_simulated_mine_data(338)
    value_counts_train = df_train["M"].value_counts()
    value_counts_test = df_test["M"].value_counts()

    assert value_counts_train.sum() + value_counts_test.sum() == 338

    do_plot = True
    if do_plot:
        df_train["is_train"] = True
        df_test["is_train"] = False
        df = pd.concat([df_test, df_train])
        fig = px.scatter_matrix(df, color="M", dimensions=["is_train", "V", "H", "S_type", "S_wet"])
        fig.show()
