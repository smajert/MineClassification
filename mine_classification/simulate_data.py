import math

import numpy as np
import pandas as pd

from mine_classification.data_preparation import SOIL_TYPE, SOIL_WETNESS

# 'a' and 'b' of linear relationship voltage = - a * h + b where h is height (maximum h = 0.2m)
# 'a' is in Volt/Meter and b is in Volt
MINE_EFFECT = {
    "no_mine": (0.25, 3),  # goes from 3.00 V to 2.95 V at maximum height
    "anti_tank": (15, 10.2),  # goes from 10.2 V to 7.2 V at maximum height
    "anti_personnel": (5, 4),  # goes from 4.0 V to  3.0 V at maximum height
    "booby_trapped_anti_personnel": (6, 4.2),  # goes from 4.2 V to 3.0 V at maximum height
    "m14_anti_personnel": (20, 8),  # goes from 8.0 V to 4.0 V at maximum height
}


def simulate_samples(n_samples: int, is_for_training: bool, voltage_noise_stdv: float = 0.5) -> pd.DataFrame:
    """
    Create an artificial land mines dataset with a linear relationship between voltage and height, where
    slope and intercept depend on the mine type.

    :param n_samples: Amount of samples to generate
    :param is_for_training: Whether the samples are for training or testing (grid heights for training, random
        heights for testing)
    :param voltage_noise_stdv: Standard deviation of noise that is superimposed on the voltage
    :return: Simulated samples
    """
    samples_per_scan = 8

    max_height = 0.2
    if is_for_training:
        heights = np.concatenate(
            [np.linspace(0, max_height, num=samples_per_scan)] * math.ceil((n_samples / (samples_per_scan - 1)))
        )[:n_samples]
    else:
        heights = np.random.uniform(low=0, high=max_height, size=n_samples)

    mine_type = np.random.choice(list(MINE_EFFECT.keys()), replace=True, size=n_samples)

    a = np.vectorize(lambda x: MINE_EFFECT[x][0])(mine_type)
    b = np.vectorize(lambda x: MINE_EFFECT[x][1])(mine_type)
    voltage = -a * heights + b
    voltage += np.random.normal(loc=0, scale=voltage_noise_stdv, size=n_samples)
    min_voltage, max_voltage = 1.0, 10.2
    voltage = np.clip(voltage, a_min=min_voltage, a_max=max_voltage)

    # soil type and wetness do not matter:
    soil_type = np.random.choice(list(SOIL_TYPE.values()), replace=True, size=n_samples)
    soil_wetness = np.random.choice(list(SOIL_WETNESS.values()), replace=True, size=n_samples)

    return pd.DataFrame({"H": heights, "V": voltage, "M": mine_type, "S_wet": soil_wetness, "S_type": soil_type})


def load_simulated_mine_data(n_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get a simulated version of the land mines dataset.
    Assume voltage linearly related to height and is
    higher for certain mine types.

    :param n_samples: Amount of samples to produce
    :return: Simulated training and test data
    """

    n_training_samples = int(n_samples * (2 / 3))
    n_test_samples = n_samples - n_training_samples

    df_train = simulate_samples(n_training_samples, is_for_training=True)
    df_test = simulate_samples(n_test_samples, is_for_training=False)

    return df_train, df_test
