from enum import auto, Enum
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
DATA_BASE_DIR = REPO_ROOT / "data"
OUTS_BASE_DIR = REPO_ROOT / "outs"


class SoilTransformation(Enum):
    IGNORE_SOIL_TYPE = auto()  # do not consider soil type
    NORMAL = auto()  # split into soil wetness and type
    RANDOMIZE = auto()  # insert random values for soil wetness and type


class Preprocessing:
    random_train_test_split: bool = True  # randomly take 1/3 of data as test or use cut-off suggested bei excel
    soil_treatment: SoilTransformation = SoilTransformation.NORMAL  # how to deal with the soil type feature

