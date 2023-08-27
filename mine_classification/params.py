from enum import auto, Enum
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
DATA_BASE_DIR = REPO_ROOT / "data"
OUTS_BASE_DIR = REPO_ROOT / "outs"


class SoilTransformation(Enum):
    RANDOMIZE = auto()  # insert random values for soil wetness and type
    REMOVE = auto()  # do not consider soil type
    SPLIT_WET_TYPE = auto()  # split into soil wetness and type


class Preprocessing:
    random_train_test_split: bool = True  # randomly take 1/3 of data as test or use cut-off suggested bei excel
    soil: SoilTransformation = SoilTransformation.SPLIT_WET_TYPE  # how to deal with the soil type feature

