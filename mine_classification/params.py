from enum import auto, Enum
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
DATA_BASE_DIR = REPO_ROOT / "data"
OUTS_BASE_DIR = REPO_ROOT / "outs"


class SoilTransformation(Enum):
    # split original soil type into soil wetness and actual soil type and:
    IGNORE_SOIL_TYPE = auto()  # just consider soil wetness, not actual soil type
    IGNORE_SOIL_WET = auto()  # just consider actual soil type, not soil wetness
    IGNORE_SOIL = auto()  # Do not consider soil type
    NORMAL = auto()  # take into account both soil wetness and actual soil type
    RANDOMIZE = auto()  # insert random values for soil wetness and actual soil type


class Preprocessing:
    random_train_test_split: bool = True  # randomly take 1/3 of data or split at index 225 (as suggested by excel)
    simulated_data: bool = False  # whether to use real data or simulated data
    soil_treatment: SoilTransformation = SoilTransformation.NORMAL  # how to deal with the soil type feature

