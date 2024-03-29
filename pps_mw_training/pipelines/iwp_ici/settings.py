from pathlib import Path
from typing import Dict, List, Union
import os


ICI_RETRIEVAL_DB_FILE = Path(
    os.environ.get(
        "ICI_RETRIEVAL_DB_FILE", "/tmp/ici_retrieval_database.nc",
    )
)
MODEL_CONFIG_PATH = Path(os.environ.get("MODEL_CONFIG_IWP_ICI", "/tmp"))


# model parameters
QUANTILES = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
N_HIDDEN_LAYERS = 4
N_NEURONS_PER_HIDDEN_LAYER = 128
ACTIVATION = "relu"
# training parameters
BATCH_SIZE = 4096
N_EPOCHS = 256
TRAIN_FRACTION = 0.7
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
NOISE = 1.0
FILL_VALUE = -2.
MISSING_FRACTION = 0.1
# learning rate parameters
INITIAL_LEARNING_RATE = 0.0001
FIRST_DECAY_STEPS = 1000
T_MUL = 2.0
M_MUL = 1.0
ALPHA = 0.0
INPUT_PARAMS: List[Dict[str, Union[str, float]]] = [
    {
        "name": "DTB_ICI_DB_ICI_01V",
        "scale": "linear",
        "min": -170.,
        "max": 30.,
    },
    {
        "name": "DTB_ICI_DB_ICI_02V",
        "scale": "linear",
        "min": -155.,
        "max": 35.,
    },
    {
        "name": "DTB_ICI_DB_ICI_03V",
        "scale": "linear",
        "min": -145.,
        "max": 35.,
    },
    {
        "name": "DTB_ICI_DB_ICI_04V",
        "scale": "linear",
        "min": -195.,
        "max": 40.,
    },
    {
        "name": "DTB_ICI_DB_ICI_04H",
        "scale": "linear",
        "min": -195.,
        "max": 50.,
    },
    {
        "name": "DTB_ICI_DB_ICI_05V",
        "scale": "linear",
        "min": -185.,
        "max": 30.,
    },
    {
        "name": "DTB_ICI_DB_ICI_06V",
        "scale": "linear",
        "min": -180.,
        "max": 30.,
    },
    {
        "name": "DTB_ICI_DB_ICI_07V",
        "scale": "linear",
        "min": -165.,
        "max": 35.,
    },
    {
        "name": "DTB_ICI_DB_ICI_08V",
        "scale": "linear",
        "min": -165.,
        "max": 35.,
    },
    {
        "name": "DTB_ICI_DB_ICI_09V",
        "scale": "linear",
        "min": -155.,
        "max": 35.,
    },
    {
        "name": "DTB_ICI_DB_ICI_10V",
        "scale": "linear",
        "min": -135.,
        "max": 25.,
    },
    {
        "name": "DTB_ICI_DB_ICI_11V",
        "scale": "linear",
        "min": -160.,
        "max": 30.,
    },
    {
        "name": "DTB_ICI_DB_ICI_11H",
        "scale": "linear",
        "min": -160.,
        "max": 30.,
    },
    {
        "name": "SurfType",
        "scale": "linear",
        "min": 0.,
        "max": 4.,
    },
    {
        "name": "SurfPres",
        "scale": "linear",
        "min": 50000.,
        "max": 106000.,
    },
    {
        "name": "SurfTemp",
        "scale": "linear",
        "min": 210.,
        "max": 320.,
    },
    {
        "name": "SurfWind",
        "scale": "linear",
        "min": 0.,
        "max": 35.,
    }
]
OUTPUT_PARAMS: List[Dict[str, Union[str, float]]] = [
    {
        "name": "TCWV",
        "scale": "log",
        "min": 0.,
        "max": 80.,
    },
    {
        "name": "LWP",
        "scale": "log",
        "min": 0.,
        "max": 2.,
    },
    {
        "name": "RWP",
        "scale": "log",
        "min": 0.,
        "max": 4.,
    },
    {
        "name": "IWP",
        "scale": "log",
        "min": 0.,
        "max": 35.,
    },
    {
        "name": "Zmean",
        "scale": "linear",
        "min": 0.,
        "max": 19000.,
    },
    {
        "name": "Dmean",
        "scale": "linear",
        "min": 0.,
        "max": 0.0017,
    }
]
