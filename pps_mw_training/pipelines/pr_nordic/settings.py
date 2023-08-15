from pathlib import Path
import os


MODEL_CONFIG_PATH = Path(
    os.environ.get("MODEL_CONFIG_PR_NORDIC", "/tmp")
)
TRAINING_DATA_PATH = Path(
    os.environ.get("TRAINING_DATA_PATH_PR_NORDIC", "/tmp")
)
# model parameters
CHANNELS = [14, 15, 16, 17, 18]
QUANTILES = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
N_UNET_BASE = 16
N_FEATURES = 128
N_LAYERS = 4
# training parameters
N_EPOCHS = 8
BATCH_SIZE = 16
TRAIN_FRACTION = 0.7
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
FILL_VALUE = -1.1
# learning rate parameters
INITIAL_LEARNING_RATE = 0.0001
FIRST_DECAY_STEPS = 1000
T_MUL = 2.0
M_MUL = 1.0
ALPHA = 0.0