import pathlib
import numpy as np

#Hyperparameters of the model
IMAGE_SHAPE = (224, 224)
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 3
EPOCHS = 300
PATIENCE = 15
hidden_units = []
additional_layers = 3
Dropout_Classification = 0.0 
OUTPUT_CHANNELS = 3  # RGB
dropout_rate = 0.05

#Dir of training and validation data
TRAIN_DIR = r"Dataset\train"
VAL_DIR = r"Dataset\validation"
Test_DIR = r"Dataset\test"

#Class names extraction
CLASS_NAMES = np.array([
    item.name for item in pathlib.Path(TRAIN_DIR).glob("*")
    if item.name != "LICENSE.txt"
])

#Ghost_DeepResNet_Model parameters 
STAGES = {
    "stage1": dict(
        num_filters      = 8,
        kernel_size      = (1, 1),
        dw_kernel_size   = (3, 3),
        strides          = (1, 1),
    ),
    "stage2": dict(
        num_filters      = 16,
        kernel_size      = (1, 1),
        dw_kernel_size   = (3, 3),
        strides          = (1, 1),
    ),
    "stage3": dict(
        num_filters      = 32,
        kernel_size      = (1, 1),
        dw_kernel_size   = (3, 3),
        strides          = (1, 1),
    ),
    "stage4": dict(
        num_filters      = 64,
        kernel_size      = (1, 1),
        dw_kernel_size   = (3, 3),
        strides          = (1, 1),
    ),
}
