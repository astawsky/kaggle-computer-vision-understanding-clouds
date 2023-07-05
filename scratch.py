import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

image_size = (1400, 2100)
batch_size = 500
validation_split = 0.3
subset = "both"
seed = 42
image_directory = "./data/raw/train_images"

import os
ll = os.walk(image_directory)

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    image_directory,
    validation_split=validation_split,
    subset=subset,
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
)


print('end')