"""
classification-tf-1.py
"""
import os
import sys

sys.path.append(
    "/Users/ashrith/Documents/Hons/Repos/_/EEG-Imagined speech recognition/"
)
print(sys.path)
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from utils.tfr_dataset import TFRDataset

keras.backend.clear_session()

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices("GPU")[0], True
)


if __name__ == "__main__":
    SEED = 42
    channels = ["FC6", "FT8", "C5", "CP3", "P3", "T7", "CP5", "C3", "CP1", "C4"]

    tfr_dataset_dir = "/Users/ashrith/Documents/Hons/Datasets/KaraOne/TFR_dataset-1"
    channel = "FC6"

    num_classes = 11
    input_shape = (256, 256, 1)

    max_epochs = 5
    batch_size = 32

    testing = True

    tfr_ds = TFRDataset(dataset_dir=tfr_dataset_dir)
    tfr_ds.load(channel=channel, verbose=False)
    tfr_ds.dataset_info()

    x_train, x_test, y_train, y_test = train_test_split(
        tfr_ds.dataset,
        tfr_ds.class_labels,
        test_size=0.2,
        shuffle=True,
        random_state=SEED,
        stratify=tfr_ds.class_labels,
    )

    # Reshape to explicitly have 1 channel
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # One hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    if testing:
        subset_size = 5
        x_train, y_train = x_train[:subset_size], y_train[:subset_size]
        x_test, y_test = x_test[:subset_size], y_test[:subset_size]

    tfr_ds.split_info(x_train, x_test, y_train, y_test)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(96, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(7, 7), activation="relu"),
            layers.Dropout(0.30),
            layers.Conv2D(216, kernel_size=(7, 7), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(256, kernel_size=(9, 9), activation="relu"),
            layers.Flatten(),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
    )

    results = model.evaluate(x_test, y_test)
    print("Test Loss:", results[0])
    print("Test Accuracy:", results[1])

    model_save_path = os.path.join("files", "model.h5")
    model.save(model_save_path)
