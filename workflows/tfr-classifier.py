"""
tfr-classifier.py
"""

import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils.config import fetch_select, load_config
from utils.tfr import TFRDataset

tf.keras.backend.clear_session()
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices("GPU")[0], True
)


if __name__ == "__main__":
    args = load_config(config_file="config.yaml")
    t_args = load_config(config_file="config.yaml", key="tfr")

    dataset_name = args.get("_select").get("dataset")
    dataset = fetch_select("dataset", dataset_name)
    d_args = args[dataset_name.lower()]

    channel = d_args["channels"][0]
    num_classes = 11
    input_shape = (256, 256, 1)
    max_epochs = t_args["max_epochs"]
    batch_size = t_args["batch_size"]
    trial_size = t_args["trial_size"]

    tfr_ds = TFRDataset(dataset_dir=d_args["tfr_dataset_dir"])
    tfr_ds.load(channel=channel, verbose=False)
    tfr_ds.dataset_info()

    x_train, x_test, y_train, y_test = train_test_split(
        tfr_ds.dataset,
        tfr_ds.class_labels,
        test_size=0.2,
        shuffle=True,
        random_state=42,
        stratify=tfr_ds.class_labels,
    )

    # Reshape to explicitly have 1 channel
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # One hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    if trial_size is None:
        subset_size = len(y_train)
    elif isinstance(trial_size, float) and trial_size <= 1.0:
        subset_size = int(trial_size * len(y_train))
    else:
        subset_size = trial_size

    x_train, y_train = x_train[:subset_size], y_train[:subset_size]
    x_test, y_test = x_test[:subset_size], y_test[:subset_size]
    tfr_ds.split_info(x_train, x_test, y_train, y_test)

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(96, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, kernel_size=(7, 7), activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Conv2D(216, kernel_size=(7, 7), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(256, kernel_size=(9, 9), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
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
