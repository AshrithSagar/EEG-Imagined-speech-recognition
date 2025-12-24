"""
tfr-classifier.py
Run classifier on the TFR dataset
"""

import os
import sys

sys.path.append(os.getcwd())
import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from eeg_isr.config import Config, fetch_dataset
from eeg_isr.tfr import TFRDataset

keras.backend.clear_session()
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices("GPU")[0], True
)


def main(args):
    t_args = args["tfr"]

    dataset_name = args.get("_select").get("dataset")
    _dataset = fetch_dataset(dataset_name)
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
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if trial_size is None:
        subset_size = len(y_train)
    elif isinstance(trial_size, float) and trial_size <= 1.0:
        subset_size = int(trial_size * len(y_train))
    else:
        subset_size = trial_size

    x_train, y_train = x_train[:subset_size], y_train[:subset_size]
    x_test, y_test = x_test[:subset_size], y_test[:subset_size]
    tfr_ds.split_info(x_train, x_test, y_train, y_test)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(96, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(128, kernel_size=(7, 7), activation="relu"),
            keras.layers.Dropout(0.30),
            keras.layers.Conv2D(216, kernel_size=(7, 7), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(256, kernel_size=(9, 9), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_dir = t_args["model_base_dir"]
    os.makedirs(model_dir, exist_ok=True)

    filename = os.path.join(model_dir, "model.png")
    keras.utils.plot_model(model, to_file=filename, show_shapes=True)

    _history = model.fit(
        x_train,
        y_train,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
    )

    results = model.evaluate(x_test, y_test)
    print("Test Loss:", results[0])
    print("Test Accuracy:", results[1])

    model_save_path = os.path.join(model_dir, "model.h5")
    model.save(model_save_path)


if __name__ == "__main__":
    args = Config.from_args("Run classifier on the TFR dataset")
    main(args)
