import os
import sys
import numpy as np
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC

sys.path.append(os.getcwd())
from utils.config import load_config, line_separator
from utils.karaone import KaraOneDataLoader


if __name__ == "__main__":
    args = load_config(key="karaone")

    subjects = [
        "MM05",
        "MM08",
        "MM09",
        "MM10",
        "MM11",
        "MM12",
        "MM14",
        "MM15",
        "MM16",
        "MM18",
        "MM19",
        "MM20",
        "MM21",
        "P02",
    ]

    pick_channels = ["FC6", "FT8", "C5", "CP3", "P3", "T7", "CP5", "C3", "CP1", "C4"]

    karaone = KaraOneDataLoader(
        data_dir=args["data_dir"],
        subjects=subjects[:1],
        sampling_freq=1000,
        num_milliseconds_per_trial=4900,
        verbose=False,
    )

    karaone.process_data(pick_channels=pick_channels, epoch_type="thinking")
    karaone.epochs_info(verbose=True)
    labels = karaone.all_epoch_labels

    karaone.extract_features(
        features_dir=args["features_dir"],
        epoch_type="thinking",
    )

    features = karaone.load_features(epoch_type="thinking", verbose=True)

    flattened_features, flattened_labels = karaone.flatten(
        features, labels, verbose=True
    )

    # Support Vector Machines (SVMs)
    test_mode = True
    subset_size = 100

    if test_mode:
        flattened_features = flattened_features[:subset_size]
        flattened_labels = flattened_labels[:subset_size]

    imputer = SimpleImputer(strategy="mean")
    imputer.fit(flattened_features)
    flattened_features_no_nan = imputer.transform(flattened_features)

    X_train, X_test, y_train, y_test = train_test_split(
        flattened_features_no_nan, flattened_labels, test_size=0.2, random_state=42
    )

    param_grid = {"C": [1, 10, 100], "gamma": [0.001, 0.0001]}
    # param_grid = {"C": [1, 10, 100, 1000], "gamma": [0.001, 0.0001, 0.00001]}
    svm_model = SVC(kernel="linear")

    grid_search = GridSearchCV(svm_model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(flattened_features_no_nan, flattened_labels)

    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    best_C = best_params["C"]
    best_gamma = best_params["gamma"]

    final_svm_model = SVC(kernel="linear", C=best_C, gamma=best_gamma)
    final_svm_model.fit(flattened_features_no_nan, flattened_labels)

    predictions = final_svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    model_dir = args["svm_model_dir"]
    filename = os.path.join(model_dir, "svm_model.pkl")
    joblib.dump(final_svm_model, filename)
