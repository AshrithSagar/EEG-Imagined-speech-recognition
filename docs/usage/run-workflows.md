# Workflows

Run the different workflows using `python3 workflows/*.py` from the project directory.

1. `download-karaone.py`:
Download the dataset into the {raw_data_dir} folder.

1. `features-karaone.py`, `features-feis.py`:
Preprocess the EEG data to extract relevant features.
Run for different epoch_types: { thinking, acoustic, ... }.
Also saves processed data as a `.fif` to {filtered_data_dir}.

1. `ifs-classifier.py`:
Train a machine learning classifier using the preprocessed EEG data.
Uses Information set theory to extract effective information from the feature matrix, to be used as features.

1. `flatten-classifier.py`:
Flattens the feature matrix to a vector, to be used as features.
Specify the number of features to be selected in features_select_k_best[k] (int).

1. `flatten-classifier-KBest.py`:
Run over multiple k's from features_select_k_best[k] (list[int]).
