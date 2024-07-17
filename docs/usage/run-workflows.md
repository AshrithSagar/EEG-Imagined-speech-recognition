# Workflows

Execute the various workflows using the command:

```bash
python3 workflows/{file}.py
```

from the root of the project directory.

## `download-karaone.py`

Download the KaraOne dataset from the KaraOne website.

Saves the dataset to the `{raw_data_dir}` folder.

## `features-karaone.py` & `features-feis.py`

Preprocess the EEG data to extract relevant features.

Run for different `epoch_types`: { "thinking", "acoustic", ... }.

Saves processed data as `.fif` files to the `{filtered_data_dir}`.

## `ifs-classifier.py`

Train a classifier using the preprocessed EEG data.
Uses Information Set Theory to extract effective information from the feature matrix to be used as features.

## `flatten-classifier.py`

Flatten the feature matrix to a vector to be used as features.

**Configuration:** Specify the number of features to be selected in `features_select_k_best[k]` (`int`).

## `flatten-classifier-KBest.py`

Run the classifier over multiple `k`'s.

**Configuration:** Iterate over values in `features_select_k_best[k]` (`list[int]`).
