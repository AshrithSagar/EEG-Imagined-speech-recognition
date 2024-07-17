# Workflows

Execute the various workflows using the command:

```bash
python3 workflows/{file}.py
```

from the root of the project directory.

| Workflow                                  | Description                                                                                                                                      |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `download-karaone.py`                     | Download the KaraOne dataset from the KaraOne website. Saves the dataset to the `{raw_data_dir}` folder.                                         |
| `features-karaone.py`, `features-feis.py` | Preprocess EEG data to extract relevant features for different `epoch_types`. Saves processed data as `.fif` files to the `{filtered_data_dir}`. |
| `ifs-classifier.py`                       | Train a classifier using preprocessed EEG data, utilizing Information Set Theory.                                                                |
| `flatten-classifier.py`                   | Flatten feature matrix to a vector for classifier input. Specify number of features in `features_select_k_best[k]` (`int`).                      |
| `flatten-classifier-KBest.py`             | Run classifier over multiple `k` values for feature selection. Iterate over values in `features_select_k_best[k]` (`list[int]`).                 |
