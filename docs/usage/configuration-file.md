# Configuration file

The configuration file `config.yaml` contains the paths to the data files and the parameters for the different workflows.

Refer to [config-template.yaml](config-template.yaml) for the template.

## _select

| Parameter  | Type    | Description                                                                     |
| ---------- | ------- | ------------------------------------------------------------------------------- |
| classifier | `(str)` | Select one from { RegularClassifier, ClassifierGridSearch, EvaluateClassifier } |
| dataset    | `(str)` | Select one from { KaraOne, FEIS }                                               |

## classifier

| Parameter              | Type                                                 | Description                                                                                                 |
| ---------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| features_select_k_best | `k`: `(int \| list[int])` <br> `score_func`: `(str)` | Number of top features to select <br> Name of the score function for ranking features (pearsonr, f_classif) |
| model_base_dir         | `(path)`                                             | Preferably use `files/Models/`                                                                              |
| models                 | `(list[str])`                                        | List of directory names containing `model.py` within them                                                   |
| n_splits               | `(int)`                                              | Number of splits in cross-validation                                                                        |
| random_state           | `(int)`                                              | Seed value                                                                                                  |
| test_size              | `(float)`                                            | Size of test split                                                                                          |
| trial_size             | `(float / null)`                                     | Fraction of dataset for testing purposes (use `null` for entire dataset)                                    |

## feis

| Parameter    | Type                            | Description                                                              |
| ------------ | ------------------------------- | ------------------------------------------------------------------------ |
| epoch_type   | `(str)`                         | One from { thinking, speaking, stimuli }                                 |
| features_dir | `(path)`                        | Preferably use `files/Features/FEIS/features-1/`                         |
| raw_data_dir | `(path)`                        | Preferably use `files/Data/FEIS/data_eeg/`                               |
| subjects     | `(all / list[int] / list[str])` | Specify subjects to use (use `'all'` for all subjects)                   |
| tasks        | `(list[int])`                   | Available tasks: `[0]`. Refer `utils/feis.py:FEISDataLoader.get_task();` |

## karaone

| Parameter         | Type                            | Description                                                                                |
| ----------------- | ------------------------------- | ------------------------------------------------------------------------------------------ |
| epoch_type        | `(str)`                         | One from { thinking, speaking, stimuli, clearing }                                         |
| features_dir      | `(path)`                        | Preferably use `files/Features/KaraOne/features-1/`                                        |
| filtered_data_dir | `(path)`                        | Preferably use `files/Data/KaraOne/EEG_data-1/`                                            |
| length_factor     | `(float)`                       | Determines the window length                                                               |
| overlap           | `(float)`                       | Determines the overlap between consecutive windows                                         |
| raw_data_dir      | `(path)`                        | Preferably use `files/Data/KaraOne/EEG_raw/`                                               |
| subjects          | `(all / list[int] / list[str])` | Specify subjects to use (use `'all'` for all subjects)                                     |
| tasks             | `(list[int])`                   | Available tasks: `[0, 1, 2, 3, 4]`. Refer `utils/karaone.py:KaraOneDataLoader.get_task();` |
| tfr_dataset_dir   | `(path)`                        | Preferably use `files/TFR/KaraOne/tfr_ds-1/`                                               |

## utils

| Parameter | Type     | Description                                           |
| --------- | -------- | ----------------------------------------------------- |
| path      | `(path)` | Absolute path to the project directory `utils` folder |
