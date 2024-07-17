# Configuration file

The configuration file `config.yaml` contains the paths to the data files and the parameters for the different workflows.

Refer to [config-template.yaml](https://github.com/AshrithSagar/EEG-Imagined-speech-recognition/blob/main/config-template.yaml) for the template.

## _select

| Parameter  | Type  | Description                                                                     |
| ---------- | ----- | ------------------------------------------------------------------------------- |
| classifier | `str` | Select one from { RegularClassifier, ClassifierGridSearch, EvaluateClassifier } |
| dataset    | `str` | Select one from { KaraOne, FEIS }                                               |

## classifier

| Parameter               | Type                      | Description                                                              |
| ----------------------- | ------------------------- | ------------------------------------------------------------------------ |
| features_select_k_best: | `dict`                    |                                                                          |
| - k                     | `int` \| <br> `list[int]` | Number of top features to select                                         |
| - score_func            | `str`                     | Name of the score function for ranking features (pearsonr, f_classif)    |
|                         |                           |                                                                          |
| model_base_dir          | `path`                    | Preferably use `files/Models/`                                           |
| models                  | `list[str]`               | List of directory names containing `model.py` within them                |
| n_splits                | `int`                     | Number of splits in cross-validation                                     |
| random_state            | `int`                     | Seed value                                                               |
| test_size               | `float`                   | Size of test split                                                       |
| trial_size              | `float` \| <br> `null`    | Fraction of dataset for testing purposes (use `null` for entire dataset) |

## feis

| Parameter    | Type                                          | Description                                                                                                                                              |
| ------------ | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| epoch_type   | `str`                                         | One from { thinking, speaking, stimuli }                                                                                                                 |
| features_dir | `path`                                        | Preferably use `files/Features/FEIS/features-1/`                                                                                                         |
| raw_data_dir | `path`                                        | Preferably use `files/Data/FEIS/data_eeg/`                                                                                                               |
| subjects     | `all` \| <br> `list[int]` \| <br> `list[str]` | Specify subjects to use (use `'all'` for all subjects)                                                                                                   |
| tasks        | `list[int]`                                   | Available tasks: `[0]`. Refer [FEISDataLoader.get_task()](https://github.com/AshrithSagar/EEG-Imagined-speech-recognition/blob/main/utils/feis.py#L356); |

## karaone

| Parameter         | Type                                          | Description                                                                                                                                                                |
| ----------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| epoch_type        | `str`                                         | One from { thinking, speaking, stimuli, clearing }                                                                                                                         |
| features_dir      | `path`                                        | Preferably use `files/Features/KaraOne/features-1/`                                                                                                                        |
| filtered_data_dir | `path`                                        | Preferably use `files/Data/KaraOne/EEG_data-1/`                                                                                                                            |
| length_factor     | `float`                                       | Determines the window length                                                                                                                                               |
| overlap           | `float`                                       | Determines the overlap between consecutive windows                                                                                                                         |
| raw_data_dir      | `path`                                        | Preferably use `files/Data/KaraOne/EEG_raw/`                                                                                                                               |
| subjects          | `all` \| <br> `list[int]` \| <br> `list[str]` | Specify subjects to use (use `'all'` for all subjects)                                                                                                                     |
| tasks             | `list[int]`                                   | Available tasks: `[0, 1, 2, 3, 4]`. Refer [KaraOneDataLoader.get_task()](https://github.com/AshrithSagar/EEG-Imagined-speech-recognition/blob/main/utils/karaone.py#L885); |
| tfr_dataset_dir   | `path`                                        | Preferably use `files/TFR/KaraOne/tfr_ds-1/`                                                                                                                               |

## utils

| Parameter | Type   | Description                                            |
| --------- | ------ | ------------------------------------------------------ |
| path      | `path` | Absolute path to the project directory `utils/` folder |
