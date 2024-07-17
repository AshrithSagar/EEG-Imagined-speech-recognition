# Configuration file

The configuration file `config.yaml` contains the paths to the data files and the parameters for the different workflows.

Refer to [config-template.yaml](config-template.yaml) for the template.

## _select

- **classifier**: `(str)`
  Select one from { RegularClassifier, ClassifierGridSearch, EvaluateClassifier }

- **dataset**: `(str)`
  Select one from { KaraOne, FEIS }

## classifier

- **features_select_k_best**:
  - **k**: `(int | list[int])`
    Number of top features to select
  - **score_func**: `(str)`
    Name of the score function to be used for ranking the features before selection.
    One from { pearsonr, f_classif }

- **model_base_dir**: `(path)`
  Preferably use `files/Models/`

- **models**: `(list[str])`
  List of directory names containing the `model.py` within them.
  Example: `[ model-1, model-2, ... ]`

- **n_splits**: `(int)`
  Number of splits in cross-validation.

- **random_state**: `(int)`
  Seed value.

- **test_size**: `(float)`
  Size of test split.

- **trial_size**: `(float / null)`
  For testing purposes. Use `null` to use the entire dataset, else this is the fraction of the dataset that will be used.

## feis

- **epoch_type**: `(str)`
  One from { thinking, speaking, stimuli }

- **features_dir**: `(path)`
  Preferably use `files/Features/FEIS/features-1/`

- **raw_data_dir**: `(path)`
  Preferably use `files/Data/FEIS/data_eeg/`

- **subjects**: `(all / list[int] / list[str])`
  Specify the subjects to be used. Use `'all'` to use all subjects.

- **tasks**: `(list[int])`
  Available tasks: `[0]`. Refer `utils/feis.py:FEISDataLoader.get_task();`

## karaone

- **epoch_type**: `(str)`
  One from { thinking, speaking, stimuli, clearing }

- **features_dir**: `(path)`
  Preferably use `files/Features/KaraOne/features-1/`

- **filtered_data_dir**: `(path)`
  Preferably use `files/Data/KaraOne/EEG_data-1/`

- **length_factor**: `(float)`
  Determines the window length.

- **overlap**: `(float)`
  Determines the overlap between consecutive windows.

- **raw_data_dir**: `(path)`
  Preferably use `files/Data/KaraOne/EEG_raw/`

- **subjects**: `(all / list[int] / list[str])`
  Specify the subjects to be used. Use `'all'` to use all subjects.

- **tasks**: `(list[int])`
  Available tasks: `[0, 1, 2, 3, 4]`. Refer `utils/karaone.py:KaraOneDataLoader.get_task();`

- **tfr_dataset_dir**: `(path)`
  Preferably use `files/TFR/KaraOne/tfr_ds-1/`

## utils

- **path**: `(path)`
  Absolute path to the project directory `utils` folder
