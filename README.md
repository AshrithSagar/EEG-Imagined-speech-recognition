# EEG Imagined Speech Recognition

![GitHub](https://img.shields.io/github/license/AshrithSagar/EEG-Imagined-speech-recognition)
![GitHub repo size](https://img.shields.io/github/repo-size/AshrithSagar/EEG-Imagined-speech-recognition)
[![CodeFactor](https://www.codefactor.io/repository/github/AshrithSagar/EEG-Imagined-speech-recognition/badge)](https://www.codefactor.io/repository/github/AshrithSagar/EEG-Imagined-speech-recognition)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![GitBook](https://img.shields.io/badge/GitBook-EEG%20Imagined%20Speech%20Recognition-tan)](https://ashrithsagar.gitbook.io/eeg-imagined-speech-recognition)
[![GitBook](https://img.shields.io/badge/GitBook-EEG%20ISR-tan)](https://ashrithsagar.gitbook.io/eeg-isr)

Imagined speech recognition through EEG signals

## Installation

<details>

<summary>Clone the repository</summary>

```shell
git clone https://github.com/AshrithSagar/EEG-Imagined-speech-recognition.git
cd EEG-Imagined-speech-recognition
```

</details>

<details>

<summary>Install uv</summary>

Install [`uv`](https://docs.astral.sh/uv/), if not already.
Check [here](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

It is recommended to use `uv`, as it will automatically install the dependencies in a virtual environment.
If you don't want to use `uv`, skip to the next step.

TL;DR: Just run

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</details>

The dependencies are listed in the [pyproject.toml](pyproject.toml) file.

Install the package in editable mode (recommended):

```shell
# Using uv
uv sync

# Or with pip
pip install -e .
```

### Additional packages

For Ubuntu: `sudo apt-get install graphviz`

For macOS (with [Homebrew](https://brew.sh/)): `brew install graphviz`

For Windows: Download and install Graphviz from the [Graphviz website](https://graphviz.org/download/).

## Usage

### Configuration file `config.yaml`

The configuration file `config.yaml` contains the paths to the data files and the parameters for the different workflows.
Create and populate it with the appropriate values.
Refer to [config-template.yaml](config-template.yaml).

```yaml
---
_select:
  classifier: (str) Select one from { RegularClassifier, ClassifierGridSearch, EvaluateClassifier }
  dataset: (str) Select one from { KaraOne, FEIS }
classifier:
  features_select_k_best:
    k: (int | list[int])
    score_func: (str) Name of the score function to be used for ranking the features before selection. One from { pearsonr, f_classif }
  model_base_dir: (path) Preferably use files/Models/
  models: (list[str]) list of directory names containing the model.py within them. Eg:- [ model-1, model-2, ... ]
  n_splits: (int) Number of splits in cross-validation.
  random_state: (int) Seed value.
  test_size: (float) Size of test split.
  trial_size: (float / null) For testing purposes. Use null to use the entire dataset, else this is the fraction of the dataset that will be used.
feis:
  epoch_type: (str) One from { thinking, speaking, stimuli }
  features_dir: (path) Preferably use files/Features/FEIS/features-1/
  raw_data_dir: (path) Preferably use files/Data/FEIS/data_eeg/
  subjects: (all / list[int] / list[str]) Specify the subjects to be used. Use 'all' to use all subjects.
  tasks: list[int]) Available tasks:- [0]; Refer utils/feis.py:FEISDataLoader.get_task();
karaone:
  epoch_type: (str) One from { thinking, speaking, stimuli, clearing }
  features_dir: (path) Preferably use files/Features/KaraOne/features-1/
  filtered_data_dir: (path) Preferably use files/Data/KaraOne/EEG_data-1/
  length_factor: (float) Determines the window length.
  overlap: (float) Determines the overlap between consecutive windows.
  raw_data_dir: (path) Preferably use files/Data/KaraOne/EEG_raw/
  subjects: (all / list[int] / list[str]) Specify the subjects to be used. Use 'all' to use all subjects.
  tasks: (list[int]) Available tasks:- [0, 1, 2, 3, 4]; Refer utils/karaone.py:KaraOneDataLoader.get_task();
  tfr_dataset_dir: (path) Preferably use files/TFR/KaraOne/tfr_ds-1/
utils:
  path: (path) Absolute path to the project directory utils folder
```

### Classifier `model.py`

In {classifier.model_base_dir}, create the `model.py` with the following template.

```python
def model():
  # Model definition here
  # Takes certain parameters like random_state from config.yaml
  return ...

def param_grid():
  # Optional. Only useful in ClassifierGridSearch, ignored otherwise.
  return ...

def resample():
  # Optional. Remove/Comment this entire function to disable sampler.
  # Takes certain parameters like random_state from config.yaml
  return ...

def cross_validation():
  # Optional. Remove/Comment this entire function to use default CV of 5 splits from StratifiedKFold.
  # Takes certain parameters like random_state, n_splits from config.yaml
  return ...

def pipeline():
  # Optional. Remove/Comment this entire function to disable any pipeline functions to be run.
```

### Workflows

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

## References

- The KARA ONE Database: Phonological Categories in imagined and articulated speech
  <https://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html>

- <https://github.com/scottwellington/FEIS>

## Citation

If you use this project in your research, please cite using the following BibTeX entries.

```bibtex
@software{Yedlapalli_EEG-Imagined-Speech-recognition,
author = {Yedlapalli, Ashrith Sagar},
license = {MIT},
title = {{EEG-Imagined-Speech-recognition}},
url = {https://github.com/AshrithSagar/EEG-Imagined-speech-recognition}
}
```

## License

This project falls under the [MIT License](LICENSE).
