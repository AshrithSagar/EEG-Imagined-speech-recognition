# EEG Imagined Speech Recognition

![GitHub](https://img.shields.io/github/license/AshrithSagar/EEG-Imagined-speech-recognition)
![GitHub repo size](https://img.shields.io/github/repo-size/AshrithSagar/EEG-Imagined-speech-recognition)
[![CodeFactor](https://www.codefactor.io/repository/github/AshrithSagar/EEG-Imagined-speech-recognition/badge)](https://www.codefactor.io/repository/github/AshrithSagar/EEG-Imagined-speech-recognition)

Imagined speech recognition through EEG signals

## Installation

Follow these steps to get started.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AshrithSagar/EEG-Imagined-speech-recognition.git
   cd EEG-Imagined-speech-recognition
   ```

2. **Install dependencies:**

   Next, install the required dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

### Linux packages

For Ubuntu: `sudo apt-get install graphviz`

For macOS (with Homebrew): `brew install graphviz`

For Windows: Download and install Graphviz from the [Graphviz website](https://graphviz.org/download/).

## Usage

### Configuration file

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

### Workflows

Run the different workflows using `python3 workflows/*.py` from the project directory.

1. `features-karaone.py`, `features-feis.py`:
Preprocess the EEG data to extract relevant features.

1. `ifs-classifier.py`:
Train a machine learning classifier using the preprocessed EEG data.
Uses Information set theory to extract effective information from the feature matrix, to be used as features.

1. `flatten-classifier.py`:
Flattens the feature matrix to a vector, to be used as features.
Specify the number of features to be selected in features_select_k_best[k] (int).

1. `flatten-classifier-KBest.py`:
Run over multiple k's from features_select_k_best[k] (list[int]).

## References

- <https://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html>
- <https://github.com/scottwellington/FEIS>

## License

This project falls under the [MIT License](LICENSE).
