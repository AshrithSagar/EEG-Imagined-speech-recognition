# Model configuration

In {classifier.model_base_dir}, create the `model.py` with the following template.

{% code title="model.py" %}

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

{% endcode %}

## Directory structure

```plaintext
{classifier.model_base_dir}/
├── model.py
├── KaraOne
│   ├── EvaluateClassifier
│   │   ├── output.txt
│   │   ├── task-0
│   │   │   └── ...
│   │   └── ...
│   ├── ClassifierGridSearch
│   │   └── ...
│   └── RegularClassifier
│       └── ...
└── FEIS
    └── ...
```
