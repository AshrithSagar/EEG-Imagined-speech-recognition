# Run workflows

Execute the various workflows using the command:

{% code fullWidth="false" %}
```bash
python3 workflows/{file}.py
```
{% endcode %}

from the root of the project directory.

### Workflows

<table data-full-width="true"><thead><tr><th>Workflow</th><th>Description</th><th data-hidden data-type="content-ref"></th></tr></thead><tbody><tr><td><code>download-karaone.py</code></td><td>Download the KaraOne dataset from the KaraOne website. Saves the dataset to the <code>{raw_data_dir}</code> folder.</td><td></td></tr><tr><td><code>features-karaone.py</code>, <code>features-feis.py</code></td><td>Preprocess EEG data to extract relevant features for different <code>epoch_types</code>. Saves processed data as <code>.fif</code> files to the <code>{filtered_data_dir}</code>.</td><td></td></tr><tr><td><code>ifs-classifier.py</code></td><td>Train a classifier using preprocessed EEG data, utilizing Information Set Theory.</td><td></td></tr><tr><td><code>flatten-classifier.py</code></td><td>Flatten feature matrix to a vector for classifier input. Specify number of features in <code>features_select_k_best[k]</code> (<code>int</code>).</td><td></td></tr><tr><td><code>flatten-classifier-KBest.py</code></td><td>Run classifier over multiple <code>k</code> values for feature selection. Iterate over values in <code>features_select_k_best[k]</code> (<code>list[int]</code>).</td><td></td></tr></tbody></table>
