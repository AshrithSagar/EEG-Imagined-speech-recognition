"""
ifs-classifier.py
Run a classifier on the effective features extracted from the dataset.
"""

import os
import sys

from rich.console import Console

sys.path.append(os.getcwd())
from utils.config import fetch_select, load_config
from utils.ifs import InformationSet


if __name__ == "__main__":
    args = load_config(config_file="config.yaml")
    c_args = load_config(config_file="config.yaml", key="classifier")

    dataset_name = args.get("_select").get("dataset")
    dataset = fetch_select("dataset", dataset_name)
    d_args = args[dataset_name.lower()]

    classifier_name = args.get("_select").get("classifier")
    classifier = fetch_select("classifier", classifier_name)

    for model in c_args["models"]:
        console = Console(record=True)
        model_dir = os.path.join(c_args["model_base_dir"], model, dataset_name)
        Console().rule(title=f"[bold blue3][Model: {model}][/]", style="blue3")

        dset = dataset(
            raw_data_dir=d_args["raw_data_dir"],
            subjects=d_args["subjects"],
            verbose=True,
            console=console,
        )

        dset.load_features(
            epoch_type=d_args["epoch_type"], features_dir=d_args["features_dir"]
        )
        features, labels = dset.flatten()

        features_ifs = InformationSet(features, verbose=True, console=console)
        eff_features = features_ifs.extract_effective_information()
        dset.dataset_info(eff_features, labels)

        for task in d_args["tasks"]:
            console.rule(title=f"[bold blue3][Task-{task}][/]", style="blue3")
            task_labels = dset.get_task(labels, task=task)
            save_dir = os.path.join(model_dir, classifier_name, f"task-{task}")

            mask = ~(task_labels == -1)
            clf = classifier(
                eff_features[mask],
                task_labels[mask],
                save_dir=save_dir,
                test_size=c_args["test_size"],
                random_state=c_args["random_state"],
                trial_size=c_args["trial_size"] or None,
                features_names=dset.features_names,
                verbose=True,
                console=console,
            )

            clf.get_model_config(
                model_file=os.path.join(c_args["model_base_dir"], model, "model.py")
            )
            clf.run()

        with open(os.path.join(model_dir, classifier_name, "output.txt"), "w") as file:
            file.write(console.export_text())
