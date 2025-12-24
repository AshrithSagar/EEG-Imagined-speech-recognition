"""
ifs-classifier.py
Run a classifier on the effective features extracted from the dataset.
"""

import os
import sys
from typing import List

from rich.console import Console

sys.path.append(os.getcwd())
from eeg_isr.config import Config, ConsoleHandler, fetch_classifier, fetch_dataset
from eeg_isr.ifs import InformationSet


def main(args: Config):
    c_args: dict = args["classifier"]

    dataset_name: str = args.get("_select").get("dataset")
    dataset = fetch_dataset(dataset_name)
    d_args: dict = args[dataset_name.lower()]

    classifier_name: str = args.get("_select").get("classifier")
    classifier = fetch_classifier(classifier_name)

    models: List[str] = c_args["models"]
    for model in models:
        console = ConsoleHandler(record=True)
        model_dir = os.path.join(c_args["model_base_dir"], model, dataset_name)
        model_file = os.path.join(c_args["model_base_dir"], model, "model.py")
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

        tasks: List[int] = d_args["tasks"]
        for task in tasks:
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
                features_select_k_best=c_args["features_select_k_best"],
                features_names=dset.features_names,
                verbose=True,
                console=console,
            )

            clf.get_model_config(model_file=model_file)
            clf.run()

        file = os.path.join(model_dir, classifier_name, "output.txt")
        console.save(file)


if __name__ == "__main__":
    args = Config.from_args("Run the classifier on the effective features")
    main(args)
