"""
flatten-classifier-KBest.py
Run a classifier on the flattened features extracted from the dataset.
Run on different k_best's.
"""

import os
import sys

from rich.console import Console

sys.path.append(os.getcwd())
from eeg_isr.config import Config, ConsoleHandler, fetch_classifier, fetch_dataset
from eeg_isr.info import KBestSummary


def main(args):
    c_args = args["classifier"]

    dataset_name = args.get("_select").get("dataset")
    dataset = fetch_dataset(dataset_name)
    d_args = args[dataset_name.lower()]

    classifier_name = args.get("_select").get("classifier")
    classifier = fetch_classifier(classifier_name)

    for model in c_args["models"]:
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
        features = features.reshape(features.shape[0], -1)
        dset.dataset_info(features, labels)

        for task in d_args["tasks"]:
            console.rule(title=f"[bold blue3][Task-{task}][/]", style="blue3")
            task_labels = dset.get_task(labels, task=task)
            mask = ~(task_labels == -1)
            task_dir = os.path.join(model_dir, classifier_name, f"task-{task}")

            for k_best in c_args["features_select_k_best"]["k"]:
                console.rule(title=f"[bold blue3][KBest-{k_best}][/]", style="blue3")
                features_select_k_best = c_args["features_select_k_best"]
                features_select_k_best["k"] = k_best
                save_dir = os.path.join(task_dir, f"k_best-{k_best}")

                clf = classifier(
                    features[mask],
                    task_labels[mask],
                    save_dir=save_dir,
                    test_size=c_args["test_size"],
                    random_state=c_args["random_state"],
                    trial_size=c_args["trial_size"] or None,
                    features_select_k_best=features_select_k_best,
                    verbose=True,
                    console=console,
                )

                clf.get_model_config(model_file=model_file)
                clf.run()

            summary = KBestSummary(task_dir=task_dir)
            summary.plot(metrics="all")

        file = os.path.join(model_dir, classifier_name, "output.txt")
        console.save(file)


if __name__ == "__main__":
    args = Config.from_args(
        "Run the classifier on flattened features with different k_best's."
    )
    main(args)
