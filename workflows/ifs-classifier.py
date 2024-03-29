import os
import sys

from rich.console import Console

sys.path.append(os.getcwd())
from utils.classifier import ClassifierGridSearch, EvaluateClassifier, RegularClassifier
from utils.config import line_separator, load_config
from utils.ifs import InformationSet
from utils.feis import FEISDataLoader
from utils.karaone import KaraOneDataLoader

dataset_map = {"feis": FEISDataLoader, "karaone": KaraOneDataLoader}
classifier_map = {
    "RegularClassifier": RegularClassifier,
    "EvaluateClassifier": EvaluateClassifier,
    "ClassifierGridSearch": ClassifierGridSearch,
}


if __name__ == "__main__":
    args = load_config(config_file="config.yaml")

    dataset_name = args.get("dataset")
    if dataset_name in dataset_map:
        args = args[dataset_name]
    else:
        raise ValueError("Invalid dataset name")

    for model in args["models"]:
        console = Console(record=True)
        model_dir = os.path.join(args["model_base_dir"], model)
        Console().rule(title=f"[bold blue3][Model: {model}][/]", style="blue3")

        dataset = dataset_map[dataset_name]
        dset = dataset(
            raw_data_dir=args["raw_data_dir"],
            subjects=args["subjects"],
            verbose=True,
            console=console,
        )

        dset.load_features(epoch_type="thinking", features_dir=args["features_dir"])
        features, labels = dset.flatten()

        features_ifs = InformationSet(features, verbose=True, console=console)
        eff_features = features_ifs.extract_effective_information()
        dset.dataset_info(eff_features, labels)

        classifier_name = args.get("classifier")
        if classifier_name in classifier_map:
            classifier = classifier_map[classifier_name]
        else:
            raise ValueError(f"Invalid classifier name: {classifier_name}")

        for task in args["tasks"]:
            console.rule(title=f"[bold blue3][Task-{task}][/]", style="blue3")
            task_labels = dset.get_task(labels, task=task)
            save_dir = os.path.join(model_dir, classifier_name, f"task-{task}")

            mask = ~(task_labels == -1)
            clf = classifier(
                eff_features[mask],
                task_labels[mask],
                save_dir=save_dir,
                test_size=0.2,
                random_state=42,
                trial_size=args["trial_size"] or None,
                features_names=dset.features_names,
                verbose=True,
                console=console,
            )

            clf.get_model_config(model_file=os.path.join(model_dir, "model.py"))
            clf.run()

        with open(os.path.join(model_dir, classifier_name, "output.txt"), "w") as file:
            file.write(console.export_text())
