import os
import sys

from rich.console import Console

sys.path.append(os.getcwd())
from utils.config import line_separator, load_config
from utils.ifs import InformationSet
from utils.karaone import KaraOneDataLoader
from utils.classifier import Classifier


if __name__ == "__main__":
    args = load_config(config_file="config.yaml", key="karaone")

    for model in args["models"]:
        console = Console(record=True)
        model_dir = os.path.join(args["model_base_dir"], model)
        Console().rule(title=f"[bold blue3][Model: {model}][/]", style="blue3")

        karaone = KaraOneDataLoader(
            raw_data_dir=args["raw_data_dir"],
            subjects="all",
            verbose=True,
            console=console,
        )

        karaone.load_features(epoch_type="thinking", features_dir=args["features_dir"])
        features, labels = karaone.flatten()

        features_ifs = InformationSet(features, verbose=True, console=console)
        eff_features = features_ifs.extract_effective_information()
        karaone.dataset_info(eff_features, labels)

        for task in args["tasks"]:
            console.rule(title=f"[bold blue3][Task-{task}][/]", style="blue3")
            task_labels = karaone.get_task(labels, task=task)
            save_dir = os.path.join(model_dir, f"task-{task}")

            clf = Classifier(
                eff_features,
                task_labels,
                save_dir=save_dir,
                test_size=0.2,
                random_state=42,
                trial_size=args["trial_size"] or None,
                run_grid_search=args["grid_search"],
                verbose=True,
                console=console,
            )

            clf.get_model_config(model_file=os.path.join(model_dir, "model.py"))
            clf.compile(load=args["evaluate_only"])
            if not args["evaluate_only"]:
                clf.train()
            clf.predict()
            clf.evaluate(show_plots=False)
            clf.save()

        with open(os.path.join(model_dir, "output.txt"), "w") as file:
            file.write(console.export_text())
