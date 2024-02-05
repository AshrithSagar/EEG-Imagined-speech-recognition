import os
import sys

from rich.console import Console

sys.path.append(os.getcwd())
from utils.config import line_separator, load_config
from utils.ifs import InformationSet
from utils.karaone import KaraOneDataLoader
from utils.classifier import Classifier


if __name__ == "__main__":
    console = Console(record=True)
    args = load_config(config_file="config.yaml", key="karaone")

    karaone = KaraOneDataLoader(
        raw_data_dir=args["raw_data_dir"],
        subjects="all",
        verbose=True,
        console=console,
    )

    karaone.load_features(epoch_type="thinking", features_dir=args["features_dir"])
    features, labels = karaone.flatten()
    labels = karaone.get_task(labels, task=args["task"])

    features_ifs = InformationSet(features, verbose=True)
    eff_features = features_ifs.extract_effective_information()
    karaone.dataset_info(eff_features, labels)

    clf = Classifier(
        eff_features,
        labels,
        save_dir=args["model_dir"],
        test_size=0.2,
        random_state=42,
        trial_size=args["trial_size"] or None,
        verbose=True,
        console=console,
    )

    clf.compile()
    if args["grid_search"]:
        clf.perform_grid_search(verbose=10)
    else:
        clf.train()
    clf.predict()
    clf.evaluate(show_plots=False)
    clf.save()

    with open(os.path.join(args["model_dir"], "output.txt"), "w") as file:
        file.write(console.export_text())
