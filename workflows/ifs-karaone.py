import os
import sys
from sklearn.svm import SVC

sys.path.append(os.getcwd())
from utils.config import line_separator, load_config
from utils.ifs import InformationSet
from utils.karaone import KaraOneDataLoader
from utils.svm import SVMClassifier


if __name__ == "__main__":
    args = load_config(key="karaone")

    karaone = KaraOneDataLoader(
        raw_data_dir=args["raw_data_dir"],
        subjects="all",
        verbose=True,
    )

    karaone.load_features(epoch_type="thinking", features_dir=args["features_dir"])
    features, labels = karaone.flatten()

    features_ifs = InformationSet(features, verbose=True)
    eff_features = features_ifs.extract_effective_information()
    karaone.dataset_info(eff_features, labels)

    # Support Vector Machines (SVMs)
    svm = SVMClassifier(
        eff_features,
        labels,
        test_size=0.2,
        random_state=42,
        trial_size=30,
        verbose=True,
    )

    model = SVC(kernel="linear", degree=3, cache_size=1000, verbose=False)
    svm.compile(model)

    svm.train()

    svm.predict()

    svm.evaluate(show_plots=False)
    svm.save(save_dir=args["svm_model_dir"])
