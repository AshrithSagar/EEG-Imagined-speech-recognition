"""
download-karaone.py
Download the KaraOne database from it's website.
"""

import os
import sys

sys.path.append(os.getcwd())
from utils.config import Config, setup_parser
from utils.karaone import KaraOneDataLoader


def main(config_file="config.yaml"):
    d_args = Config(config_file)["karaone"]

    karaone = KaraOneDataLoader(
        raw_data_dir=d_args["raw_data_dir"],
        subjects=d_args["subjects"],
        verbose=True,
    )

    base_url = "http://www.cs.toronto.edu/~complingweb/data/karaOne/"
    karaone.download(base_url=base_url)
    karaone.unzip(delete_zip=False)


if __name__ == "__main__":
    args = setup_parser("Download KaraOne database from source")
    main(args.config)
