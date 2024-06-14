"""
download-karaone.py
Download the KaraOne database from it's website.
"""

import os
import sys

sys.path.append(os.getcwd())
from utils.config import load_config
from utils.karaone import KaraOneDataLoader


def main():
    d_args = load_config(config_file="config.yaml", key="karaone")

    karaone = KaraOneDataLoader(
        raw_data_dir=d_args["raw_data_dir"],
        subjects=d_args["subjects"],
        verbose=True,
    )

    base_url = "http://www.cs.toronto.edu/~complingweb/data/karaOne/"
    karaone.download(base_url=base_url)
    karaone.unzip(delete_zip=False)


if __name__ == "__main__":
    main()
