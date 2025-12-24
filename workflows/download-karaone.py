"""
download-karaone.py
Download the KaraOne database from it's website.
"""

from eeg_isr.config import Config
from eeg_isr.karaone import KaraOneDataLoader


def main(args):
    d_args = args["karaone"]

    karaone = KaraOneDataLoader(
        raw_data_dir=d_args["raw_data_dir"],
        subjects=d_args["subjects"],
        verbose=True,
    )

    base_url = "http://www.cs.toronto.edu/~complingweb/data/karaOne/"
    karaone.download(base_url=base_url)
    karaone.unzip(delete_zip=False)


if __name__ == "__main__":
    args = Config.from_args("Download KaraOne database from source")
    main(args)
