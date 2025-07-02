# Installation

Follow these steps to get started.

1.  **Clone the repository:**

    ```shell
    git clone https://github.com/AshrithSagar/EEG-Imagined-speech-recognition.git
    cd EEG-Imagined-speech-recognition
    ```

2.  **Install uv (recommended):**

    Install [`uv`](https://docs.astral.sh/uv/), if not already.
    Check [here](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

    It is recommended to use `uv`, as it will automatically install the dependencies in a virtual environment.
    If you don't want to use `uv`, skip to the next step.

    TL;DR: Just run

    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  **Install dependencies:**

    The dependencies are listed in the [pyproject.toml](pyproject.toml) file.

    Install the package in editable mode (recommended):

    ```shell
    # Using uv
    uv pip install -e .

    # Or with pip
    pip install -e .
    ```

### Additional packages

For Ubuntu: `sudo apt-get install graphviz`

For macOS (with [Homebrew](https://brew.sh/)): `brew install graphviz`

For Windows: Download and install Graphviz from the [Graphviz website](https://graphviz.org/download/).
