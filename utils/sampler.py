"""
sampler.py
Sampler classes
"""

from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler


def fetch_sampler(sampler_name):
    """
    Fetches a sampler instance based on the given sampler name.

    Parameters:
    - sampler_name (str): Name of the sampler to fetch.

    Returns:
    - sampler_instance: Instance of the specified sampler.
    """

    samplers = {
        "RandomOverSampler": RandomOverSampler(
            sampling_strategy="auto",
            random_state=42,
            shrinkage=None,
        ),
        "SMOTE": SMOTE(
            random_state=42,
            k_neighbors=5,
            n_jobs=1,
        ),
        "ADASYN": ADASYN(
            sampling_strategy="auto",
            random_state=42,
            n_neighbors=5,
            n_jobs=1,
        ),
    }

    if sampler_name not in samplers:
        raise ValueError(f"Sampler '{sampler_name}' not found.")

    return samplers[sampler_name]
