"""
sampler.py
Collection of sampler instances.
"""

from imblearn.over_sampling import (
    ADASYN,
    RandomOverSampler,
    KMeansSMOTE,
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
    SMOTENC,
    SMOTEN,
)


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
        ),
        "ADASYN": ADASYN(
            sampling_strategy="auto",
            random_state=42,
            n_neighbors=5,
        ),
        "KMeansSMOTE": KMeansSMOTE(
            random_state=42,
            k_neighbors=5,
        ),
        "BorderlineSMOTE": BorderlineSMOTE(
            random_state=42,
            k_neighbors=5,
            m_neighbors=10,
            kind="borderline-1",
        ),
        "SVMSMOTE": SVMSMOTE(
            random_state=42,
            k_neighbors=5,
            m_neighbors=10,
        ),
        "SMOTENC": SMOTENC(
            categorical_features="auto",
            sampling_strategy="auto",
            random_state=42,
            k_neighbors=5,
        ),
        "SMOTEN": SMOTEN(
            sampling_strategy="auto",
            random_state=42,
            k_neighbors=5,
        ),
    }

    if sampler_name not in samplers:
        raise ValueError(f"Sampler '{sampler_name}' not found.")

    return samplers[sampler_name]
