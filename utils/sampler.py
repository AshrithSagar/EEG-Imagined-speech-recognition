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
            shrinkage=None,
        ),
        "SMOTE": SMOTE(
            k_neighbors=5,
        ),
        "ADASYN": ADASYN(
            sampling_strategy="auto",
            n_neighbors=5,
        ),
        "KMeansSMOTE": KMeansSMOTE(
            k_neighbors=5,
        ),
        "BorderlineSMOTE": BorderlineSMOTE(
            k_neighbors=5,
            m_neighbors=10,
            kind="borderline-1",
        ),
        "SVMSMOTE": SVMSMOTE(
            k_neighbors=5,
            m_neighbors=10,
        ),
        "SMOTENC": SMOTENC(
            categorical_features="auto",
            sampling_strategy="auto",
            k_neighbors=5,
        ),
        "SMOTEN": SMOTEN(
            sampling_strategy="auto",
            k_neighbors=5,
        ),
    }

    if sampler_name not in samplers:
        raise ValueError(f"Sampler '{sampler_name}' not found.")

    return samplers[sampler_name]
