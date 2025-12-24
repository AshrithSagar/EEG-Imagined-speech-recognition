"""
dataset.py
Dataset meta class
"""

from abc import ABC, abstractmethod

from rich.console import Console


class DatasetLoader(ABC):
    """Abstract Base Class (ABC) for Dataset Loaders"""

    @abstractmethod
    def __init__(
        self,
        raw_data_dir: str,
        subjects: list[str] | list[int] | str,
        all_subjects: list[str] | list[int],
        console: Console | None = None,
        verbose: bool = False,
    ) -> None:
        self.raw_data_dir = raw_data_dir
        self.all_subjects = all_subjects
        self.subjects = self.get_subjects(subjects)
        self.console = console if console else Console()
        self.verbose = verbose

    def get_subjects(self, subjects):
        """Retrieve a list of subjects based on input criteria.

        'subjects' parameter can be one of the following:
            - "all": Retrieve all available subjects.
            - A list of integers: Retrieve subjects by their indices.
            - A list of subject names: Retrieve subjects by their names.
        """

        if subjects == "all":
            return self.all_subjects
        elif isinstance(subjects, list):
            if all(isinstance(subject, int) for subject in subjects):
                return [self.all_subjects[index] for index in subjects]
            elif all(subject in self.all_subjects for subject in subjects):
                return subjects

        raise ValueError(
            """Invalid value for 'subjects'.
            Should be 'all', a list of subject indices, or a list of subject names."""
        )

    @abstractmethod
    def load_features(self, epoch_type, features_dir):
        pass

    @abstractmethod
    def get_task(self, labels, task):
        pass
