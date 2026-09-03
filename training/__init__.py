"""Argus Monitor model training pipeline.

Scripts in this package are meant to be run standalone from the repo root,
e.g. ``python training/prepare_dataset.py --help``. The package also exposes
importable, unit-testable pure functions (see ``training.prepare_dataset``)
so the dataset-splitting logic can be exercised without touching the real
(large, gitignored) dataset on disk.
"""
