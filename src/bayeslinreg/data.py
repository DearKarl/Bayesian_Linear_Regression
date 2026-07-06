"""Dataset loading and metadata for the Boston Housing benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "medv"

FEATURE_DESCRIPTIONS = {
    "crim": "Per-capita crime rate by town",
    "zn": "Residential land zoned for lots over 25,000 sq.ft.",
    "indus": "Non-retail business acres per town",
    "chas": "Charles River dummy variable",
    "nox": "Nitric oxides concentration",
    "rm": "Average rooms per dwelling",
    "age": "Owner-occupied units built before 1940",
    "dis": "Weighted distance to Boston employment centres",
    "rad": "Accessibility index for radial highways",
    "tax": "Property tax rate",
    "ptratio": "Pupil-teacher ratio by town",
    "b": "Legacy racial composition transform from the original dataset",
    "lstat": "Percentage of lower-status population",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_boston_csv(path: str | Path | None = None) -> pd.DataFrame:
    """Load the bundled Boston Housing CSV with numeric coercion and imputation."""

    csv_path = Path(path) if path is not None else project_root() / "BostonHousing_data.csv"
    data = pd.read_csv(csv_path)
    data.columns = [column.strip().lower() for column in data.columns]
    data = data.apply(pd.to_numeric, errors="coerce")

    if TARGET not in data.columns:
        raise ValueError(f"Expected target column {TARGET!r} in {csv_path}")

    if data.isna().any().any():
        data = data.fillna(data.mean(numeric_only=True))

    return data


def make_feature_target(
    data: pd.DataFrame,
    *,
    drop_legacy_race_feature: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split features and target.

    The Boston Housing dataset is kept for legacy comparability. Passing
    ``drop_legacy_race_feature=True`` supports a sensitivity analysis that
    excludes the controversial original ``b`` variable.
    """

    features = data.drop(columns=[TARGET]).copy()
    if drop_legacy_race_feature and "b" in features.columns:
        features = features.drop(columns=["b"])
    target = data[TARGET].copy()
    return features, target
