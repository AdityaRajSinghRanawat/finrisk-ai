from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "Age",
    "Income",
    "Credit_Score",
    "Loan_Amount",
    "Loan_Term",
]

CATEGORICAL_FEATURES = [
    "Employment_Status",
]

TARGET_COLUMN = "Loan_Approved"


def load_dataset(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    return pd.read_csv(file_path)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    expected_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_features = [col for col in expected_features if col not in df.columns]
    if missing_features:
        missing_str = ", ".join(missing_features)
        raise ValueError(f"Dataset is missing expected feature columns: {missing_str}")

    x = df[expected_features].copy()
    y = df[target_col].copy()
    return x, y
