from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from src.preprocess import TARGET_COLUMN, build_preprocessor, load_dataset, split_features_target
    from src.utils import MODEL_PATH
except ModuleNotFoundError:
    from preprocess import TARGET_COLUMN, build_preprocessor, load_dataset, split_features_target
    from utils import MODEL_PATH


def train_models(df: pd.DataFrame, target_col: str, random_state: int = 42):
    x, y = split_features_target(df, target_col=target_col)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()

    model_candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced",
        ),
    }

    best_name = ""
    best_pipeline = None
    best_f1 = -1.0

    print("\\nModel Evaluation Summary")
    print("=" * 30)

    for name, model in model_candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(f"\\n{name}")
        print(f"accuracy: {acc:.4f}")
        print(f"f1-score: {f1:.4f}")
        print("confusion matrix:")
        print(confusion_matrix(y_test, preds))
        print("classification report:")
        print(classification_report(y_test, preds))

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model was successfully trained.")

    print(f"\\nBest model selected: {best_name} (f1={best_f1:.4f})")
    return best_name, best_pipeline


def save_model(pipeline: Pipeline, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved model to: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train loan default risk models")
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/loan_prediction_dataset.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=TARGET_COLUMN,
        help="Target column in dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MODEL_PATH),
        help="Output path for trained model",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset(args.data)
    _, best_pipeline = train_models(dataset, target_col=args.target)
    save_model(best_pipeline, Path(args.output))


if __name__ == "__main__":
    main()
