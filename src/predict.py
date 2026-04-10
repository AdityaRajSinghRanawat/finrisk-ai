from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

try:
    from src.utils import MODEL_PATH, categorize_risk, financial_health_score
except ModuleNotFoundError:
    from utils import MODEL_PATH, categorize_risk, financial_health_score


REQUIRED_FIELDS = [
    "Age",
    "Income",
    "Credit_Score",
    "Loan_Amount",
    "Loan_Term",
    "Employment_Status",
]


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Train the model first using src/train.py"
        )
    with path.open("rb") as f:
        return pickle.load(f)


def predict_risk(model, payload: dict) -> dict:
    missing = [k for k in REQUIRED_FIELDS if k not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    sample_df = pd.DataFrame([payload])
    classes = list(getattr(model, "classes_", []))
    approved_probability = float(model.predict_proba(sample_df)[0][classes.index(1)]) if 1 in classes else float(model.predict_proba(sample_df)[0][-1])
    default_probability = 1.0 - approved_probability
    risk_level = categorize_risk(default_probability)
    financial_score = financial_health_score(default_probability)

    return {
        "default_probability": round(default_probability, 4),
        "risk_level": risk_level,
        "financial_score": financial_score,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run risk prediction for a single user")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Model path")
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(Path(args.model))

    # Example payload for quick local testing.
    payload = {
        "Age": 32,
        "Income": 65000,
        "Credit_Score": 710,
        "Loan_Amount": 180000,
        "Loan_Term": 48,
        "Employment_Status": "Employed",
    }

    result = predict_risk(model, payload)
    print(result)


if __name__ == "__main__":
    main()
