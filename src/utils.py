from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "loan_model.pkl"


def categorize_risk(default_probability: float) -> str:
    if default_probability < 0.25:
        return "Low"
    if default_probability < 0.55:
        return "Medium"
    return "High"


def financial_health_score(default_probability: float) -> float:
    # Higher default probability lowers health score on a 0-100 scale.
    score = max(0.0, min(100.0, 100.0 * (1.0 - default_probability)))
    return round(score, 2)


def risk_suggestions(risk_level: str) -> list[str]:
    if risk_level == "Low":
        return [
            "Maintain current repayment discipline.",
            "Keep debt levels stable and preserve cash flow.",
        ]
    if risk_level == "Medium":
        return [
            "Reduce outstanding debt before applying for a larger loan.",
            "Improve income stability and keep EMIs conservative.",
        ]
    return [
        "Lower the loan amount or wait until your financial position improves.",
        "Strengthen repayment capacity before taking on more debt.",
    ]
