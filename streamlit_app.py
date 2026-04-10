from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.predict import load_model, predict_risk
    from src.utils import MODEL_PATH, risk_suggestions
except Exception:
    from predict import load_model, predict_risk
    from utils import MODEL_PATH, risk_suggestions


st.set_page_config(
    page_title="finrisk-ai",
    page_icon="💳",
    layout="centered",
)

st.title("finrisk-ai")
st.subheader("Smart Loan and Financial Risk Predictor")
st.caption("Uses your trained model at models/loan_model.pkl")

model_path = MODEL_PATH
model_exists = Path(model_path).exists()

if not model_exists:
    st.error("Model file not found. Train first using: python src/train.py --data data/raw/loan_prediction_dataset.csv --target Loan_Approved")
    st.stop()

with st.form("risk_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=32, step=1)
        income = st.number_input("Income", min_value=1.0, value=65000.0, step=1000.0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=710, step=1)

    with col2:
        loan_amount = st.number_input("Loan Amount", min_value=1.0, value=180000.0, step=1000.0)
        loan_term = st.number_input("Loan Term (months)", min_value=1.0, value=48.0, step=1.0)
        employment_status = st.selectbox(
            "Employment Status",
            options=["Employed", "Self-Employed", "Unemployed", "Student", "Retired"],
            index=0,
        )

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    model = load_model(model_path)

    payload = {
        "Age": int(age),
        "Income": float(income),
        "Credit_Score": int(credit_score),
        "Loan_Amount": float(loan_amount),
        "Loan_Term": float(loan_term),
        "Employment_Status": employment_status,
    }

    result = predict_risk(model, payload)
    suggestions = risk_suggestions(result["risk_level"])

    st.markdown("### Prediction Result")
    st.metric("Default Probability", f"{result['default_probability'] * 100:.2f}%")
    st.metric("Risk Level", result["risk_level"])
    st.metric("Financial Score", f"{result['financial_score']:.2f}")

    st.markdown("### Suggestions")
    for item in suggestions:
        st.write(f"- {item}")
