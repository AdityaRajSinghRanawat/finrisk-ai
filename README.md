# finrisk-ai

Portfolio-grade loan approval risk prediction project for data science recruiter review.

## Problem Statement

Financial institutions need a fast and interpretable way to estimate whether a loan application is likely to default. This project builds an end-to-end ML workflow that predicts risk and presents outputs in a simple Streamlit app.

## What This Project Demonstrates

- Structured EDA with data quality checks, class balance analysis, and feature exploration
- Reproducible preprocessing and model training pipeline
- Model benchmarking across Logistic Regression, Random Forest, and Decision Tree
- Practical inference outputs for decision support:
	- default probability
	- risk level (Low/Medium/High)
	- financial health score
	- actionable suggestions

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

## Dataset and Target

- Dataset: `data/raw/loan_prediction_dataset.csv`
- Target column: `Loan_Approved`
- Features used:
	- `Age`
	- `Income`
	- `Credit_Score`
	- `Loan_Amount`
	- `Loan_Term`
	- `Employment_Status`

## Quick Start

From project root:

```bash
python -m pip install -r requirements.txt
python src/train.py --data data/raw/loan_prediction_dataset.csv --target Loan_Approved
python -m streamlit run streamlit_app.py
```

Open: `http://localhost:8501`

Note: training is required once to generate `models/loan_model.pkl`.

## Common Errors and Fixes

1. Error: `streamlit is not recognized`

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

2. Error: `Model file not found`

```bash
python src/train.py --data data/raw/loan_prediction_dataset.csv --target Loan_Approved
python -m streamlit run streamlit_app.py
```

3. App opened on different port

If `8501` is busy, Streamlit may use `8502` (check terminal URL).

## Project Structure

- `streamlit_app.py`: Streamlit UI
- `src/preprocess.py`: preprocessing and feature pipeline
- `src/train.py`: training and model selection
- `src/predict.py`: inference logic
- `src/utils.py`: scoring and helper utilities
- `notebooks/eda.ipynb`: EDA and benchmarking notebook
- `requirements.txt`: dependencies

## Recruiter Notes

- This project emphasizes end-to-end DS execution, not only model accuracy.
- The notebook includes a caveat on possible synthetic/rule-driven data behavior and over-optimistic metrics.
- Code is modular, readable, and ready for extension into production APIs.

## Next Improvements

- Add model explainability (SHAP)
- Add calibration and threshold tuning
- Add automated tests and CI pipeline
- Deploy Streamlit app to Streamlit Community Cloud
