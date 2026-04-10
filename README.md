# finrisk-ai

End-to-end machine learning project for loan risk estimation, including training, inference, and a Streamlit interface.

## Overview

This project trains a classifier on loan application data and converts model output into business-facing risk indicators:

- Default probability (derived from model approval probability)
- Risk level (`Low`, `Medium`, `High`)
- Financial health score (0-100)
- Practical risk suggestions

It is designed as a portfolio-ready DS workflow with modular code in `src/`, EDA in `notebooks/`, and a runnable UI in `streamlit_app.py`.

## Features

- Data loading and schema validation
- Reusable preprocessing pipeline:
	- Numeric median imputation + scaling
	- Categorical mode imputation + one-hot encoding
- Multi-model benchmarking (`LogisticRegression`, `RandomForestClassifier`, `DecisionTreeClassifier`)
- Automatic best-model selection by test F1 score
- Model serialization to `models/loan_model.pkl`
- Streamlit app for interactive single-applicant scoring

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

## Dataset and Schema

- Input file: `data/raw/loan_prediction_dataset.csv`
- Default target column: `Loan_Approved`
- Required feature columns:
	- `Age`
	- `Income`
	- `Credit_Score`
	- `Loan_Amount`
	- `Loan_Term`
	- `Employment_Status`

## Setup

From the project root:

```bash
python -m pip install -r requirements.txt
```

## Train a Model

```bash
python src/train.py --data data/raw/loan_prediction_dataset.csv --target Loan_Approved
```

This command evaluates candidate models and saves the best pipeline to:

- `models/loan_model.pkl`

Optional arguments:

- `--output` to change model output path

## Run Inference (CLI)

```bash
python src/predict.py --model models/loan_model.pkl
```

`src/predict.py` includes an example payload in `main()` and prints a result dictionary.

## Launch the Streamlit App

```bash
python -m streamlit run streamlit_app.py
```

Then open the URL shown in terminal (typically `http://localhost:8501`).

## Common Issues

1. `streamlit is not recognized`

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

2. `Model file not found`

```bash
python src/train.py --data data/raw/loan_prediction_dataset.csv --target Loan_Approved
python -m streamlit run streamlit_app.py
```

3. Port 8501 already in use

Streamlit will automatically use another port (for example, 8502). Use the URL printed in terminal.

## Project Structure

- `streamlit_app.py`: Streamlit interface
- `src/preprocess.py`: data loading, validation, and preprocessing pipeline
- `src/train.py`: training, evaluation, best-model selection, persistence
- `src/predict.py`: model loading and single-record risk inference
- `src/utils.py`: risk categorization, scoring, and suggestion helpers
- `notebooks/eda.ipynb`: exploratory data analysis and model exploration
- `requirements.txt`: dependency list

## Notes

- Risk labeling uses thresholds from `src/utils.py`:
	- `Low`: default probability < 0.25
	- `Medium`: 0.25 to < 0.55
	- `High`: >= 0.55
- Financial score is computed as `100 * (1 - default_probability)`, clipped to [0, 100].

## Possible Next Steps

- Add explainability (SHAP or feature importance views in app)
- Add probability calibration and threshold tuning
- Add test suite for preprocessing and inference contracts
- Add CI checks and deployment workflow
