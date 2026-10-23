# Loan Approval Prediction Using Machine Learning

**Topic:** Comparative Analysis of Machine Learning Models for Loan Approval Prediction Using Categorical Feature Encoding and Model Evaluation

## Project Overview

This educational machine learning project predicts whether a loan application is labelled **Approved** or **Rejected** in the selected dataset. It demonstrates exploratory data analysis, data cleaning, categorical encoding, feature scaling, stratified train/validation/test splitting, model comparison, evaluation, feature importance, a learning curve, and an optional Gradio demo.

> **Disclaimer:** This is an academic/portfolio project. Predictions are based on one dataset and must not be used to make real lending decisions. The reported scores do not guarantee similar performance on real bank applications or other datasets.

## Dataset

- **Source:** [Loan Approval Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- **Original size:** 4,269 rows and 13 columns.
- **Target:** `loan_status` (`Approved` / `Rejected`).
- **Identifier excluded from modelling:** `loan_id`.
- **Rows after cleaning:** 4,241.

The project uses these 11 input features:

| Feature | Description |
|---|---|
| `no_of_dependents` | Number of dependents |
| `education` | Graduate / Not Graduate |
| `self_employed` | Self-employment status |
| `income_annum` | Annual income |
| `loan_amount` | Requested loan amount |
| `loan_term` | Loan term |
| `cibil_score` | Credit score |
| `residential_assets_value` | Residential asset value |
| `commercial_assets_value` | Commercial asset value |
| `luxury_assets_value` | Luxury asset value |
| `bank_asset_value` | Bank asset value |

## Methodology

1. **Load and inspect data:** review the data shape, column types, missing values, duplicates, and target distribution.
2. **Clean data:** 28 rows with negative `residential_assets_value` were excluded. This left 4,241 rows. No missing values or duplicate rows were found in the inspected dataset.
3. **Separate features and target:** drop `loan_id` from the inputs and use `loan_status` as the target.
4. **Split the dataset:** stratified 70% training, 15% validation, and 15% test split.
5. **Preprocess:** median imputation and `StandardScaler` for numeric features; most-frequent imputation and `OneHotEncoder` for categorical features. The preprocessing pipeline is fitted on training data only, then applied to validation and test data.
6. **Train and compare models:** Logistic Regression, Decision Tree, and Random Forest.
7. **Evaluate:** compare validation Accuracy, Precision, Recall, F1-score, and ROC-AUC. The Random Forest was selected based on the validation results and then evaluated on the held-out test set.
8. **Interpret:** plot the ROC curve, confusion matrix, feature importance, and learning curve.

### Split sizes

| Subset | Rows | Share |
|---|---:|---:|
| Training | 2,968 | 70% (approximately) |
| Validation | 636 | 15% (approximately) |
| Test | 637 | 15% (approximately) |
| **Total after cleaning** | **4,241** | **100%** |

## Model Results

### Validation-set comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9230 | 0.9437 | 0.9318 | 0.9377 | 0.9660 |
| Decision Tree | 0.9780 | 0.9848 | 0.9798 | 0.9823 | 0.9774 |
| Random Forest | **0.9811** | **0.9848** | **0.9848** | **0.9848** | **0.9974** |

### Random Forest — final held-out test set

| Metric | Test score |
|---|---:|
| Accuracy | 0.9890 (98.90%) |
| Precision | 0.9875 (98.75%) |
| Recall | 0.9949 (99.49%) |
| F1-score | 0.9912 (99.12%) |
| ROC-AUC | 0.9981 |

Only Random Forest's test result is reported here because it was the selected model. The validation metrics for the other models are not test metrics.

## Feature Importance

Random Forest impurity-based feature importance from this run:

| Feature | Importance |
|---|---:|
| `cibil_score` | 0.8181 |
| `loan_term` | 0.0568 |
| `loan_amount` | 0.0293 |
| `luxury_assets_value` | 0.0181 |
| `income_annum` | 0.0172 |
| `commercial_assets_value` | 0.0166 |
| `bank_asset_value` | 0.0154 |
| `residential_assets_value` | 0.0153 |
| `no_of_dependents` | 0.0083 |
| `self_employed` | 0.0026 |
| `education` | 0.0021 |

`cibil_score` dominates this model's importance values. Feature importance describes the model's use of a feature; it does not establish causation or universal lending policy.

## How to Run

1. Open `Loan_Approval_Prediction.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run the notebook cells from top to bottom.
3. When the first cell asks for a file, upload the Kaggle ZIP archive that contains `loan_approval_dataset.csv`, or upload `loan_approval_dataset.csv` directly.
4. The notebook performs data inspection, cleaning, split, preprocessing, training, evaluation, and plots.
5. The results cell creates `loan_approval_final_results.csv` for download.
6. The last cells install Gradio and launch a demo. A share link may be public and temporary; do not enter real personal or financial information.

The notebook was prepared for Google Colab. Runtime packages may be installed within the notebook where needed.

## Limitations

- Results are based on a single dataset and one stratified train/validation/test split.
- Cross-validation learning-curve values and held-out test scores measure different things and should not be conflated.
- The strong test score and high CIBIL-score importance should be interpreted cautiously and independently validated before any real-world use.
- This project is not a bank decision system and does not assess fairness, legal compliance, or lending-policy suitability.

## Technologies

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Joblib (optional model persistence), and Gradio for the interactive demo.
