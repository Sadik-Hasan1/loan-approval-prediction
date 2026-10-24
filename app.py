"""Gradio web app for the Loan Approval Prediction project.

Place this file next to loan_approval_model.joblib and deploy as a Render
Free Web Service. This is an educational demo, not a real lending decision tool.
"""

from pathlib import Path
import os

import gradio as gr
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_approval_model.joblib"

# Match the raw input columns used to train the notebook's preprocessor.
FEATURE_COLUMNS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


def _first_present(mapping, candidates):
    """Return the first non-None value found under any candidate key."""
    for key in candidates:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Required file loan_approval_model.joblib was not found. "
        "Upload it to the same repository folder as app.py."
    )

bundle = joblib.load(MODEL_PATH)

# The saved artifact is expected to contain the estimator and preprocessor.
# A few alternate key names are accepted to make the app resilient to the
# bundle's naming convention. A complete sklearn Pipeline is also supported.
PIPELINE = None
MODEL = None
PREPROCESSOR = None
MODEL_NAME = "Random Forest"

if isinstance(bundle, dict):
    PIPELINE = _first_present(bundle, ["pipeline", "model_pipeline", "full_pipeline"])
    MODEL = _first_present(
        bundle,
        ["model", "best_model", "selected_model", "classifier", "estimator", "trained_model"],
    )
    PREPROCESSOR = _first_present(
        bundle,
        ["preprocessor", "preprocessing", "transformer", "feature_preprocessor"],
    )
    MODEL_NAME = str(bundle.get("model_name", MODEL_NAME))
else:
    # A directly saved sklearn Pipeline can accept the raw input DataFrame.
    PIPELINE = bundle if hasattr(bundle, "predict") else None

if PIPELINE is None and (MODEL is None or PREPROCESSOR is None):
    raise ValueError(
        "The model artifact was loaded, but its estimator/preprocessor could not "
        "be identified. Expected a dictionary containing 'model' and 'preprocessor' "
        "(or a complete sklearn Pipeline). Check how loan_approval_model.joblib was saved."
    )


def predict_loan(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
):
    """Predict a dataset label and return the model's approval probability."""
    values = [
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
    ]
    if any(value is None for value in values):
        return "Please complete every field before submitting.", 0.0

    applicant = pd.DataFrame([dict(zip(FEATURE_COLUMNS, values))], columns=FEATURE_COLUMNS)
    applicant["education"] = applicant["education"].astype(str).str.strip()
    applicant["self_employed"] = applicant["self_employed"].astype(str).str.strip()

    if PIPELINE is not None:
        prediction = PIPELINE.predict(applicant)[0]
        probabilities = PIPELINE.predict_proba(applicant)[0] if hasattr(PIPELINE, "predict_proba") else None
        classes = list(getattr(PIPELINE, "classes_", []))
    else:
        transformed = PREPROCESSOR.transform(applicant)
        prediction = MODEL.predict(transformed)[0]
        probabilities = MODEL.predict_proba(transformed)[0] if hasattr(MODEL, "predict_proba") else None
        classes = list(getattr(MODEL, "classes_", []))

    # Notebook target mapping: Rejected = 0, Approved = 1.
    is_approved = str(prediction).strip().lower() == "approved" or prediction == 1
    status = "Approved" if is_approved else "Rejected"

    if probabilities is not None and classes:
        approval_index = next(
            (index for index, value in enumerate(classes)
             if str(value).strip().lower() == "approved" or value == 1),
            None,
        )
        approval_probability = float(probabilities[approval_index]) * 100 if approval_index is not None else 0.0
    else:
        approval_probability = 0.0

    result = (
        f"### Predicted Loan Status: {status}\n\n"
        f"**Model:** {MODEL_NAME}\n\n"
        "*Educational demonstration only. This is a model prediction, "
        "not an actual bank decision or financial advice.*"
    )
    return result, round(approval_probability, 2)


demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Number of Dependents", value=2, precision=0),
        gr.Dropdown(["Graduate", "Not Graduate"], label="Education", value="Graduate"),
        gr.Dropdown(["Yes", "No"], label="Self Employed", value="No"),
        gr.Number(label="Annual Income", value=5000000),
        gr.Number(label="Loan Amount", value=15000000),
        gr.Number(label="Loan Term", value=10, precision=0),
        gr.Number(label="CIBIL Score", value=700, precision=0),
        gr.Number(label="Residential Assets Value", value=5000000),
        gr.Number(label="Commercial Assets Value", value=2000000),
        gr.Number(label="Luxury Assets Value", value=8000000),
        gr.Number(label="Bank Asset Value", value=3000000),
    ],
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Number(label="Estimated Approval Probability (%)"),
    ],
    title="Loan Approval Prediction System",
    description=(
        "Enter applicant details to see the trained model's prediction. "
        "This project is for academic demonstration only."
    ),
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
