"""Premium, interactive Gradio app for the Loan Approval Prediction project.

Place app.py beside loan_approval_model.joblib in the repository root.
This is an academic demonstration, not a real lending decision system.
"""
from __future__ import annotations

import html
import logging
import os
from pathlib import Path
from typing import Any

import gradio as gr
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("loan_approval_app")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_approval_model.joblib"

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

DEFAULT_VALUES = [2, "Graduate", "No", 5_000_000, 15_000_000, 10, 700,
                  5_000_000, 2_000_000, 8_000_000, 3_000_000]

# These metrics are from the project's previously evaluated held-out test split.
# They describe this dataset split only, not performance at a real lender.
TEST_METRICS = {
    "Accuracy": 98.90,
    "Precision": 98.75,
    "Recall": 99.49,
    "F1-score": 99.12,
    "ROC-AUC": 99.81,
}

# Global feature importances reported by the project's Random Forest analysis.
# These are associations used by the model, not causal effects.
FEATURE_IMPORTANCE = [
    ("CIBIL score", 81.81),
    ("Loan term", 5.68),
    ("Loan amount", 2.93),
    ("Luxury assets", 1.81),
    ("Annual income", 1.72),
    ("Commercial assets", 1.66),
    ("Bank assets", 1.54),
    ("Residential assets", 1.53),
    ("Number of dependents", 0.83),
    ("Self-employed", 0.26),
    ("Education", 0.21),
]


def first_present(mapping: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Required file loan_approval_model.joblib was not found. Upload it to the "
        "same repository folder as app.py."
    )

bundle = joblib.load(MODEL_PATH)
PIPELINE = None
MODEL = None
PREPROCESSOR = None
MODEL_NAME = "Random Forest"

if isinstance(bundle, dict):
    PIPELINE = first_present(bundle, ["pipeline", "model_pipeline", "full_pipeline"])
    MODEL = first_present(
        bundle,
        ["model", "best_model", "selected_model", "classifier", "estimator", "trained_model"],
    )
    PREPROCESSOR = first_present(
        bundle,
        ["preprocessor", "preprocessing", "transformer", "feature_preprocessor"],
    )
    MODEL_NAME = str(bundle.get("model_name", MODEL_NAME))
else:
    PIPELINE = bundle if hasattr(bundle, "predict") else None

if PIPELINE is None and (MODEL is None or PREPROCESSOR is None):
    raise ValueError(
        "The saved artifact loaded, but its estimator/preprocessor could not be identified. "
        "Check how loan_approval_model.joblib was saved."
    )


def _result_card(state: str = "ready", message: str | None = None) -> str:
    if state == "error":
        safe_message = html.escape(message or "Check the values and try again.")
        return f"""
        <section class="result-card result-error">
          <div class="result-topline"><span class="result-dot"></span> ANALYSIS NEEDS ATTENTION</div>
          <div class="result-title">Unable to complete analysis</div>
          <p class="result-copy">{safe_message}</p>
        </section>"""
    return """
    <section class="result-card result-empty">
      <div class="result-ornament">✳</div>
      <div class="result-topline"><span class="result-dot"></span> PREDICTION WORKSPACE</div>
      <div class="result-title">Your result starts here.</div>
      <p class="result-copy">Enter applicant details, then run the model to see a predicted class and its estimated approval probability.</p>
      <div class="result-placeholder"><span>01</span><span>Applicant data</span><i></i><span>02</span><span>Model output</span></div>
    </section>"""


def _probability_card(probability: float | None = None) -> str:
    if probability is None:
        return """
        <section class="probability-card">
          <div class="metric-head"><span class="metric-label">Estimated approval probability</span><span class="metric-tag">MODEL OUTPUT</span></div>
          <div class="probability-value muted">—<span>%</span></div>
          <div class="probability-track"><div class="probability-fill" style="width:0%"></div></div>
          <div class="probability-foot"><span>Awaiting analysis</span><span>Not a guarantee</span></div>
        </section>"""
    shown = max(0.0, min(100.0, float(probability)))
    return f"""
    <section class="probability-card">
      <div class="metric-head"><span class="metric-label">Estimated approval probability</span><span class="metric-tag">MODEL OUTPUT</span></div>
      <div class="probability-value">{shown:.2f}<span>%</span></div>
      <div class="probability-track"><div class="probability-fill" style="width:{shown:.2f}%"></div></div>
      <div class="probability-foot"><span>Probability for the Approved class</span><span>Not a guarantee</span></div>
    </section>"""


def _is_approved(value: Any) -> bool:
    return str(value).strip().lower() in {"approved", "1", "true"}


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
    values = [
        no_of_dependents, education, self_employed, income_annum, loan_amount,
        loan_term, cibil_score, residential_assets_value,
        commercial_assets_value, luxury_assets_value, bank_asset_value,
    ]
    if any(value is None for value in values):
        return _result_card("error", "Complete every field before running the analysis."), _probability_card()

    try:
        numeric_values = [
            no_of_dependents, income_annum, loan_amount, loan_term, cibil_score,
            residential_assets_value, commercial_assets_value,
            luxury_assets_value, bank_asset_value,
        ]
        if any(float(value) < 0 for value in numeric_values):
            return _result_card("error", "Numeric values cannot be negative."), _probability_card()
        if not 0 <= int(no_of_dependents) <= 5:
            return _result_card("error", "Number of dependents must be between 0 and 5."), _probability_card()
        if float(no_of_dependents) != int(no_of_dependents):
            return _result_card("error", "Number of dependents must be a whole number."), _probability_card()
        if not 300 <= float(cibil_score) <= 900:
            return _result_card("error", "CIBIL score must be between 300 and 900."), _probability_card()
        if float(cibil_score) != int(cibil_score):
            return _result_card("error", "CIBIL score must be a whole number."), _probability_card()
        if not 2 <= float(loan_term) <= 20:
            return _result_card("error", "Loan term must be between 2 and 20, as in the source dataset."), _probability_card()
        if float(loan_term) != int(loan_term):
            return _result_card("error", "Loan term must be a whole number."), _probability_card()
        if education not in {"Graduate", "Not Graduate"}:
            return _result_card("error", "Choose one of the education categories."), _probability_card()
        if self_employed not in {"Yes", "No"}:
            return _result_card("error", "Choose Yes or No for self-employment."), _probability_card()

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

        approved = _is_approved(prediction)
        status = "Approved" if approved else "Rejected"
        status_class = "result-approved" if approved else "result-rejected"
        status_icon = "✓" if approved else "×"
        status_copy = (
            "The model classified this sample as approved based on the supplied features."
            if approved else
            "The model classified this sample as rejected based on the supplied features."
        )

        approval_probability = None
        if probabilities is not None and classes:
            approval_index = next((i for i, cls in enumerate(classes) if _is_approved(cls)), None)
            if approval_index is not None:
                approval_probability = float(probabilities[approval_index]) * 100

        safe_model = html.escape(MODEL_NAME)
        result_html = f"""
        <section class="result-card {status_class}">
          <div class="result-topline"><span class="result-dot"></span> MODEL PREDICTION <span class="result-chip">{safe_model}</span></div>
          <div class="result-heading"><span class="result-status-icon">{status_icon}</span><span>{status}</span></div>
          <p class="result-copy">{status_copy}</p>
          <div class="result-divider"></div>
          <div class="result-detail-row"><span>Decision class</span><strong>{status}</strong></div>
          <div class="result-detail-row"><span>Model</span><strong>{safe_model}</strong></div>
          <div class="result-disclaimer">Academic demonstration only. This output is not a real bank decision, financial advice, or a guarantee of approval.</div>
        </section>"""
        return result_html, _probability_card(approval_probability)
    except Exception:
        LOGGER.exception("Prediction failed")
        return _result_card("error", "The model could not process these inputs. Check the values or review the server logs."), _probability_card()


def apply_profile(profile: str):
    """Load illustrative example values; user must still run the prediction."""
    profiles = {
        "Balanced sample": [2, "Graduate", "No", 5_000_000, 15_000_000, 10, 700, 5_000_000, 2_000_000, 8_000_000, 3_000_000],
        "Strong credit profile": [1, "Graduate", "No", 8_000_000, 12_000_000, 8, 820, 12_000_000, 6_000_000, 10_000_000, 7_000_000],
        "Lower credit score": [3, "Not Graduate", "Yes", 2_000_000, 18_000_000, 18, 480, 1_000_000, 500_000, 1_500_000, 200_000],
    }
    return [gr.update(value=value) for value in profiles[profile]]


def reset_form():
    return tuple(gr.update(value=value) for value in DEFAULT_VALUES) + (_result_card(), _probability_card())


def metric_card(value: str, label: str, note: str = "") -> str:
    safe_value, safe_label, safe_note = map(html.escape, (value, label, note))
    return f"""
    <div class="insight-metric"><div class="insight-value">{safe_value}</div>
      <div class="insight-label">{safe_label}</div><div class="insight-note">{safe_note}</div></div>"""


def importance_bars() -> str:
    rows = []
    for label, value in FEATURE_IMPORTANCE:
        safe_label = html.escape(label)
        rows.append(
            f'<div class="importance-row"><div class="importance-head"><span>{safe_label}</span>'
            f'<strong>{value:.2f}%</strong></div><div class="importance-track">'
            f'<div class="importance-fill" style="width:{value:.2f}%"></div></div></div>'
        )
    return '<div class="importance-list">' + "".join(rows) + "</div>"


CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#07111e; --panel:#0d1b2b; --panel-2:#102338; --line:rgba(162,190,213,.15); --ink:#edf6ff; --muted:#92a8bb; --mint:#69eed0; --teal:#19bba5; --gold:#e0c28c; }
html, body { background:#07111e !important; }
body, .gradio-container { background:radial-gradient(ellipse at 15% -8%,rgba(22,105,103,.22),transparent 33%),radial-gradient(ellipse at 100% 28%,rgba(46,72,122,.17),transparent 34%),#07111e !important; color:var(--ink) !important; font-family:'DM Sans',sans-serif !important; }
.gradio-container { max-width:1420px !important; padding:22px 24px 36px !important; margin:auto !important; }
footer { display:none !important; }
header { background:transparent !important; }
.hero { position:relative; isolation:isolate; overflow:hidden; border-radius:28px; padding:32px 36px; margin:0 0 18px; background:linear-gradient(116deg,rgba(13,29,47,.98) 0%,rgba(13,43,55,.96) 62%,rgba(8,91,83,.86) 125%); border:1px solid rgba(151,220,211,.16); box-shadow:0 24px 70px rgba(0,0,0,.24),inset 0 1px rgba(255,255,255,.05); }
.hero:before { content:''; position:absolute; z-index:-1; width:440px;height:440px;right:-120px;top:-230px;border-radius:50%;border:1px solid rgba(130,241,219,.18);box-shadow:0 0 0 36px rgba(130,241,219,.035),0 0 0 76px rgba(130,241,219,.025),0 0 80px rgba(23,200,170,.12); }
.hero:after { content:'';position:absolute;right:16%;bottom:-95px;width:210px;height:180px;background:radial-gradient(ellipse,rgba(24,188,164,.15),transparent 70%);filter:blur(10px);pointer-events:none; }
.hero-layout { display:flex; justify-content:space-between; align-items:center; gap:28px; position:relative; z-index:1; }
.hero-copy { max-width:760px; }
.brand-line { display:flex; align-items:center; gap:10px; color:#b7d2df; text-transform:uppercase; letter-spacing:1.8px; font-size:10px; font-weight:800; margin-bottom:19px; }
.brand-mark { display:inline-flex; width:29px;height:29px;align-items:center;justify-content:center;border:1px solid rgba(115,236,212,.34);border-radius:10px;color:var(--mint);background:rgba(50,183,158,.09);font-size:17px; }
.hero h1 { font-family:'Manrope',sans-serif !important; color:#f6fbff !important;font-size:clamp(30px,4.2vw,48px) !important;line-height:1.08 !important;letter-spacing:-1.8px !important;font-weight:800 !important;margin:0 0 15px !important; }
.hero h1 em { color:var(--mint);font-style:normal; }
.hero p { max-width:680px;color:#b3c9d8;font-size:14px;line-height:1.8;margin:0; }
.hero-chips { display:flex;gap:8px;flex-wrap:wrap;margin-top:22px; }
.hero-chip { padding:8px 11px;border:1px solid rgba(197,226,235,.15);border-radius:999px;background:rgba(255,255,255,.035);color:#d0e3ed;font-size:10px;font-weight:700;letter-spacing:.2px; }
.hero-side { flex:0 0 210px; padding:21px; border-radius:20px; border:1px solid rgba(186,239,225,.17); background:linear-gradient(155deg,rgba(255,255,255,.08),rgba(255,255,255,.025)); box-shadow:inset 0 1px rgba(255,255,255,.06); backdrop-filter:blur(16px); }
.hero-side-label { color:#a3c6d0;font-size:10px;font-weight:800;letter-spacing:1.4px;text-transform:uppercase; }
.hero-side-value { font-family:'Manrope',sans-serif;color:#83f0d5;font-size:37px;font-weight:800;letter-spacing:-1.5px;line-height:1.3;margin:8px 0 2px; }
.hero-side-caption { color:#9cb7c6;font-size:11px;line-height:1.55; }
.hero-side-line { height:1px;background:rgba(215,244,239,.12);margin:16px 0 12px; }
.hero-side-foot { display:flex;align-items:center;justify-content:space-between;gap:8px;color:#c5d9e3;font-size:11px; }
.online-dot { width:7px;height:7px;border-radius:50%;background:#69eed0;box-shadow:0 0 10px rgba(105,238,208,.65);display:inline-block;margin-right:6px; }
.section-kicker { display:flex;align-items:center;justify-content:space-between;gap:10px;margin:24px 2px 13px; }
.section-kicker h2 { color:#edf6ff !important;font-family:'Manrope',sans-serif !important;font-size:18px !important;letter-spacing:-.4px !important;font-weight:800 !important;margin:0 !important; }
.section-kicker p { color:#8198ad !important;font-size:11px !important;margin:4px 0 0 !important; }
.micro-badge { color:#8edfcf;border:1px solid rgba(105,238,208,.19);background:rgba(31,171,147,.08);border-radius:999px;padding:7px 10px;font-size:10px;font-weight:800;white-space:nowrap; }
.panel { background:linear-gradient(145deg,rgba(15,31,48,.94),rgba(11,25,40,.95)) !important;border:1px solid var(--line) !important;border-radius:20px !important;padding:21px !important;box-shadow:0 14px 32px rgba(0,0,0,.13),inset 0 1px rgba(255,255,255,.025) !important;margin-bottom:14px !important; }
.panel-heading { display:flex;align-items:flex-start;gap:12px;margin:0 0 5px; }
.panel-index { display:inline-flex;flex:0 0 31px;width:31px;height:31px;align-items:center;justify-content:center;border-radius:10px;color:var(--mint);border:1px solid rgba(105,238,208,.17);background:rgba(37,174,148,.09);font-family:'Manrope',sans-serif;font-size:11px;font-weight:800; }
.panel-title { color:#eaf5fc;font-family:'Manrope',sans-serif;font-weight:800;font-size:14px;letter-spacing:-.2px; }
.panel-subtitle { color:#7992a8;font-size:11px;margin:5px 0 17px;line-height:1.6; }
label, .wrap label, .label-wrap, .svelte-1gfkn6j { color:#b7c9d8 !important;font-size:11px !important;font-weight:700 !important; }
input, textarea, .wrap input, .wrap textarea, .wrap select, select { background:#0b1a2a !important;color:#edf6ff !important;border:1px solid #20384c !important;border-radius:11px !important;min-height:42px !important; }
input::placeholder, textarea::placeholder { color:#60778d !important; }
input:focus, textarea:focus, select:focus { border-color:#2ac4aa !important;box-shadow:0 0 0 2px rgba(42,196,170,.11) !important; }
.block, .form, .wrap, .gr-box { border-radius:12px !important; }
.gradio-container .tabs { border-bottom:1px solid rgba(162,190,213,.12) !important; }
.gradio-container button[role="tab"] { color:#92a8bb !important;font-size:12px !important;font-weight:800 !important;border:0 !important;background:transparent !important; }
.gradio-container button[role="tab"][aria-selected="true"] { color:#8bf0d8 !important;border-bottom:2px solid #44d5b9 !important; }
.preset-label { color:#88a0b5;font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:1.1px;margin:0 0 9px; }
.preset-btn { border-radius:10px !important;background:#102438 !important;border:1px solid #263f53 !important;color:#cde0eb !important;font-size:10px !important;font-weight:800 !important;min-height:38px !important; }
.preset-btn:hover { border-color:#41cdb4 !important;background:#133447 !important; }
.predict-btn { min-height:51px !important;border-radius:13px !important;font-family:'Manrope',sans-serif !important;font-weight:800 !important;font-size:13px !important;letter-spacing:.1px !important;background:linear-gradient(105deg,#19a891,#42d8b5) !important;color:#06251f !important;border:0 !important;box-shadow:0 10px 25px rgba(13,179,151,.16) !important;transition:transform .18s,box-shadow .18s !important; }
.predict-btn:hover { transform:translateY(-1px);box-shadow:0 13px 28px rgba(13,179,151,.23) !important; }
.reset-btn { min-height:42px !important;border-radius:12px !important;border:1px solid #263c50 !important;background:#102033 !important;color:#a8bdcd !important;font-size:11px !important;font-weight:800 !important; }
.result-card { border-radius:17px;padding:21px;border:1px solid rgba(164,195,215,.16);background:linear-gradient(145deg,#0e2031,#0b1a29);min-height:222px;overflow:hidden;position:relative; }
.result-card:after { content:'';position:absolute;right:-46px;top:-60px;width:150px;height:150px;border:1px solid rgba(151,205,210,.09);border-radius:50%;box-shadow:0 0 0 22px rgba(151,205,210,.018),0 0 0 45px rgba(151,205,210,.014);pointer-events:none; }
.result-empty { background:linear-gradient(145deg,#102437,#0c1b2b); }
.result-approved { background:linear-gradient(145deg,rgba(9,68,59,.52),#0b1c2b 75%);border-color:rgba(90,227,191,.28); }
.result-rejected { background:linear-gradient(145deg,rgba(92,41,48,.4),#0b1c2b 75%);border-color:rgba(239,139,133,.26); }
.result-error { background:linear-gradient(145deg,rgba(92,41,48,.32),#0b1c2b);border-color:rgba(239,139,133,.26); }
.result-topline { display:flex;align-items:center;gap:7px;color:#8ba6b9;font-size:9px;letter-spacing:1.25px;font-weight:900;margin-bottom:20px; }
.result-dot { width:6px;height:6px;border-radius:50%;background:#43d8b5;display:inline-block;box-shadow:0 0 10px rgba(67,216,181,.5); }
.result-chip { margin-left:auto;border:1px solid rgba(162,190,213,.16);border-radius:999px;padding:5px 7px;color:#abc3d1;font-size:9px;letter-spacing:0; }
.result-ornament { display:flex;align-items:center;justify-content:center;width:39px;height:39px;border-radius:13px;background:rgba(71,215,183,.1);color:#73edce;font-size:22px;margin-bottom:16px; }
.result-title { color:#e9f5ff;font-family:'Manrope',sans-serif;font-size:22px;line-height:1.25;font-weight:800;letter-spacing:-.6px;margin:0 0 10px;position:relative;z-index:1; }
.result-heading { display:flex;align-items:center;gap:11px;color:#7ff0ce;font-family:'Manrope',sans-serif;font-weight:800;font-size:27px;line-height:1.2;position:relative;z-index:1; }
.result-rejected .result-heading { color:#ffaaa3; }
.result-status-icon { display:inline-flex;flex:0 0 36px;height:36px;width:36px;justify-content:center;align-items:center;border-radius:12px;background:rgba(72,219,179,.12);color:#80efd0;font-size:21px; }
.result-rejected .result-status-icon { background:rgba(255,129,119,.11);color:#ffaaa3; }
.result-copy { color:#9bb1c2;font-size:12px;line-height:1.7;margin:13px 0 0;position:relative;z-index:1; }
.result-divider { height:1px;background:rgba(162,190,213,.13);margin:17px 0 12px; }
.result-detail-row { display:flex;align-items:center;justify-content:space-between;gap:14px;color:#7f98ac;font-size:11px;margin:9px 0; }
.result-detail-row strong { color:#d8e9f3;font-size:11px; }
.result-disclaimer { color:#6f879a;font-size:10px;line-height:1.6;margin-top:16px;padding-top:13px;border-top:1px solid rgba(162,190,213,.1);position:relative;z-index:1; }
.probability-card { border:1px solid rgba(162,190,213,.15);background:linear-gradient(145deg,rgba(16,36,54,.96),rgba(10,26,42,.98));border-radius:17px;padding:20px;margin-top:13px; }
.metric-head { display:flex;align-items:center;justify-content:space-between;gap:12px; }
.metric-label { color:#a9bfd0;font-size:11px;font-weight:800; }
.metric-tag { font-size:8px;letter-spacing:1px;color:#6fdcc4;border:1px solid rgba(105,238,208,.17);background:rgba(105,238,208,.05);padding:5px 7px;border-radius:999px;font-weight:900;white-space:nowrap; }
.probability-value { color:#7cebd0;font-family:'Manrope',sans-serif;font-size:42px;font-weight:800;line-height:1.18;letter-spacing:-1.6px;margin:13px 0 15px; }
.probability-value span { color:#4ea998;font-size:18px;margin-left:3px;letter-spacing:0; }
.probability-value.muted { color:#526b7e; }
.probability-track { width:100%;height:7px;background:#1e3447;border-radius:999px;overflow:hidden; }
.probability-fill { height:100%;border-radius:999px;background:linear-gradient(90deg,#148e82,#78f1d0);box-shadow:0 0 15px rgba(60,221,185,.3);transition:width .4s ease; }
.probability-foot { display:flex;justify-content:space-between;gap:10px;color:#718a9e;font-size:9px;line-height:1.5;margin-top:11px; }
.note-card { border:1px solid rgba(162,190,213,.12);background:rgba(10,25,41,.74);border-radius:14px;padding:14px 15px;color:#8199ad;font-size:10px;line-height:1.7;margin-top:13px; }
.note-card strong { color:#c5d8e5; }
.footer { display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;padding:20px 3px 0;color:#617b91;font-size:10px; }
.footer b { color:#8faabc;font-weight:800; }
.insight-metric { border:1px solid rgba(162,190,213,.14);border-radius:15px;background:linear-gradient(145deg,#102337,#0b1b2a);padding:18px;min-height:112px; }
.insight-value { font-family:'Manrope',sans-serif;font-weight:800;color:#79ebcf;font-size:27px;letter-spacing:-1px; }
.insight-label { color:#d4e3ed;font-weight:800;font-size:11px;margin-top:5px; }
.insight-note { color:#7891a5;font-size:9px;margin-top:5px; }
.method-card { border:1px solid rgba(162,190,213,.14);border-radius:16px;background:linear-gradient(145deg,#0f2133,#0b1a29);padding:19px;height:100%; }
.method-title { color:#e7f3fa;font-family:'Manrope',sans-serif;font-size:13px;font-weight:800;margin:0 0 9px; }
.method-copy { color:#8da5b8;font-size:11px;line-height:1.8;margin:0; }
.method-list { display:flex;flex-direction:column;gap:10px;margin-top:13px; }
.method-item { display:flex;gap:10px;color:#9ab1c2;font-size:11px;line-height:1.6; }
.method-check { display:inline-flex;flex:0 0 19px;width:19px;height:19px;align-items:center;justify-content:center;border-radius:7px;background:rgba(63,213,180,.1);color:#74e8cd;font-size:11px;font-weight:900; }
.importance-row { margin:0 0 12px; }
.importance-head { display:flex;align-items:center;justify-content:space-between;gap:12px;color:#a9bdcb;font-size:10px;margin-bottom:6px; }
.importance-head strong { color:#dceaf2;font-size:10px; }
.importance-track { height:6px;border-radius:999px;overflow:hidden;background:#1a3042; }
.importance-fill { height:100%;border-radius:999px;background:linear-gradient(90deg,#178f86,#71e8ca); }
.disclaimer-banner { border:1px solid rgba(224,194,140,.19);background:rgba(124,91,38,.075);border-radius:13px;padding:13px 15px;color:#b9a987;font-size:10px;line-height:1.75;margin-top:14px; }
.disclaimer-banner strong { color:#e0c28c; }
@media (max-width: 850px) { .gradio-container {padding:12px 12px 26px !important;} .hero {padding:24px 22px;border-radius:21px;} .hero-layout {align-items:flex-start;flex-direction:column;} .hero-side {width:100%;flex:1;} .hero-side-value {font-size:32px;} .panel {padding:16px !important;} .section-kicker {margin-top:19px;} .result-card {min-height:unset;} .footer {flex-direction:column;align-items:flex-start;} }
@media (prefers-reduced-motion: reduce) { *,*::before,*::after {transition:none !important;animation:none !important;scroll-behavior:auto !important;} }

/* Premium Emerald Glass Theme */
:root {
  --bg: #10251f;
  --panel: rgba(24, 53, 43, 0.78);
  --panel-2: rgba(35, 69, 56, 0.72);
  --line: rgba(218, 197, 151, 0.22);
  --ink: #f5f2e9;
  --muted: #c0cbbf;
  --mint: #d9bd7e;
  --teal: #4eb99a;
  --gold: #e1c58b;
}

html, body,
body .gradio-container {
  background:
    radial-gradient(ellipse at 10% 0%, rgba(61, 133, 99, 0.25), transparent 40%),
    radial-gradient(ellipse at 95% 30%, rgba(195, 161, 94, 0.12), transparent 36%),
    #10251f !important;
  color: #f5f2e9 !important;
}

.hero {
  background: linear-gradient(
    125deg,
    rgba(17, 48, 38, 0.96),
    rgba(35, 83, 63, 0.91),
    rgba(79, 111, 78, 0.82)
  ) !important;
  border-color: rgba(218, 197, 151, 0.28) !important;
  backdrop-filter: blur(22px);
}

.hero h1 em,
.hero-side-value,
.result-ornament,
.probability-value {
  color: #e3c98e !important;
}

.hero-side,
.panel,
.result-card,
.probability-card,
.insight-metric,
.method-card,
.note-card {
  background: rgba(229, 239, 229, 0.065) !important;
  border: 1px solid rgba(218, 225, 202, 0.17) !important;
  backdrop-filter: blur(18px);
  box-shadow:
    0 12px 35px rgba(0, 0, 0, 0.15),
    inset 0 1px rgba(255, 255, 255, 0.06) !important;
}

input,
textarea,
.wrap input,
.wrap textarea,
.wrap select,
select {
  background: rgba(10, 34, 26, 0.72) !important;
  color: #f5f2e9 !important;
  border: 1px solid rgba(218, 197, 151, 0.23) !important;
}

input:focus,
textarea:focus,
select:focus {
  border-color: #d9bd7e !important;
  box-shadow: 0 0 0 2px rgba(217, 189, 126, 0.15) !important;
}

.predict-btn {
  background: linear-gradient(110deg, #c8a967, #e5d2a1) !important;
  color: #17372b !important;
  box-shadow: 0 8px 24px rgba(200, 169, 103, 0.17) !important;
}

.predict-btn:hover {
  background: linear-gradient(110deg, #d8bb7b, #f0dfb4) !important;
}

.reset-btn,
.preset-btn {
  background: rgba(220, 235, 223, 0.07) !important;
  color: #e9e9dc !important;
  border-color: rgba(218, 197, 151, 0.22) !important;
}

.micro-badge,
.metric-tag {
  color: #e1c58b !important;
  background: rgba(217, 189, 126, 0.08) !important;
  border-color: rgba(217, 189, 126, 0.24) !important;
}

.probability-track,
.importance-track {
  background: rgba(224, 235, 222, 0.12) !important;
}

.probability-fill,
.importance-fill {
  background: linear-gradient(90deg, #3a9875, #9bd4a8) !important;
}

.result-approved {
  border-color: rgba(123, 211, 161, 0.35) !important;
}

.result-rejected,
.result-error {
  border-color: rgba(231, 143, 130, 0.32) !important;
}

@media (max-width: 850px) {
  .hero {
    padding: 22px 18px !important;
  }

  .panel {
    padding: 15px !important;
  }

  .hero-layout {
    gap: 18px;
  }
}

"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    radius_size="lg",
    spacing_size="lg",
)

with gr.Blocks(theme=THEME, css=CSS, title="Loan Approval Intelligence | ML Studio", analytics_enabled=False) as demo:
    gr.HTML(
        f"""
        <header class="hero">
          <div class="hero-layout">
            <div class="hero-copy">
              <div class="brand-line"><span class="brand-mark">◈</span> MODEL LAB <span style="opacity:.5">/</span> INTERACTIVE FINTECH ANALYTICS</div>
              <h1>Loan Approval<br><em>Intelligence.</em></h1>
              <p>A polished machine-learning demo that turns applicant and financial attributes into an interpretable prediction workflow. Explore sample profiles, run the trained model, and inspect the project's evaluation metrics.</p>
              <div class="hero-chips"><span class="hero-chip">✦ Random Forest</span><span class="hero-chip">◉ 11 input features</span><span class="hero-chip">⌁ Interactive analysis</span><span class="hero-chip">Academic portfolio project</span></div>
            </div>
            <aside class="hero-side">
              <div class="hero-side-label">Held-out test accuracy</div>
              <div class="hero-side-value">{TEST_METRICS['Accuracy']:.2f}%</div>
              <div class="hero-side-caption">Measured on the project's test split — not a guarantee of real-world lending performance.</div>
              <div class="hero-side-line"></div>
              <div class="hero-side-foot"><span><i class="online-dot"></i>Model loaded</span><span>{html.escape(MODEL_NAME)}</span></div>
            </aside>
          </div>
        </header>
        """
    )

    gr.Markdown(
        """<div class="section-kicker"><div><h2>Interactive workspace</h2><p>Experiment with applicant attributes and observe the model's output.</p></div><span class="micro-badge">DEMO ENVIRONMENT</span></div>"""
    )

    with gr.Tabs():
        with gr.Tab("◈  Applicant simulator"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=7, min_width=320):
                    with gr.Group(elem_classes=["panel"]):
                        gr.HTML('<div class="panel-heading"><span class="panel-index">01</span><div class="panel-title">Quick scenarios</div></div><div class="panel-subtitle">Load an illustrative profile to explore how changing inputs affects the model output.</div>')
                        gr.HTML('<div class="preset-label">Try a sample profile</div>')
                        with gr.Row():
                            balanced_btn = gr.Button("Balanced sample", elem_classes=["preset-btn"])
                            strong_btn = gr.Button("Strong credit profile", elem_classes=["preset-btn"])
                            low_score_btn = gr.Button("Lower credit score", elem_classes=["preset-btn"])
                        gr.HTML('<div class="note-card">Sample values are illustrative. They are not real applicants and are not intended as financial guidance.</div>')

                    with gr.Group(elem_classes=["panel"]):
                        gr.HTML('<div class="panel-heading"><span class="panel-index">02</span><div class="panel-title">Applicant profile</div></div><div class="panel-subtitle">Personal and employment attributes</div>')
                        with gr.Row():
                            dependents = gr.Number(label="Number of dependents", value=2, minimum=0, maximum=5, precision=0, info="Whole number · 0–5")
                            education = gr.Dropdown(["Graduate", "Not Graduate"], label="Education", value="Graduate", allow_custom_value=False)
                        with gr.Row():
                            self_employed = gr.Dropdown(["Yes", "No"], label="Self-employed", value="No", allow_custom_value=False)
                            cibil_score = gr.Number(label="CIBIL score", value=700, minimum=300, maximum=900, precision=0, info="300–900")

                    with gr.Group(elem_classes=["panel"]):
                        gr.HTML('<div class="panel-heading"><span class="panel-index">03</span><div class="panel-title">Loan details</div></div><div class="panel-subtitle">Annual income and requested lending terms · enter values in the source dataset’s units</div>')
                        with gr.Row():
                            income_annum = gr.Number(label="Annual income", value=5_000_000, minimum=0, precision=0)
                            loan_amount = gr.Number(label="Requested loan amount", value=15_000_000, minimum=0, precision=0)
                        with gr.Row():
                            loan_term = gr.Number(label="Loan term", value=10, minimum=2, maximum=20, precision=0, info="Whole number · 2–20")
                            gr.HTML('<div class="note-card" style="margin-top:22px"><strong>Input note</strong><br>Amounts are passed directly to the saved preprocessing pipeline. No currency conversion is performed.</div>')

                    with gr.Group(elem_classes=["panel"]):
                        gr.HTML('<div class="panel-heading"><span class="panel-index">04</span><div class="panel-title">Asset profile</div></div><div class="panel-subtitle">Declared assets used as model features</div>')
                        with gr.Row():
                            residential_assets = gr.Number(label="Residential assets", value=5_000_000, minimum=0, precision=0)
                            commercial_assets = gr.Number(label="Commercial assets", value=2_000_000, minimum=0, precision=0)
                        with gr.Row():
                            luxury_assets = gr.Number(label="Luxury assets", value=8_000_000, minimum=0, precision=0)
                            bank_assets = gr.Number(label="Bank assets", value=3_000_000, minimum=0, precision=0)

                    input_components = [dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets, commercial_assets, luxury_assets, bank_assets]
                    with gr.Row():
                        predict_button = gr.Button("Run prediction  →", variant="primary", elem_classes=["predict-btn"], scale=3)
                        reset_button = gr.Button("Reset inputs", variant="secondary", elem_classes=["reset-btn"], scale=1)

                with gr.Column(scale=5, min_width=300):
                    gr.HTML('<div class="section-kicker" style="margin-top:0"><div><h2>Prediction summary</h2><p>Results update after analysis.</p></div><span class="micro-badge">LIVE OUTPUT</span></div>')
                    result_output = gr.HTML(value=_result_card())
                    probability_output = gr.HTML(value=_probability_card())
                    gr.HTML('<div class="note-card"><strong>How to read this:</strong> the class is the model’s predicted label. The percentage is the model-estimated probability for the Approved class; it is not necessarily calibrated and does not guarantee a real lending outcome.</div>')

            predict_button.click(fn=predict_loan, inputs=input_components, outputs=[result_output, probability_output], api_name="predict_loan", show_progress="minimal")
            reset_button.click(fn=reset_form, inputs=None, outputs=input_components + [result_output, probability_output], api_name="reset_form")
            for button, profile in [(balanced_btn, "Balanced sample"), (strong_btn, "Strong credit profile"), (low_score_btn, "Lower credit score")]:
                button.click(fn=lambda p=profile: apply_profile(p), inputs=None, outputs=input_components, api_name=f"load_{profile.lower().replace(' ', '_')}")

        with gr.Tab("▤  Model insights"):
            gr.Markdown("""<div class="section-kicker"><div><h2>Evaluation snapshot</h2><p>Reported metrics from the held-out test split used in this project.</p></div><span class="micro-badge">637 TEST ROWS</span></div>""")
            with gr.Row():
                for value, label, note in [
                    (f"{TEST_METRICS['Accuracy']:.2f}%", "Accuracy", "Overall correct predictions"),
                    (f"{TEST_METRICS['Precision']:.2f}%", "Precision", "Approved predictions that were correct"),
                    (f"{TEST_METRICS['Recall']:.2f}%", "Recall", "Approved cases identified"),
                    (f"{TEST_METRICS['ROC-AUC']:.2f}%", "ROC-AUC", "Class separation metric"),
                ]:
                    gr.HTML(metric_card(value, label, note))

            with gr.Row(equal_height=False):
                with gr.Column(scale=6, min_width=300):
                    with gr.Group(elem_classes=["panel"]):
                        gr.HTML('<div class="panel-heading"><span class="panel-index">A</span><div class="panel-title">Global feature importance</div></div><div class="panel-subtitle">Reported from the project’s Random Forest analysis. Higher values indicate greater contribution to the model’s split-based importance score, not causation.</div>')
                        gr.HTML(importance_bars())
                with gr.Column(scale=6, min_width=300):
                    with gr.Group(elem_classes=["panel"]):
                        gr.HTML('<div class="panel-heading"><span class="panel-index">B</span><div class="panel-title">Methodology at a glance</div></div><div class="panel-subtitle">Reproducible supervised-learning workflow</div>')
                        gr.HTML("""
                        <div class="method-card">
                          <div class="method-title">Model pipeline</div><p class="method-copy">The project compares Logistic Regression, Decision Tree, and Random Forest models; the Random Forest is selected for the final held-out test evaluation.</p>
                          <div class="method-list">
                            <div class="method-item"><span class="method-check">✓</span><span><b>Data cleaning</b><br>Removed records with invalid negative asset values.</span></div>
                            <div class="method-item"><span class="method-check">✓</span><span><b>Preprocessing</b><br>Median imputation and standardization for numeric fields; one-hot encoding for categorical fields.</span></div>
                            <div class="method-item"><span class="method-check">✓</span><span><b>Evaluation</b><br>Stratified train/validation/test split with classification metrics and ROC-AUC.</span></div>
                            <div class="method-item"><span class="method-check">✓</span><span><b>Deployment</b><br>Interactive Gradio interface hosted as a Render web service.</span></div>
                          </div>
                        </div>
                        """)

            gr.HTML("""
            <div class="disclaimer-banner"><strong>Limitations & responsible use.</strong> These metrics describe the supplied dataset and held-out split only. They do not establish fairness, explainability, calibration, or expected performance at a real lender. Credit scoring can affect people materially; this academic app should not be used to approve or reject actual applications.</div>
            """)

    gr.HTML('<div class="footer"><span><b>LOAN APPROVAL INTELLIGENCE</b> &nbsp;·&nbsp; Machine learning portfolio project</span><span>Built with Python · scikit-learn · Gradio</span><span>Educational use only</span></div>')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, show_error=True)
