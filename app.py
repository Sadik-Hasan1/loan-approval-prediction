"""Loan Approval Intelligence — premium Emerald & Champagne Gradio dashboard.
Place app.py beside loan_approval_model.joblib in the repository root.
Academic demonstration only; not a real lending decision system.
"""
from __future__ import annotations
import html, logging, os
from pathlib import Path
from typing import Any
import gradio as gr
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("loan_approval_app")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_approval_model.joblib"
FEATURE_COLUMNS = ["no_of_dependents", "education", "self_employed", "income_annum", "loan_amount", "loan_term", "cibil_score", "residential_assets_value", "commercial_assets_value", "luxury_assets_value", "bank_asset_value"]
DEFAULT_VALUES = [2, "Graduate", "No", 5_000_000, 15_000_000, 10, 700, 5_000_000, 2_000_000, 8_000_000, 3_000_000]
TEST_METRICS = {"Accuracy": 98.90, "Precision": 98.75, "Recall": 99.49, "F1-score": 99.12, "ROC-AUC": 99.81}
FEATURE_IMPORTANCE = [("CIBIL score",81.81),("Loan term",5.68),("Loan amount",2.93),("Luxury assets",1.81),("Annual income",1.72),("Commercial assets",1.66),("Bank assets",1.54),("Residential assets",1.53),("Number of dependents",0.83),("Self-employed",0.26),("Education",0.21)]

def first_present(mapping: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        if mapping.get(key) is not None: return mapping[key]
    return None

if not MODEL_PATH.exists():
    raise FileNotFoundError("Required file loan_approval_model.joblib was not found. Upload it to the same repository folder as app.py.")
bundle = joblib.load(MODEL_PATH)
PIPELINE = MODEL = PREPROCESSOR = None
MODEL_NAME = "Random Forest"
if isinstance(bundle, dict):
    PIPELINE = first_present(bundle, ["pipeline", "model_pipeline", "full_pipeline"])
    MODEL = first_present(bundle, ["model", "best_model", "selected_model", "classifier", "estimator", "trained_model"])
    PREPROCESSOR = first_present(bundle, ["preprocessor", "preprocessing", "transformer", "feature_preprocessor"])
    MODEL_NAME = str(bundle.get("model_name", MODEL_NAME))
else:
    PIPELINE = bundle if hasattr(bundle, "predict") else None
if PIPELINE is None and (MODEL is None or PREPROCESSOR is None):
    raise ValueError("The saved artifact loaded, but its estimator/preprocessor could not be identified. Check how loan_approval_model.joblib was saved.")

def _result_card(state: str = "ready", message: str | None = None) -> str:
    if state == "error":
        return f'<section class="result-card result-error"><div class="eyebrow"><i class="dot"></i> ANALYSIS NEEDS ATTENTION</div><h3>Unable to complete analysis</h3><p>{html.escape(message or "Check the values and try again.")}</p></section>'
    return '<section class="result-card result-empty"><div class="eyebrow"><i class="dot"></i> PREDICTION WORKSPACE</div><div class="result-symbol">✳</div><h3>Your result starts here.</h3><p>Complete the applicant profile and run the model to see its predicted class and estimated approval probability.</p><div class="step-line"><span>01 · Applicant data</span><i></i><span>02 · Model output</span></div></section>'

def _probability_card(probability: float | None = None) -> str:
    shown = None if probability is None else max(0.0, min(100.0, float(probability)))
    value = "—" if shown is None else f"{shown:.2f}"
    width = "0" if shown is None else f"{shown:.2f}"
    foot = "Awaiting analysis" if shown is None else "Probability for the Approved class"
    muted = " muted" if shown is None else ""
    return f'<section class="glass-card probability-card"><div class="card-label-row"><span>Estimated approval probability</span><span class="tag">MODEL OUTPUT</span></div><div class="probability-value{muted}">{value}<small>%</small></div><div class="progress-track"><div class="progress-fill" style="width:{width}%"></div></div><div class="progress-caption"><span>{foot}</span><span>Not a guarantee</span></div></section>'

def _is_approved(value: Any) -> bool:
    return str(value).strip().lower() in {"approved", "1", "true"}

def predict_loan(no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value):
    values = [no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]
    if any(value is None for value in values):
        return _result_card("error", "Complete every field before running the analysis."), _probability_card()
    try:
        numeric_values = [no_of_dependents, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]
        if any(float(value) < 0 for value in numeric_values): return _result_card("error", "Numeric values cannot be negative."), _probability_card()
        if not 0 <= int(no_of_dependents) <= 5: return _result_card("error", "Number of dependents must be between 0 and 5."), _probability_card()
        if float(no_of_dependents) != int(no_of_dependents): return _result_card("error", "Number of dependents must be a whole number."), _probability_card()
        if not 300 <= float(cibil_score) <= 900: return _result_card("error", "CIBIL score must be between 300 and 900."), _probability_card()
        if float(cibil_score) != int(cibil_score): return _result_card("error", "CIBIL score must be a whole number."), _probability_card()
        if not 2 <= float(loan_term) <= 20: return _result_card("error", "Loan term must be between 2 and 20, as in the source dataset."), _probability_card()
        if float(loan_term) != int(loan_term): return _result_card("error", "Loan term must be a whole number."), _probability_card()
        if education not in {"Graduate", "Not Graduate"}: return _result_card("error", "Choose one of the education categories."), _probability_card()
        if self_employed not in {"Yes", "No"}: return _result_card("error", "Choose Yes or No for self-employment."), _probability_card()
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
        icon = "✓" if approved else "×"
        copy = "The model classified this sample as approved based on the supplied features." if approved else "The model classified this sample as rejected based on the supplied features."
        approval_probability = None
        if probabilities is not None and classes:
            idx = next((i for i, cls in enumerate(classes) if _is_approved(cls)), None)
            if idx is not None: approval_probability = float(probabilities[idx]) * 100
        safe_model = html.escape(MODEL_NAME)
        result = f'<section class="result-card {status_class}"><div class="eyebrow"><i class="dot"></i> MODEL PREDICTION <span class="tag model-tag">{safe_model}</span></div><div class="result-heading"><span class="result-status-icon">{icon}</span><span>{status}</span></div><p>{copy}</p><div class="result-divider"></div><div class="detail-row"><span>Decision class</span><strong>{status}</strong></div><div class="detail-row"><span>Model</span><strong>{safe_model}</strong></div><div class="disclaimer">Academic demonstration only. This output is not a real bank decision, financial advice, or a guarantee of approval.</div></section>'
        return result, _probability_card(approval_probability)
    except Exception:
        LOGGER.exception("Prediction failed")
        return _result_card("error", "The model could not process these inputs. Check the values or review the server logs."), _probability_card()

def apply_profile(profile: str):
    profiles = {
        "Balanced sample": [2,"Graduate","No",5_000_000,15_000_000,10,700,5_000_000,2_000_000,8_000_000,3_000_000],
        "Strong credit profile": [1,"Graduate","No",8_000_000,12_000_000,8,820,12_000_000,6_000_000,10_000_000,7_000_000],
        "Lower credit score": [3,"Not Graduate","Yes",2_000_000,18_000_000,18,480,1_000_000,500_000,1_500_000,200_000],
    }
    return [gr.update(value=value) for value in profiles[profile]]

def reset_form():
    return tuple(gr.update(value=value) for value in DEFAULT_VALUES) + (_result_card(), _probability_card())

def metric_card(value: str, label: str, note: str = "") -> str:
    v, l, n = map(html.escape, (value, label, note))
    return f'<div class="metric-tile glass-card"><div class="metric-label">{l}</div><div class="metric-value">{v}</div><div class="metric-note">{n}</div></div>'

def importance_bars() -> str:
    rows = []
    for label, value in FEATURE_IMPORTANCE:
        rows.append(f'<div class="importance-row"><div class="importance-head"><span>{html.escape(label)}</span><strong>{value:.2f}%</strong></div><div class="progress-track"><div class="importance-fill" style="width:{value:.2f}%"></div></div></div>')
    return '<div class="importance-list">' + ''.join(rows) + '</div>'

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#0b2118;--line:rgba(222,201,153,.19);--ink:#f6f3e9;--gold:#dec18a;--emerald:#70c69a}
html,body,.gradio-container{background:#0b2118!important;color:var(--ink)!important}body,.gradio-container{font-family:'DM Sans',sans-serif!important}.gradio-container{max-width:1580px!important;padding:18px 22px 32px!important;margin:auto!important}footer{display:none!important}
body:before{content:'';position:fixed;inset:0;pointer-events:none;background:radial-gradient(ellipse at 70% 0%,rgba(82,150,109,.17),transparent 35%),radial-gradient(ellipse at 0% 70%,rgba(207,176,109,.08),transparent 32%)}
.app-shell{gap:18px;align-items:flex-start}.sidebar{position:sticky;top:14px;min-height:calc(100vh - 50px);padding:20px 15px;border:1px solid var(--line);border-radius:23px;background:linear-gradient(155deg,rgba(29,63,48,.82),rgba(14,37,28,.88));backdrop-filter:blur(22px);box-shadow:0 20px 50px #0003,inset 0 1px #ffffff0d}
.brand{display:flex;align-items:center;gap:10px;padding:4px 5px 22px;border-bottom:1px solid #dccda51c}.brand-icon{width:38px;height:38px;display:grid;place-items:center;border-radius:13px;background:linear-gradient(145deg,#e4ca91,#a5c69d);color:#173a2a;font-size:20px;font-weight:900}.brand-name{font:800 11px Manrope,sans-serif;letter-spacing:.3px}.brand-sub{font-size:8px;color:#9eafa1;margin-top:4px;letter-spacing:1.2px}.nav-label{font-size:9px;color:#728f7b;letter-spacing:1.7px;font-weight:800;margin:24px 8px 10px}.nav-item{display:flex;align-items:center;gap:10px;padding:12px 10px;border-radius:11px;color:#9db3a3;font-size:10px;font-weight:700;margin:4px 0}.nav-item.active{color:#f5e6bf;background:linear-gradient(100deg,#cab17629,#649e7717);border:1px solid #dac2882b}.nav-icon{width:25px;height:25px;display:grid;place-items:center;border-radius:8px;background:#ffffff0c}.sidebar-bottom{margin-top:28px;padding:13px;border:1px solid #dac28826;border-radius:14px;background:#ffffff08}.sidebar-bottom b{font-size:9px;color:#dfc78f}.sidebar-bottom p{font-size:9px;color:#91a696;line-height:1.7;margin:7px 0 0}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:5px 3px 16px}.breadcrumb{font-size:9px;color:#7f9a87}.breadcrumb b{color:#e6d3a5}.top-status{display:flex;align-items:center;gap:7px;border:1px solid #71be9033;background:#50996d15;border-radius:999px;padding:8px 11px;color:#b9d8c1;font-size:8px;font-weight:800}.live-dot,.dot{width:6px;height:6px;background:#77d9a2;border-radius:50%;display:inline-block;box-shadow:0 0 9px #77d9a255}
.hero{position:relative;isolation:isolate;overflow:hidden;border-radius:24px;padding:28px 30px;margin-bottom:16px;background:linear-gradient(118deg,#183c2bf7,#1f4e36e8 58%,#646946c9);border:1px solid #dec9993d;box-shadow:0 24px 65px #0003,inset 0 1px #ffffff12}.hero:before{content:'';position:absolute;z-index:-1;width:350px;height:350px;right:-100px;top:-220px;border-radius:50%;border:1px solid #e5cf973d;box-shadow:0 0 0 34px #e5cf970a,0 0 0 70px #e5cf9707,0 0 80px #64c4901f}.hero-layout{display:flex;justify-content:space-between;align-items:center;gap:24px}.hero-copy{max-width:720px}.eyebrow{display:flex;align-items:center;gap:8px;color:#c1d4c1;text-transform:uppercase;letter-spacing:1.5px;font-size:8px;font-weight:800;margin-bottom:14px}.brand-mark{display:grid;place-items:center;width:27px;height:27px;border-radius:9px;background:#e3cc911f;border:1px solid #e3cc9147;color:#ead49c;font-size:14px}.hero h1{font:800 clamp(28px,3.2vw,42px)/1.08 Manrope,sans-serif!important;letter-spacing:-1.6px!important;color:#f8f4e9!important;margin:0 0 12px!important}.hero h1 em{font-style:normal;color:#e6cc91}.hero p{max-width:680px;color:#c0d1c1;font-size:11px;line-height:1.85;margin:0}.hero-chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:18px}.hero-chip{padding:7px 9px;border-radius:999px;border:1px solid #dfd7b82b;background:#ffffff0b;color:#e1e4d2;font-size:8px;font-weight:700}.hero-side{flex:0 0 175px;padding:16px;border:1px solid #e1cf9d3b;border-radius:17px;background:#ffffff12;backdrop-filter:blur(18px)}.hero-side-label{font-size:8px;color:#bdcbb7;font-weight:800;letter-spacing:1px;text-transform:uppercase}.hero-side-value{font:800 31px Manrope,sans-serif;color:#ead39b;letter-spacing:-1px;margin:8px 0 4px}.hero-side-caption{font-size:8px;line-height:1.7;color:#a9bda9}.hero-side-line{height:1px;background:#e4dab829;margin:12px 0}.hero-side-foot{font-size:8px;color:#d6e0ce;display:flex;justify-content:space-between;gap:6px}
.section-heading{display:flex;justify-content:space-between;align-items:end;gap:10px;margin:21px 2px 12px}.section-heading h2{font:800 16px Manrope,sans-serif!important;color:#f1efdf!important;letter-spacing:-.4px!important;margin:0!important}.section-heading p{font-size:9px;color:#8ea591;margin:5px 0 0}.tag{font-size:7px;letter-spacing:1px;color:#e2c88d;border:1px solid #dcc18a3d;background:#dcc18a12;padding:6px 8px;border-radius:999px;font-weight:900;white-space:nowrap}
.glass-card,.panel{background:linear-gradient(145deg,#214131bd,#10291ec7)!important;border:1px solid var(--line)!important;border-radius:17px!important;backdrop-filter:blur(20px);box-shadow:0 14px 35px #0002,inset 0 1px #ffffff0b!important}.panel{padding:17px!important;margin-bottom:12px!important}.panel-heading{display:flex;align-items:center;gap:10px;margin-bottom:5px}.panel-index{width:28px;height:28px;display:grid;place-items:center;border-radius:9px;background:#ddc2881a;border:1px solid #ddc28833;color:#e5cc91;font:800 9px Manrope,sans-serif}.panel-title{color:#f0efdf;font:800 11px Manrope,sans-serif}.panel-subtitle{font-size:9px;color:#8da592;line-height:1.7;margin:0 0 14px 38px}.preset-label{font-size:8px;color:#90a894;text-transform:uppercase;letter-spacing:1.1px;font-weight:800;margin:0 0 8px}
label,.wrap label,.label-wrap{color:#c3d0bd!important;font-size:9px!important;font-weight:700!important}input,textarea,.wrap input,.wrap textarea,.wrap select,select{background:#081d14b3!important;color:#f2f0e4!important;border:1px solid #dcc99930!important;border-radius:10px!important;min-height:40px!important}input::placeholder,textarea::placeholder{color:#6f8977!important}input:focus,textarea:focus,select:focus{border-color:#d9bd7e!important;box-shadow:0 0 0 2px #d9bd7e20!important}.gradio-container button[role=tab]{background:transparent!important;border:0!important;color:#8da590!important;font-size:10px!important;font-weight:800!important}.gradio-container button[role=tab][aria-selected=true]{color:#e4cc92!important;border-bottom:2px solid #d8bd83!important}.gradio-container .tabs{border-bottom:1px solid #dcc99920!important}
.preset-btn,.reset-btn{background:#e6ecdb0f!important;border:1px solid #dcc99930!important;color:#d5dfcf!important;border-radius:10px!important;min-height:37px!important;font-size:8px!important;font-weight:800!important}.preset-btn:hover,.reset-btn:hover{background:#dcc9991c!important;border-color:#dcc99959!important}.predict-btn{min-height:48px!important;border-radius:12px!important;background:linear-gradient(105deg,#c3a366,#ead49c)!important;color:#173827!important;border:0!important;font:800 11px Manrope,sans-serif!important;box-shadow:0 9px 25px #c4a66726!important;transition:transform .18s,box-shadow .18s!important}.predict-btn:hover{transform:translateY(-1px);box-shadow:0 12px 28px #c4a6673d!important}
.result-card{position:relative;overflow:hidden;min-height:220px;padding:19px;border:1px solid #dcc99933;border-radius:17px;background:linear-gradient(145deg,#234231d9,#0d251bd9)}.result-card:after{content:'';position:absolute;right:-48px;top:-62px;width:145px;height:145px;border:1px solid #dcc9991f;border-radius:50%;box-shadow:0 0 0 22px #dcc99906,0 0 0 44px #dcc99904;pointer-events:none}.result-approved{border-color:#71d29d59;background:linear-gradient(145deg,#1e5837a6,#0d251bd9)}.result-rejected,.result-error{border-color:#e28b7f52;background:linear-gradient(145deg,#59302770,#0d251bd9)}.result-card h3{position:relative;z-index:1;color:#f1efdf;font:800 20px/1.25 Manrope,sans-serif;margin:0 0 9px}.result-card p{position:relative;z-index:1;color:#a7bca8;font-size:10px;line-height:1.8;margin:10px 0 0}.result-symbol{display:grid;place-items:center;width:36px;height:36px;border-radius:12px;background:#ddc2881a;color:#e3ca91;font-size:20px;margin:0 0 13px}.step-line{display:flex;align-items:center;gap:9px;color:#8fa591;font-size:8px;margin-top:21px}.step-line i{height:1px;flex:1;background:linear-gradient(90deg,#d4bb81,#d4bb811a)}.result-heading{display:flex;align-items:center;gap:10px;color:#9be0b3;font:800 26px Manrope,sans-serif;position:relative;z-index:1}.result-rejected .result-heading{color:#f1a79a}.result-status-icon{width:35px;height:35px;display:grid;place-items:center;border-radius:11px;background:#6cd0931f;font-size:20px}.result-rejected .result-status-icon{background:#e884771f}.model-tag{margin-left:auto}.result-divider{height:1px;background:#dcc99921;margin:15px 0 11px}.detail-row{display:flex;justify-content:space-between;gap:10px;color:#8fa592;font-size:9px;margin:9px 0}.detail-row strong{color:#e3e8d7}.disclaimer{position:relative;z-index:1;color:#8b9e8d;font-size:8px;line-height:1.7;margin-top:14px;padding-top:11px;border-top:1px solid #dcc9991f}
.probability-card{padding:17px;margin-top:11px}.card-label-row{display:flex;justify-content:space-between;align-items:center;gap:8px;color:#c4d0bd;font-size:9px;font-weight:800}.probability-value{font:800 38px/1.2 Manrope,sans-serif;color:#e2c78c;letter-spacing:-1.4px;margin:12px 0}.probability-value small{font-size:16px;color:#c0a76e;margin-left:3px}.probability-value.muted{color:#718977}.progress-track{height:6px;border-radius:99px;overflow:hidden;background:#e1ebda1a}.progress-fill,.importance-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#4a9d75,#b8d5a7);transition:width .4s ease}.progress-caption{display:flex;justify-content:space-between;gap:8px;color:#819985;font-size:8px;margin-top:9px}.note-card{margin-top:11px;padding:12px 13px;border:1px solid #dcc99924;border-radius:12px;background:#ffffff08;color:#94a895;font-size:8px;line-height:1.8}.note-card strong{color:#e2c98f}.metric-tile{padding:15px;min-height:103px}.metric-label{color:#a9b9a6;font-size:9px;font-weight:800}.metric-value{font:800 24px Manrope,sans-serif;color:#e2c98f;letter-spacing:-.7px;margin-top:11px}.metric-note{color:#879d8b;font-size:8px;margin-top:4px}.importance-row{margin:0 0 12px}.importance-head{display:flex;justify-content:space-between;gap:10px;color:#b5c5b2;font-size:9px;margin-bottom:6px}.importance-head strong{color:#e3d3a8;font-size:9px}.importance-fill{background:linear-gradient(90deg,#4a9d75,#dfc58b)}.method-card{padding:17px;border-radius:14px;background:#ffffff06;border:1px solid #dcc99924}.method-title{font:800 12px Manrope,sans-serif;color:#f1efdf;margin-bottom:8px}.method-copy{color:#9db09e;font-size:9px;line-height:1.8;margin:0}.method-list{display:flex;flex-direction:column;gap:11px;margin-top:14px}.method-item{display:flex;gap:9px;color:#a8b9a8;font-size:9px;line-height:1.7}.method-check{flex:0 0 19px;width:19px;height:19px;display:grid;place-items:center;border-radius:7px;color:#e3cb91;background:#dcc18a1a}.disclaimer-banner{padding:12px 14px;margin-top:13px;border-radius:12px;border:1px solid #dcc18a33;background:#dcc18a0e;color:#b8b79b;font-size:8px;line-height:1.8}.disclaimer-banner strong{color:#e1c78f}.footer{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;color:#718b78;font-size:8px;padding:17px 3px 0}.footer b{color:#b4c3ae}
@media(max-width:1050px){.app-shell{gap:12px}.sidebar{padding:15px 10px}.hero{padding:22px}.hero-side{flex-basis:155px}.gradio-container{padding:12px!important}}@media(max-width:760px){.app-shell{display:block}.sidebar{position:relative;top:auto;min-height:0;margin-bottom:12px;padding:12px 14px}.brand{padding-bottom:10px}.nav-label,.nav-item,.sidebar-bottom{display:none}.hero-layout{align-items:flex-start;flex-direction:column}.hero-side{width:100%;box-sizing:border-box}.hero{padding:21px 18px}.topbar{padding:4px 2px 12px}.section-heading{align-items:flex-start}.panel{padding:13px!important}.result-card{min-height:0}.footer{flex-direction:column}.gradio-container{padding:10px!important}}
/* CrediSense polish + working sidebar navigation */
.sidebar .brand-icon{color:#173a2a;flex:0 0 38px}.sidebar .brand-icon svg{width:30px;height:30px;display:block}
.sidebar button.nav-item{appearance:none;-webkit-appearance:none;width:100%;font-family:inherit;text-align:left;cursor:pointer;background:transparent;border:1px solid transparent;line-height:1.2;transition:background .18s ease,border-color .18s ease,color .18s ease}
.sidebar button.nav-item:hover,.sidebar button.nav-item:focus-visible{color:#f5e6bf;background:#dcc28812;border-color:#dac2882b;outline:none}
.sidebar button.nav-item:focus-visible{box-shadow:0 0 0 2px #d8bd7940}
.sidebar .nav-icon{flex:0 0 25px}
.app-shell>.column{min-width:0}
.panel .row{align-items:stretch!important;gap:14px!important}
.panel .row>div{min-width:0!important}
.panel .form,.panel .wrap{min-width:0!important}
.input-note{margin-top:12px!important}
.metric-tile{height:100%;box-sizing:border-box}
.result-card,.probability-card,.panel,.hero-side,.metric-tile{box-sizing:border-box}
.gradio-container .block,.gradio-container .form{min-width:0}
.gradio-container button{transition:background .18s ease,border-color .18s ease,color .18s ease,transform .18s ease}
@media(max-width:1050px){.sidebar{padding:17px 11px}.brand{gap:8px}.brand-name{font-size:10px}.hero-layout{gap:16px}.hero-side{flex-basis:160px}}
@media(max-width:760px){.sidebar{padding:12px 13px;min-height:0}.brand{padding-bottom:12px}.sidebar .nav-label{display:none}.sidebar button.nav-item{display:inline-flex;width:auto;vertical-align:middle;white-space:nowrap;margin:10px 4px 0 0;padding:7px 9px;gap:7px}.sidebar .nav-icon{width:22px;height:22px;flex-basis:22px}.sidebar-bottom{display:block;margin-top:12px}.sidebar-bottom p{margin-top:4px}.panel .row{gap:9px!important}.hero h1{letter-spacing:-1px!important}.metric-tile{min-height:88px;padding:12px}.metric-value{font-size:21px}}
@media(max-width:430px){.sidebar button.nav-item{font-size:9px;padding:7px 8px}.sidebar .nav-icon{display:none}.section-heading{flex-wrap:wrap}.tag{white-space:normal}.panel-subtitle{margin-left:0}.hero{border-radius:18px}.hero-side{flex-basis:auto}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{transition:none!important;animation:none!important;scroll-behavior:auto!important}}
"""

THEME = gr.themes.Base(primary_hue=gr.themes.colors.emerald, secondary_hue=gr.themes.colors.amber, neutral_hue=gr.themes.colors.stone, radius_size="lg", spacing_size="lg")

with gr.Blocks(theme=THEME, css=CSS, title="CrediSense | Loan Approval Prediction", analytics_enabled=False) as demo:
    with gr.Row(elem_classes=["app-shell"]):
        with gr.Column(scale=0, min_width=185):
            gr.HTML('''<aside class="sidebar"><div class="brand"><div class="brand-icon"><svg viewBox="0 0 64 64" aria-hidden="true"><path d="M44 13A23 23 0 1 0 44 51" fill="none" stroke="currentColor" stroke-width="5" stroke-linecap="round"/><path d="M13 33h12l7-10 7 17 7-10h6" fill="none" stroke="#173a2a" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/><circle cx="52" cy="30" r="3" fill="#173a2a"/></svg></div><div><div class="brand-name">CrediSense</div><div class="brand-sub">LOAN APPROVAL PREDICTION</div></div></div><div class="nav-label">WORKSPACE</div><button type="button" class="nav-item active" onclick="window.scrollTo({top:0,behavior:'smooth'})"><span class="nav-icon">⌂</span><span>Overview</span></button><button type="button" class="nav-item" onclick="const t=[...document.querySelectorAll('button[role=tab]')].find(x=>x.innerText.includes('Applicant simulator'));if(t)t.click();window.scrollTo({top:0,behavior:'smooth'})"><span class="nav-icon">◉</span><span>Applicant simulator</span></button><button type="button" class="nav-item" onclick="const t=[...document.querySelectorAll('button[role=tab]')].find(x=>x.innerText.includes('Model insights'));if(t)t.click();setTimeout(()=>{const e=document.querySelector('.importance-list');if(e)e.scrollIntoView({behavior:'smooth',block:'center'});else window.scrollTo({top:0,behavior:'smooth'});},180)"><span class="nav-icon">▤</span><span>Model insights</span></button><div class="nav-label">PROJECT</div><button type="button" class="nav-item" onclick="const t=[...document.querySelectorAll('button[role=tab]')].find(x=>x.innerText.includes('Model insights'));if(t)t.click();setTimeout(()=>{const e=document.querySelector('.importance-list');if(e)e.scrollIntoView({behavior:'smooth',block:'center'});},180)"><span class="nav-icon">◇</span><span>Random Forest</span></button><div class="sidebar-bottom"><b>ACADEMIC DEMO</b><p>Explore a trained model with sample data. Outputs are not real lending decisions.</p></div></aside>''')
        with gr.Column(scale=6, min_width=300):
            gr.HTML(f'''<div class="topbar"><div class="breadcrumb">ML STUDIO&nbsp; / &nbsp;<b>LOAN APPROVAL</b></div><div class="top-status"><span class="live-dot"></span> MODEL LOADED · {html.escape(MODEL_NAME)}</div></div><header class="hero"><div class="hero-layout"><div class="hero-copy"><div class="eyebrow"><span class="brand-mark">C</span> CREDISENSE / MODEL LAB</div><h1>Loan Approval<br><em>Intelligence.</em></h1><p>Explore an interactive credit-risk classification workflow. Adjust applicant details, run the trained model, and review its output alongside the project's held-out evaluation metrics.</p><div class="hero-chips"><span class="hero-chip">✦ Random Forest</span><span class="hero-chip">◉ 11 input features</span><span class="hero-chip">⌁ Interactive analysis</span><span class="hero-chip">Academic portfolio project</span></div></div><aside class="hero-side"><div class="hero-side-label">Held-out test accuracy</div><div class="hero-side-value">{TEST_METRICS['Accuracy']:.2f}%</div><div class="hero-side-caption">Measured on this project's test split; not a guarantee of real-world lending performance.</div><div class="hero-side-line"></div><div class="hero-side-foot"><span><span class="live-dot"></span> Model status</span><span>Ready</span></div></aside></div></header><div class="section-heading"><div><h2>Dashboard overview</h2><p>Model evaluation and applicant prediction workspace.</p></div><span class="tag">DEMO ENVIRONMENT</span></div>''')
            with gr.Row():
                for value, label, note in [(f"{TEST_METRICS['Accuracy']:.2f}%","Accuracy","Overall correct predictions"),(f"{TEST_METRICS['Precision']:.2f}%","Precision","Correct approved predictions"),(f"{TEST_METRICS['Recall']:.2f}%","Recall","Approved cases identified"),(f"{TEST_METRICS['ROC-AUC']:.2f}%","ROC-AUC","Class separation metric")]:
                    gr.HTML(metric_card(value,label,note))
            with gr.Tabs():
                with gr.Tab("◈ Applicant simulator"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=7, min_width=290):
                            with gr.Group(elem_classes=["panel"]):
                                gr.HTML('<div class="panel-heading"><span class="panel-index">01</span><div class="panel-title">Quick scenarios</div></div><div class="panel-subtitle">Load a sample profile, then adjust values as needed.</div><div class="preset-label">Explore a sample</div>')
                                with gr.Row():
                                    balanced_btn = gr.Button("Balanced sample", elem_classes=["preset-btn"])
                                    strong_btn = gr.Button("Strong credit", elem_classes=["preset-btn"])
                                    low_score_btn = gr.Button("Lower score", elem_classes=["preset-btn"])
                                gr.HTML('<div class="note-card">Sample values are illustrative, not real applicants or financial guidance.</div>')
                            with gr.Group(elem_classes=["panel"]):
                                gr.HTML('<div class="panel-heading"><span class="panel-index">02</span><div class="panel-title">Applicant profile</div></div><div class="panel-subtitle">Personal and employment attributes</div>')
                                with gr.Row():
                                    dependents = gr.Number(label="Number of dependents", value=2, minimum=0, maximum=5, precision=0, info="Whole number · 0–5")
                                    education = gr.Dropdown(["Graduate","Not Graduate"], label="Education", value="Graduate", allow_custom_value=False)
                                with gr.Row():
                                    self_employed = gr.Dropdown(["Yes","No"], label="Self-employed", value="No", allow_custom_value=False)
                                    cibil_score = gr.Number(label="CIBIL score", value=700, minimum=300, maximum=900, precision=0, info="300–900")
                            with gr.Group(elem_classes=["panel"]):
                                gr.HTML('<div class="panel-heading"><span class="panel-index">03</span><div class="panel-title">Loan details</div></div><div class="panel-subtitle">Annual income and requested loan terms · source dataset units</div>')
                                with gr.Row():
                                    income_annum = gr.Number(label="Annual income", value=5_000_000, minimum=0, precision=0)
                                    loan_amount = gr.Number(label="Requested loan amount", value=15_000_000, minimum=0, precision=0)
                                with gr.Row():
                                    loan_term = gr.Number(label="Loan term", value=10, minimum=2, maximum=20, precision=0, info="Whole number · 2–20")
                                gr.HTML('<div class="note-card input-note"><strong>Input note</strong><br>Amounts pass directly to the saved pipeline. No currency conversion is performed.</div>')
                            with gr.Group(elem_classes=["panel"]):
                                gr.HTML('<div class="panel-heading"><span class="panel-index">04</span><div class="panel-title">Asset profile</div></div><div class="panel-subtitle">Declared assets used as model features</div>')
                                with gr.Row():
                                    residential_assets = gr.Number(label="Residential assets", value=5_000_000, minimum=0, precision=0)
                                    commercial_assets = gr.Number(label="Commercial assets", value=2_000_000, minimum=0, precision=0)
                                with gr.Row():
                                    luxury_assets = gr.Number(label="Luxury assets", value=8_000_000, minimum=0, precision=0)
                                    bank_assets = gr.Number(label="Bank assets", value=3_000_000, minimum=0, precision=0)
                            input_components = [dependents,education,self_employed,income_annum,loan_amount,loan_term,cibil_score,residential_assets,commercial_assets,luxury_assets,bank_assets]
                            with gr.Row():
                                predict_button = gr.Button("Run prediction →", variant="primary", elem_classes=["predict-btn"], scale=3)
                                reset_button = gr.Button("Reset inputs", variant="secondary", elem_classes=["reset-btn"], scale=1)
                        with gr.Column(scale=5, min_width=260):
                            gr.HTML('<div class="section-heading" style="margin-top:0"><div><h2>Prediction summary</h2><p>Model output appears after analysis.</p></div><span class="tag">LIVE OUTPUT</span></div>')
                            result_output = gr.HTML(value=_result_card())
                            probability_output = gr.HTML(value=_probability_card())
                            gr.HTML('<div class="note-card"><strong>Interpretation:</strong> the class is the model-predicted label. The percentage is the estimated probability for the Approved class; it may not be calibrated and does not guarantee a real lending outcome.</div>')
                    predict_button.click(fn=predict_loan, inputs=input_components, outputs=[result_output,probability_output], api_name="predict_loan", show_progress="minimal")
                    reset_button.click(fn=reset_form, inputs=None, outputs=input_components+[result_output,probability_output], api_name="reset_form")
                    for button, profile in [(balanced_btn,"Balanced sample"),(strong_btn,"Strong credit profile"),(low_score_btn,"Lower credit score")]:
                        button.click(fn=lambda p=profile: apply_profile(p), inputs=None, outputs=input_components, api_name=f"load_{profile.lower().replace(' ','_')}")
                with gr.Tab("▤ Model insights"):
                    gr.HTML('<div class="section-heading"><div><h2>Evaluation snapshot</h2><p>Reported metrics from the held-out test split.</p></div><span class="tag">637 TEST ROWS</span></div>')
                    with gr.Row():
                        gr.HTML(metric_card(f"{TEST_METRICS['F1-score']:.2f}%","F1-score","Balance of precision and recall"))
                        gr.HTML(metric_card(f"{TEST_METRICS['ROC-AUC']:.2f}%","ROC-AUC","Class separation metric"))
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=6, min_width=260):
                            with gr.Group(elem_classes=["panel"]):
                                gr.HTML('<div class="panel-heading"><span class="panel-index">A</span><div class="panel-title">Global feature importance</div></div><div class="panel-subtitle">Reported from the Random Forest analysis. Importance reflects split-based contribution, not causation.</div>')
                                gr.HTML(importance_bars())
                        with gr.Column(scale=6, min_width=260):
                            with gr.Group(elem_classes=["panel"]):
                                gr.HTML('<div class="panel-heading"><span class="panel-index">B</span><div class="panel-title">Methodology at a glance</div></div><div class="panel-subtitle">Supervised-learning workflow</div><div class="method-card"><div class="method-title">Model pipeline</div><p class="method-copy">The project compares Logistic Regression, Decision Tree, and Random Forest; Random Forest is selected for the final held-out test evaluation.</p><div class="method-list"><div class="method-item"><span class="method-check">✓</span><span><b>Data cleaning</b><br>Removed records with invalid negative asset values.</span></div><div class="method-item"><span class="method-check">✓</span><span><b>Preprocessing</b><br>Median imputation and standardization for numeric fields; one-hot encoding for categorical fields.</span></div><div class="method-item"><span class="method-check">✓</span><span><b>Evaluation</b><br>Stratified train/validation/test split with classification metrics and ROC-AUC.</span></div><div class="method-item"><span class="method-check">✓</span><span><b>Deployment</b><br>Interactive Gradio interface hosted as a Render web service.</span></div></div></div>')
                    gr.HTML('<div class="disclaimer-banner"><strong>Limitations & responsible use.</strong> Metrics describe this dataset and held-out split only. They do not establish fairness, explainability, calibration, or performance at a real lender. This academic app should not be used to approve or reject actual applications.</div>')
            gr.HTML('<div class="footer"><span><b>CREDISENSE</b> · Loan approval prediction · ML portfolio project</span><span>Python · scikit-learn · Gradio</span><span>Educational use only</span></div>')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, show_error=True)
