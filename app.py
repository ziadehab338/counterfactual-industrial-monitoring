from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional, Tuple
import json
import html

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

try:
    import dice_ml
    DICE_AVAILABLE = True
except Exception:
    DICE_AVAILABLE = False

try:
    from alibi.explainers import AnchorTabular
    ANCHOR_AVAILABLE = True
except Exception:
    ANCHOR_AVAILABLE = False


st.set_page_config(
    page_title="Machine Failure Prediction System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FAILURE_TYPE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


# ==========================================
# UI Styling
# ==========================================
def inject_global_css() -> None:
    st.markdown(
        dedent(
            """
            <style>
                .stApp {
                    background:
                        radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 26%),
                        radial-gradient(circle at top right, rgba(20,184,166,0.15), transparent 22%),
                        linear-gradient(135deg, #05111f 0%, #0a1730 44%, #08121e 100%);
                    color: #eef6ff;
                }
                [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
                    background: transparent !important;
                }
                section[data-testid="stSidebar"] { display: none !important; }
                .block-container {
                    max-width: 1450px;
                    padding-top: 1rem;
                    padding-bottom: 2rem;
                }
                .section-title {
                    font-size: 1.34rem;
                    font-weight: 800;
                    margin-bottom: 0.45rem;
                    color: #ffffff;
                }
                .section-subtitle {
                    color: #bfdbfe;
                    margin-bottom: 1rem;
                    line-height: 1.6;
                }
                .glass-card {
                    background: rgba(10, 20, 40, 0.64);
                    border: 1px solid rgba(255,255,255,0.09);
                    border-radius: 26px;
                    padding: 1.2rem 1.2rem;
                    box-shadow: 0 22px 60px rgba(0,0,0,0.28);
                    backdrop-filter: blur(16px);
                }
                .metric-card {
                    background: linear-gradient(180deg, rgba(12, 23, 45, 0.92), rgba(7, 14, 28, 0.82));
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 24px;
                    padding: 1rem 1rem;
                    min-height: 118px;
                    box-shadow: 0 14px 35px rgba(0,0,0,0.22);
                }
                .metric-label {
                    color: #cbd5e1;
                    font-size: 0.88rem;
                    margin-bottom: 0.3rem;
                }
                .metric-value {
                    font-size: 1.65rem;
                    font-weight: 900;
                    color: #ffffff;
                    line-height: 1.15;
                }
                .metric-caption {
                    color: #93c5fd;
                    font-size: 0.84rem;
                    margin-top: 0.18rem;
                }
                .risk-banner {
                    border-radius: 26px;
                    padding: 1.15rem 1.15rem;
                    border: 1px solid rgba(255,255,255,0.08);
                    margin-bottom: 0.95rem;
                    box-shadow: 0 16px 38px rgba(0,0,0,0.18);
                }
                .risk-low { background: linear-gradient(135deg, rgba(16,185,129,0.25), rgba(8,47,73,0.35)); }
                .risk-medium { background: linear-gradient(135deg, rgba(245,158,11,0.28), rgba(73,44,8,0.30)); }
                .risk-high { background: linear-gradient(135deg, rgba(239,68,68,0.28), rgba(69,10,10,0.30)); }
                .risk-title { font-size: 1.16rem; font-weight: 900; margin-bottom: 0.22rem; }
                .risk-text, .small-note { color: #dde7f3; line-height: 1.65; font-size: 0.95rem; }
                .pill {
                    display: inline-block;
                    padding: 0.4rem 0.76rem;
                    border-radius: 999px;
                    font-size: 0.8rem;
                    font-weight: 800;
                    margin-right: 0.45rem;
                    margin-bottom: 0.45rem;
                    border: 1px solid rgba(255,255,255,0.08);
                }
                .pill-blue { background: rgba(59,130,246,0.18); color: #dbeafe; }
                .pill-green { background: rgba(16,185,129,0.18); color: #d1fae5; }
                .pill-amber { background: rgba(245,158,11,0.18); color: #fef3c7; }
                .pill-red { background: rgba(239,68,68,0.18); color: #fee2e2; }
                .pill-slate { background: rgba(148,163,184,0.18); color: #e2e8f0; }
                div[data-testid="stForm"] {
                    background: rgba(10, 20, 40, 0.66);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 26px;
                    padding: 0.75rem 1rem 1rem 1rem;
                    box-shadow: 0 22px 60px rgba(0,0,0,0.28);
                }
                .stButton > button,
                div[data-testid="stDownloadButton"] > button {
                    border-radius: 16px !important;
                    border: 1px solid rgba(255,255,255,0.12) !important;
                    background: linear-gradient(135deg, #2563eb, #14b8a6) !important;
                    color: white !important;
                    font-weight: 800 !important;
                    padding: 0.68rem 1rem !important;
                    box-shadow: 0 12px 28px rgba(37,99,235,0.28) !important;
                }
                .cover-shell {
                    position: relative;
                    border-radius: 34px;
                    padding: 2.1rem;
                    min-height: 760px;
                    overflow: hidden;
                    background:
                        radial-gradient(circle at 15% 10%, rgba(56,189,248,0.22), transparent 22%),
                        radial-gradient(circle at 85% 12%, rgba(20,184,166,0.18), transparent 20%),
                        linear-gradient(135deg, #051121 0%, #0a1d36 45%, #07111f 100%);
                    border: 1px solid rgba(255,255,255,0.08);
                    box-shadow: 0 28px 80px rgba(0,0,0,.34);
                    transition: all .35s ease;
                }
                .cover-grid {
                    position: relative;
                    z-index: 2;
                    display: grid;
                    grid-template-columns: 1.15fr 0.95fr;
                    gap: 2rem;
                    align-items: center;
                    min-height: 620px;
                }
                .cover-title {
                    font-size: 4.15rem;
                    line-height: 0.98;
                    font-weight: 900;
                    letter-spacing: -1.8px;
                    color: #ffffff;
                    margin-bottom: 1rem;
                    max-width: 660px;
                }
                .cover-copy {
                    font-size: 1.1rem;
                    color: #d7e7ff;
                    line-height: 1.85;
                    max-width: 700px;
                }
                .cover-pills { margin-top: 1.2rem; display: flex; flex-wrap: wrap; gap: 0.65rem; }
                .cover-pill {
                    padding: 0.66rem 0.9rem;
                    border-radius: 999px;
                    background: rgba(255,255,255,0.08);
                    border: 1px solid rgba(255,255,255,0.08);
                    font-weight: 800;
                    font-size: 0.82rem;
                    color: #e5efff;
                }
                .home-stack { display: grid; gap: 1rem; }
                .home-card {
                    border-radius: 28px;
                    padding: 1.45rem 1.35rem;
                    background: linear-gradient(180deg, rgba(11,20,39,0.90), rgba(5,10,20,0.80));
                    border: 1px solid rgba(255,255,255,0.08);
                    box-shadow: 0 18px 50px rgba(0,0,0,0.28);
                    transition: transform .28s ease, box-shadow .28s ease, border-color .28s ease;
                    cursor: pointer;
                }
                .home-card:hover { transform: translateY(-6px) scale(1.015); border-color: rgba(125,211,252,.35); }
                .mode-link { text-decoration: none !important; color: inherit !important; display: block; }
                .home-top { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
                .home-title { font-size: 1.82rem; font-weight: 900; color: #ffffff; }
                .home-tag {
                    font-size: 0.76rem;
                    font-weight: 800;
                    padding: 0.5rem 0.8rem;
                    border-radius: 999px;
                    color: #dbeafe;
                    background: rgba(59,130,246,0.14);
                    border: 1px solid rgba(186,230,253,.16);
                }
                .home-text { margin-top: 0.95rem; color: #d0dcee; line-height: 1.8; }
                .subtle-label {
                    color: #93c5fd;
                    font-size: 0.85rem;
                    letter-spacing: 0.08em;
                    font-weight: 800;
                    text-transform: uppercase;
                    margin-bottom: 0.8rem;
                }
                .footer-note { text-align: center; color: #9bb0cb; margin-top: 1rem; font-size: 0.88rem; }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


# ==========================================
# Routing and basic utilities
# ==========================================
def get_route() -> str:
    try:
        page = st.query_params.get("page", None)
        if isinstance(page, list):
            page = page[0]
        if page in ["home", "user", "developer"]:
            st.session_state["route"] = str(page)
            return str(page)
    except Exception:
        pass
    return str(st.session_state.get("route", "home"))


def set_route(page: str) -> None:
    st.session_state["route"] = page
    try:
        st.query_params["page"] = page
    except Exception:
        pass


def save_json(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def read_json(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return default


def locate_dataset_path() -> Path:
    candidates = [
        BASE_DIR / "data" / "ai4i2020.csv",
        BASE_DIR / "ai4i2020.csv",
        BASE_DIR.parent / "data" / "ai4i2020.csv",
        BASE_DIR.parent / "ai4i2020.csv",
        Path.cwd() / "data" / "ai4i2020.csv",
        Path.cwd() / "ai4i2020.csv",
        Path.cwd().parent / "data" / "ai4i2020.csv",
        Path.cwd().parent / "ai4i2020.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for root in [BASE_DIR, Path.cwd(), BASE_DIR.parent, Path.cwd().parent]:
        try:
            matches = list(root.rglob("ai4i2020.csv"))
            if matches:
                return matches[0]
        except Exception:
            continue
    raise FileNotFoundError(
        "Dataset file 'ai4i2020.csv' was not found. Put it in a folder named 'data' next to app.py or in the project root."
    )


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    path = locate_dataset_path()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_dataset_path_text() -> str:
    return str(locate_dataset_path())


def get_failure_type(row: pd.Series) -> str:
    if int(row["Machine failure"]) == 0:
        return "No Failure"
    if int(row["TWF"]) == 1:
        return "Tool Wear Failure"
    if int(row["HDF"]) == 1:
        return "Heat Dissipation Failure"
    if int(row["PWF"]) == 1:
        return "Power Failure"
    if int(row["OSF"]) == 1:
        return "Overstrain Failure"
    if int(row["RNF"]) == 1:
        return "Random Failure"
    return "Unknown"


# ==========================================
# Model training / loading
# ==========================================
@st.cache_resource(show_spinner=False)
def load_or_train_artifacts():
    binary_path = ARTIFACTS_DIR / "binary_model.joblib"
    failure_type_path = ARTIFACTS_DIR / "failure_type_model.joblib"
    label_encoder_path = ARTIFACTS_DIR / "label_encoder.joblib"
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    metadata_path = ARTIFACTS_DIR / "metadata.json"

    if binary_path.exists() and failure_type_path.exists() and label_encoder_path.exists():
        binary_model = joblib.load(binary_path)
        failure_type_model = joblib.load(failure_type_path)
        label_encoder = joblib.load(label_encoder_path)
        metrics = read_json(metrics_path, {})
        metadata = read_json(metadata_path, {})
        return binary_model, failure_type_model, label_encoder, metrics, metadata

    dataset = load_dataset().copy()
    X = dataset[FEATURES].copy()
    y = dataset["Machine failure"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=1, stratify=y
    )

    binary_preprocessor = ColumnTransformer(
        transformers=[("type_encoder", OneHotEncoder(drop="first", handle_unknown="ignore"), ["Type"])],
        remainder="passthrough",
    )

    binary_model = Pipeline(
        steps=[
            ("preprocessor", binary_preprocessor),
            ("imputer", SimpleImputer(strategy="mean")),
            ("classifier", LogisticRegression(max_iter=2000, class_weight={0: 1, 1: 5})),
        ]
    )
    binary_model.fit(X_train, y_train)

    y_train_pred = binary_model.predict(X_train)
    y_test_pred = binary_model.predict(X_test)

    binary_metrics = {
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_test_pred,
            target_names=["No Failure", "Failure"],
            output_dict=True,
            zero_division=0,
        ),
    }

    dataset["Failure Type"] = dataset.apply(get_failure_type, axis=1)
    failure_data = dataset[dataset["Machine failure"] == 1].copy()
    X_type = failure_data[FEATURES].copy()
    y_type = failure_data["Failure Type"].copy()

    label_encoder = LabelEncoder()
    y_type_encoded = label_encoder.fit_transform(y_type)

    X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(
        X_type,
        y_type_encoded,
        test_size=0.20,
        random_state=1,
        stratify=y_type_encoded,
    )

    failure_type_preprocessor = ColumnTransformer(
        transformers=[("type_encoder", OneHotEncoder(drop="first", handle_unknown="ignore"), ["Type"])],
        remainder="passthrough",
    )

    failure_type_model = Pipeline(
        steps=[
            ("preprocessor", failure_type_preprocessor),
            ("imputer", SimpleImputer(strategy="mean")),
            (
                "classifier",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=len(label_encoder.classes_),
                    n_estimators=250,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=1,
                    eval_metric="mlogloss",
                ),
            ),
        ]
    )
    failure_type_model.fit(X_train_type, y_train_type)

    y_train_type_pred = failure_type_model.predict(X_train_type)
    y_test_type_pred = failure_type_model.predict(X_test_type)

    failure_type_metrics = {
        "train_accuracy": float(accuracy_score(y_train_type, y_train_type_pred)),
        "test_accuracy": float(accuracy_score(y_test_type, y_test_type_pred)),
        "confusion_matrix": confusion_matrix(y_test_type, y_test_type_pred).tolist(),
        "classification_report": classification_report(
            y_test_type,
            y_test_type_pred,
            target_names=list(label_encoder.classes_),
            output_dict=True,
            zero_division=0,
        ),
        "classes": list(label_encoder.classes_),
    }

    metrics = {"binary": binary_metrics, "failure_type": failure_type_metrics}
    metadata = {"features": FEATURES, "numeric_features": NUMERIC_FEATURES}

    try:
        joblib.dump(binary_model, binary_path)
        joblib.dump(failure_type_model, failure_type_path)
        joblib.dump(label_encoder, label_encoder_path)
        save_json(metrics_path, metrics)
        save_json(metadata_path, metadata)
    except Exception:
        pass

    return binary_model, failure_type_model, label_encoder, metrics, metadata


# ==========================================
# DiCE and Anchor explainers
# ==========================================
@st.cache_resource(show_spinner=False)
def build_dice_explainers():
    if not DICE_AVAILABLE:
        return None

    dataset = load_dataset().copy()
    binary_model, _, _, _, _ = load_or_train_artifacts()

    data_dice = dice_ml.Data(
        dataframe=dataset[FEATURES + ["Machine failure"]].copy(),
        continuous_features=NUMERIC_FEATURES,
        categorical_features=["Type"],
        outcome_name="Machine failure",
    )
    model_dice = dice_ml.Model(model=binary_model, backend="sklearn")

    explainers = {
        "Genetic": dice_ml.Dice(data_dice, model_dice, method="genetic"),
        "Random": dice_ml.Dice(data_dice, model_dice, method="random"),
        "KDTree": dice_ml.Dice(data_dice, model_dice, method="kdtree"),
    }
    return explainers


@st.cache_resource(show_spinner=False)
def build_anchor_explainer():
    if not ANCHOR_AVAILABLE:
        return None

    dataset = load_dataset().copy()
    binary_model, _, _, _, _ = load_or_train_artifacts()

    X = dataset[FEATURES].copy()
    y = dataset["Machine failure"].astype(int).values

    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=1, stratify=y
    )

    preprocessor = binary_model.named_steps["preprocessor"]
    imputer = binary_model.named_steps["imputer"]
    classifier = binary_model.named_steps["classifier"]

    X_train_processed = preprocessor.transform(X_train)
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    X_train_processed = imputer.transform(X_train_processed)

    feature_names = list(preprocessor.get_feature_names_out())

    def anchor_predict_fn(x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return classifier.predict(x)

    anchor_explainer = AnchorTabular(
        predictor=anchor_predict_fn,
        feature_names=feature_names,
    )

    # More detailed discretization makes numerical ranges more precise.
    anchor_explainer.fit(
        X_train_processed,
        disc_perc=(5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95),
    )

    return {
        "explainer": anchor_explainer,
        "preprocessor": preprocessor,
        "imputer": imputer,
        "classifier": classifier,
    }


def explain_anchor_for_input(input_df: pd.DataFrame, threshold: float = 0.95) -> Dict:
    if not ANCHOR_AVAILABLE:
        return {"available": False, "message": "Anchor explanation requires the alibi package."}

    anchor_bundle = build_anchor_explainer()
    if anchor_bundle is None:
        return {"available": False, "message": "Anchor explainer could not be initialized."}

    preprocessor = anchor_bundle["preprocessor"]
    imputer = anchor_bundle["imputer"]
    classifier = anchor_bundle["classifier"]
    anchor_explainer = anchor_bundle["explainer"]

    input_processed = preprocessor.transform(input_df)
    if hasattr(input_processed, "toarray"):
        input_processed = input_processed.toarray()
    input_processed = imputer.transform(input_processed)

    prediction = int(classifier.predict(input_processed)[0])
    probability = float(classifier.predict_proba(input_processed)[0][1])

    try:
        explanation = anchor_explainer.explain(input_processed[0], threshold=threshold)
        precision_value = float(np.asarray(explanation.precision).item())
        coverage_value = float(np.asarray(explanation.coverage).item())
        anchor_rules = list(explanation.anchor)
        message = ""
    except Exception as exc:
        precision_value = 0.0
        coverage_value = 0.0
        anchor_rules = []
        message = str(exc)

    return {
        "available": True,
        "prediction": prediction,
        "probability": probability,
        "anchor": anchor_rules,
        "precision": precision_value,
        "coverage": coverage_value,
        "message": message,
    }


def render_anchor_explanation(anchor_info: Dict, show_metrics: bool = False) -> None:
    st.markdown(
        '<div class="section-title">Why the Model Made This Prediction</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">This rule-based explanation highlights the operating conditions that support the current model decision.</div>',
        unsafe_allow_html=True,
    )

    if not anchor_info.get("available", False):
        st.warning(anchor_info.get("message", "Anchor explanation is not available."))
        return

    anchor_rules = anchor_info.get("anchor", [])
    if len(anchor_rules) == 0:
        rule_text = """
        <div class="small-note">
            No specific feature-based anchor rule was required for this input.
            The model decision is already stable for the selected operating condition.
        </div>
        """
    else:
        rule_items = "".join(["<li>{}</li>".format(rule) for rule in anchor_rules])
        rule_text = """
        <div class="small-note">
            The model decision is mainly supported by the following operating conditions:
            <ul>{}</ul>
        </div>
        """.format(rule_items)

    metrics_html = ""
    if show_metrics:
        metrics_html = """
        <div style="margin-top: 1rem;">
            <span class="pill pill-blue">Precision: {:.4f}</span>
            <span class="pill pill-green">Coverage: {:.4f}</span>
        </div>
        """.format(anchor_info.get("precision", 0.0), anchor_info.get("coverage", 0.0))

    st.markdown(
        """
        <div class="glass-card">
            {}
            {}
        </div>
        """.format(rule_text, metrics_html),
        unsafe_allow_html=True,
    )


# ==========================================
# Helper functions
# ==========================================
def get_feature_ranges(df: pd.DataFrame) -> Dict[str, List[float]]:
    return {column: [float(df[column].min()), float(df[column].max())] for column in NUMERIC_FEATURES}


def get_counterfactual_feature_ranges(df: pd.DataFrame, input_df: pd.DataFrame) -> Dict[str, List[float]]:
    """
    Build permitted ranges for counterfactual generation.

    All numeric features can vary across their dataset range, except Tool wear.
    Tool wear represents accumulated usage, so the counterfactual generator is
    allowed to keep it unchanged or reduce it, but it is not allowed to increase it.
    """
    permitted_range = get_feature_ranges(df)

    tool_wear_feature = "Tool wear [min]"
    if tool_wear_feature in permitted_range and tool_wear_feature in input_df.columns:
        dataset_min = float(df[tool_wear_feature].min())
        dataset_max = float(df[tool_wear_feature].max())
        current_tool_wear = float(input_df.iloc[0][tool_wear_feature])

        upper_bound = min(current_tool_wear, dataset_max)
        lower_bound = min(dataset_min, upper_bound)
        permitted_range[tool_wear_feature] = [lower_bound, upper_bound]

    return permitted_range


def infer_risk(probability: float) -> Tuple[str, str, str]:
    if probability < 0.30:
        return (
            "Low Risk",
            "risk-low",
            "Safe operating side. No immediate machine failure is predicted. Keep monitoring the machine under the current conditions.",
        )
    if probability < 0.70:
        return (
            "Medium Risk",
            "risk-medium",
            "Be careful. The machine is still operating, but the current pattern indicates elevated risk. Preventive adjustment is recommended.",
        )
    return (
        "High Risk",
        "risk-high",
        "Critical alert. The current operating conditions are highly likely to lead to machine failure. Immediate corrective action is strongly recommended.",
    )


def get_failure_type_label(prediction, label_encoder) -> str:
    return str(label_encoder.inverse_transform([int(prediction)])[0])


def recommendation_text(risk_level: str, failure_type: Optional[str]) -> str:
    if risk_level == "Low Risk":
        return "You are on the safe side. Maintain the current operating range and continue routine monitoring."
    if risk_level == "Medium Risk":
        return "Be careful. Inspect temperature, rotational speed, torque, and tool wear before the next operating cycle."
    if failure_type:
        return (
            "Immediate action is recommended. The model indicates a likely failure event associated with "
            + failure_type
            + ". Apply corrective changes and verify the machine before continuing operation."
        )
    return "Immediate action is recommended. Apply corrective changes and verify the machine before continuing operation."


def metric_card(title: str, value: str, caption: str = "") -> None:
    st.markdown(
        dedent(
            """
            <div class="metric-card">
                <div class="metric-label">{}</div>
                <div class="metric-value">{}</div>
                <div class="metric-caption">{}</div>
            </div>
            """.format(title, value, caption)
        ),
        unsafe_allow_html=True,
    )


def risk_banner(title: str, css_class: str, message: str) -> None:
    st.markdown(
        dedent(
            """
            <div class="risk-banner {}">
                <div class="risk-title">{}</div>
                <div class="risk-text">{}</div>
            </div>
            """.format(css_class, title, message)
        ),
        unsafe_allow_html=True,
    )


def build_probability_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "white"}},
            title={"text": "Failure Probability", "font": {"size": 20, "color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#cbd5e1"},
                "bar": {"color": "#38bdf8"},
                "bgcolor": "rgba(255,255,255,0.03)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(16,185,129,0.35)"},
                    {"range": [30, 70], "color": "rgba(245,158,11,0.32)"},
                    {"range": [70, 100], "color": "rgba(239,68,68,0.32)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 4},
                    "thickness": 0.8,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=55, b=20), paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    return fig


def build_comparison_chart(original_row: pd.Series, cf_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=NUMERIC_FEATURES, y=[float(original_row[f]) for f in NUMERIC_FEATURES], name="Original"))
    for index, (_, row) in enumerate(cf_df.iterrows(), start=1):
        fig.add_trace(go.Bar(x=NUMERIC_FEATURES, y=[float(row[f]) for f in NUMERIC_FEATURES], name="Solution {}".format(index)))
    fig.update_layout(
        barmode="group",
        height=430,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(title="Feature", tickangle=-15),
        yaxis=dict(title="Value"),
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(l=20, r=20, t=65, b=45),
    )
    return fig


def build_delta_chart(original_row: pd.Series, cf_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for index, (_, row) in enumerate(cf_df.iterrows(), start=1):
        deltas = [float(row[f]) - float(original_row[f]) for f in NUMERIC_FEATURES]
        fig.add_trace(go.Bar(x=NUMERIC_FEATURES, y=deltas, name="Solution {}".format(index)))
    fig.update_layout(
        barmode="group",
        height=390,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(title="Feature", tickangle=-15),
        yaxis=dict(title="Change from Original"),
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(l=20, r=20, t=65, b=45),
        shapes=[dict(type="line", x0=-0.5, x1=len(NUMERIC_FEATURES) - 0.5, y0=0, y1=0, line=dict(color="rgba(255,255,255,0.32)", width=1))],
    )
    return fig


def build_recommendations(original_row: pd.Series, cf_row: pd.Series) -> List[str]:
    recommendations = []
    for column in NUMERIC_FEATURES:
        old_value = float(original_row[column])
        new_value = float(cf_row[column])
        if old_value != new_value:
            direction = "Increase" if new_value > old_value else "Decrease"
            recommendations.append("{} {} from {:.2f} to {:.2f}".format(direction, column, old_value, new_value))
    return recommendations


def calculate_solution_cost(original_row: pd.Series, cf_row: pd.Series, cost_weights: Dict[str, float], dataset: pd.DataFrame) -> float:
    total_cost = 0.0
    for feature in NUMERIC_FEATURES:
        old_value = float(original_row[feature])
        new_value = float(cf_row[feature])
        feature_range = float(dataset[feature].max() - dataset[feature].min())
        normalized_change = 0.0 if feature_range == 0 else abs(new_value - old_value) / feature_range
        total_cost += normalized_change * float(cost_weights.get(feature, 1.0))
    return total_cost


def generate_counterfactuals_for_method(method_name: str, input_df: pd.DataFrame, dataset: pd.DataFrame, desired_class, total_cfs: int = 2):
    """
    Generate counterfactuals for one method.

    Important behavior:
    - The UI should display up to `total_cfs` valid solutions.
    - Tool wear is allowed to stay the same or decrease only.
    - Because invalid tool-wear-increase solutions may be removed after generation,
      the function internally asks DiCE for more than 2 solutions, then keeps the
      first valid 2 solutions after filtering.
    - KDTree can be stricter than Random/Genetic, so it gets a larger internal request.
    """
    if not DICE_AVAILABLE:
        return None, "DiCE is not installed in this environment. Install dice-ml to activate counterfactual analysis."

    explainers = build_dice_explainers()
    if explainers is None:
        return None, "Counterfactual explainers could not be initialized."

    tool_wear_feature = "Tool wear [min]"
    explainer = explainers[method_name]

    # Ask for more candidates than we finally display.
    # This is necessary because some candidates may be removed by the Tool wear constraint.
    if method_name == "KDTree":
        internal_total_cfs = max(total_cfs * 10, 20)
    elif method_name == "Random":
        internal_total_cfs = max(total_cfs * 6, 12)
    else:
        internal_total_cfs = max(total_cfs * 5, 10)

    def keep_valid_tool_wear_rows(raw_cf_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if raw_cf_df is None or raw_cf_df.empty:
            return pd.DataFrame()

        filtered_df = raw_cf_df.copy()

        if tool_wear_feature in filtered_df.columns and tool_wear_feature in input_df.columns:
            current_tool_wear = float(input_df.iloc[0][tool_wear_feature])
            filtered_df = filtered_df[
                filtered_df[tool_wear_feature].astype(float) <= current_tool_wear + 1e-9
            ].copy()

        # Remove duplicate solutions after filtering, then show only the requested number.
        filtered_df = filtered_df.drop_duplicates().head(total_cfs).copy()
        return filtered_df

    try:
        kwargs = {
            "query_instances": input_df,
            "total_CFs": internal_total_cfs,
            "desired_class": desired_class,
            "features_to_vary": NUMERIC_FEATURES,
            "permitted_range": get_counterfactual_feature_ranges(dataset, input_df),
        }

        if method_name == "Random":
            kwargs["sample_size"] = 20000

        counterfactuals = explainer.generate_counterfactuals(**kwargs)
        cf_df = counterfactuals.cf_examples_list[0].final_cfs_df
        valid_cf_df = keep_valid_tool_wear_rows(cf_df)

        if not valid_cf_df.empty:
            return valid_cf_df, None

        # KDTree fallback: if KDTree cannot produce enough valid rows when Tool wear is variable,
        # retry with Tool wear fixed and allow the other operating parameters to change.
        if method_name == "KDTree":
            fallback_features = [feature for feature in NUMERIC_FEATURES if feature != tool_wear_feature]
            fallback_kwargs = {
                "query_instances": input_df,
                "total_CFs": internal_total_cfs,
                "desired_class": desired_class,
                "features_to_vary": fallback_features,
                "permitted_range": {
                    feature: [float(dataset[feature].min()), float(dataset[feature].max())]
                    for feature in fallback_features
                },
            }

            fallback_counterfactuals = explainer.generate_counterfactuals(**fallback_kwargs)
            fallback_df = fallback_counterfactuals.cf_examples_list[0].final_cfs_df
            valid_fallback_df = keep_valid_tool_wear_rows(fallback_df)

            if not valid_fallback_df.empty:
                return valid_fallback_df, None

        return None, (
            "No valid counterfactual solutions were found after applying the Tool wear constraint. "
            "Tool wear is allowed to stay the same or decrease only."
        )

    except Exception as exc:
        # One more KDTree fallback when the first KDTree call itself raises an exception.
        if method_name == "KDTree":
            try:
                fallback_features = [feature for feature in NUMERIC_FEATURES if feature != tool_wear_feature]
                fallback_kwargs = {
                    "query_instances": input_df,
                    "total_CFs": internal_total_cfs,
                    "desired_class": desired_class,
                    "features_to_vary": fallback_features,
                    "permitted_range": {
                        feature: [float(dataset[feature].min()), float(dataset[feature].max())]
                        for feature in fallback_features
                    },
                }

                fallback_counterfactuals = explainer.generate_counterfactuals(**fallback_kwargs)
                fallback_df = fallback_counterfactuals.cf_examples_list[0].final_cfs_df
                valid_fallback_df = keep_valid_tool_wear_rows(fallback_df)

                if not valid_fallback_df.empty:
                    return valid_fallback_df, None

            except Exception as fallback_exc:
                return None,  (
                     "KDTree did not find a feasible solution for this input under the Tool wear constraint. "
                    "This means no nearby KDTree counterfactual was available without increasing Tool wear. "
                     "Please review the Genetic and Random solutions."
                    
                )

        return None, str(exc)



# ==========================================
# Page header and home page
# ==========================================
def render_header(title: str, subtitle: str) -> None:
    left, right = st.columns([5, 1])
    with left:
        st.markdown('<div class="section-title">{}</div>'.format(title), unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">{}</div>'.format(subtitle), unsafe_allow_html=True)
    with right:
        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        if st.button("← Home", key="home_{}".format(title)):
            set_route("home")
            st.rerun()


def render_cover_page() -> None:
    st.markdown(
        dedent(
            """
            <div class="cover-shell">
                <div class="cover-grid">
                    <div>
                        <div class="subtle-label">Machine Failure Prediction and Explainability Platform</div>
                        <div class="cover-title">Predict failures.<br>Explain the cause.<br>Recommend safer machine settings.</div>
                        <div class="cover-copy">
                            A two-mode experience for operators, supervisors, and technical reviewers.
                            User Mode is designed for live prediction, anchor explanations, custom cost values, and safer-setting guidance.
                            Developer Mode is designed for dataset exploration, model inspection, anchor precision and coverage, and detailed counterfactual analysis.
                        </div>
                        <div class="cover-pills">
                            <div class="cover-pill">Binary Failure Prediction</div>
                            <div class="cover-pill">Failure Type Classification</div>
                            <div class="cover-pill">Anchor Explanation</div>
                            <div class="cover-pill">Cost-Aware Counterfactuals</div>
                        </div>
                    </div>
                    <div class="home-stack">
                        <a class="mode-link" href="?page=user" target="_self">
                            <div class="home-card user-card">
                                <div class="home-top">
                                    <div class="home-title">User Mode</div>
                                    <div class="home-tag">Operational</div>
                                </div>
                                <div class="home-text">
                                    Enter live machine sensor readings, define the cost of changing each feature,
                                    inspect why the model made the prediction, and receive cost-ranked counterfactual recommendations.
                                </div>
                            </div>
                        </a>
                        <a class="mode-link" href="?page=developer" target="_self">
                            <div class="home-card dev-card">
                                <div class="home-top">
                                    <div class="home-title">Developer Mode</div>
                                    <div class="home-tag">Analytical</div>
                                </div>
                                <div class="home-text">
                                    Browse the dataset, inspect metrics, create charts, select any row,
                                    view anchor precision and coverage, and run detailed counterfactual analysis.
                                </div>
                            </div>
                        </a>
                    </div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    st.markdown('<div class="footer-note">Each card is clickable. Click any card to enter that workspace.</div>', unsafe_allow_html=True)


# ==========================================
# Counterfactual rendering with user-defined cost ranking
# ==========================================
def render_method_outputs(
    input_df: pd.DataFrame,
    dataset: pd.DataFrame,
    desired_class,
    key_prefix: str,
    cost_weights: Optional[Dict[str, float]] = None,
) -> None:
    """
    Render counterfactual outputs in the requested layout:
    - Cost-based ranking table at the top.
    - Method tabs.
    - Recommended Actions cards on the left.
    - Visual comparison charts on the right.
    - Full counterfactual table under the actions/charts.

    The action cards are built as one continuous HTML string so the
    increase/decrease actions stay inside the card.
    """
    if cost_weights is None:
        cost_weights = {feature: 1.0 for feature in NUMERIC_FEATURES}

    method_names = ["Genetic", "Random", "KDTree"]
    original_row = input_df.iloc[0]
    method_results = {}
    ranking_rows = []

    for method_name in method_names:
        with st.spinner("Running {} counterfactual search...".format(method_name)):
            cf_df, error_message = generate_counterfactuals_for_method(
                method_name=method_name,
                input_df=input_df,
                dataset=dataset,
                desired_class=desired_class,
                total_cfs=2,
            )

        method_results[method_name] = {
            "cf_df": cf_df,
            "error": error_message,
        }

        if cf_df is not None and not cf_df.empty:
            for solution_number, (_, row) in enumerate(cf_df.iterrows(), start=1):
                total_cost = calculate_solution_cost(
                    original_row=original_row,
                    cf_row=row,
                    cost_weights=cost_weights,
                    dataset=dataset,
                )

                changed_features = [
                    feature
                    for feature in NUMERIC_FEATURES
                    if float(original_row[feature]) != float(row[feature])
                ]

                ranking_rows.append(
                    {
                        "Method": method_name,
                        "Solution": solution_number,
                        "Cost Score": round(total_cost, 4),
                        "Changed Features": ", ".join(changed_features) if changed_features else "No change",
                    }
                )

    if ranking_rows:
        ranking_df = (
            pd.DataFrame(ranking_rows)
            .sort_values("Cost Score", ascending=True)
            .reset_index(drop=True)
        )
        ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))

        st.markdown(
            '<div class="section-title">Cost-Based Recommended Solutions</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtitle">'
            "The table ranks all generated counterfactual solutions using the feature-change "
            "costs entered by the user. Lower cost indicates a more practical solution."
            "</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(ranking_df, use_container_width=True)

        best_solution = ranking_df.iloc[0]
        st.success(
            "Best practical solution: {} solution {} with the lowest cost score ({:.4f}).".format(
                best_solution["Method"],
                int(best_solution["Solution"]),
                float(best_solution["Cost Score"]),
            )
        )

    tabs = st.tabs(method_names)

    for tab, method_name in zip(tabs, method_names):
        with tab:
            cf_df = method_results[method_name]["cf_df"]
            error_message = method_results[method_name]["error"]

            if error_message:
                st.warning("{}: {}".format(method_name, error_message))
                continue

            if cf_df is None or cf_df.empty:
                st.warning("{}: No counterfactual solutions were found.".format(method_name))
                continue

            action_col, chart_col = st.columns([0.95, 1.05], gap="large")

            with action_col:
                st.markdown(
                    '<div class="section-title" style="font-size:1.15rem;">Recommended Actions</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="section-subtitle">'
                    "Each solution is translated into direct increase/decrease instructions "
                    "so non-technical users can understand the required machine setting changes."
                    "</div>",
                    unsafe_allow_html=True,
                )

                for idx, (_, row) in enumerate(cf_df.iterrows()):
                    recs = build_recommendations(original_row, row)
                    solution_cost = calculate_solution_cost(
                        original_row=original_row,
                        cf_row=row,
                        cost_weights=cost_weights,
                        dataset=dataset,
                    )

                    if recs:
                        action_rows = "".join(
                            "<div style='display:flex; gap:0.55rem; align-items:flex-start; margin:0.58rem 0;'>"
                            "<span style='font-size:1.05rem; line-height:1.45;'>•</span>"
                            "<span>{}</span>"
                            "</div>".format(html.escape(str(rec)))
                            for rec in recs
                        )
                    else:
                        action_rows = (
                            "<div style='display:flex; gap:0.55rem; align-items:flex-start; margin:0.58rem 0;'>"
                            "<span style='font-size:1.05rem; line-height:1.45;'>•</span>"
                            "<span>No numeric changes were returned for this solution.</span>"
                            "</div>"
                        )

                    card_html = (
                        "<div class='glass-card' style='min-height:360px; margin-bottom:1rem; padding:1.25rem 1.35rem;'>"
                        "<div class='section-title' style='font-size:1.08rem; margin-bottom:0.45rem;'>"
                        f"Solution {idx + 1}"
                        "</div>"
                        "<div class='small-note' style='margin-bottom:0.75rem;'>"
                        f"Cost score: <strong>{solution_cost:.4f}</strong>"
                        "</div>"
                        "<div class='small-note' style='margin-bottom:0.45rem; font-weight:700;'>"
                        "Recommended feature changes:"
                        "</div>"
                        "<div class='small-note' style='line-height:1.65;'>"
                        f"{action_rows}"
                        "</div>"
                        "</div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

            with chart_col:
                st.markdown(
                    '<div class="section-title" style="font-size:1.15rem;">Visual Comparison</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="section-subtitle">'
                    "The charts compare the original machine settings with the generated "
                    "counterfactual solutions and show the exact change from the original values."
                    "</div>",
                    unsafe_allow_html=True,
                )

                st.plotly_chart(
                    build_comparison_chart(
                        original_row,
                        cf_df,
                        "{} • Original vs Counterfactual Values".format(method_name),
                    ),
                    use_container_width=True,
                    key="{}_{}_compare".format(key_prefix, method_name),
                )

                st.plotly_chart(
                    build_delta_chart(
                        original_row,
                        cf_df,
                        "{} • Change from Original Settings".format(method_name),
                    ),
                    use_container_width=True,
                    key="{}_{}_delta".format(key_prefix, method_name),
                )

            with st.expander("View {} counterfactual table".format(method_name)):
                st.dataframe(cf_df, use_container_width=True)


# ==========================================
# User Mode
# ==========================================
def render_user_mode() -> None:
    render_header(
        "User Mode",
        "Enter machine operating conditions, define feature-change costs, view the model explanation, and compare cost-ranked counterfactual solutions.",
    )

    dataset = load_dataset()
    binary_model, failure_type_model, label_encoder, metrics, metadata = load_or_train_artifacts()

    st.markdown('<div class="small-note" style="margin-bottom:0.9rem;">Dataset source: {}</div>'.format(get_dataset_path_text()), unsafe_allow_html=True)

    left, right = st.columns([0.96, 1.14], gap="large")
    type_options = sorted(dataset["Type"].dropna().unique().tolist())
    numeric_ranges = {
        feature: (float(dataset[feature].min()), float(dataset[feature].max()), float(dataset[feature].median()))
        for feature in NUMERIC_FEATURES
    }

    with left:
        st.markdown('<div class="section-title">Machine Input Panel</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Provide the current sensor readings and the relative cost of changing each operating feature.</div>',
            unsafe_allow_html=True,
        )
        with st.form("user_form", clear_on_submit=False):
            machine_type = st.selectbox("Machine Type", options=type_options)
            c1, c2 = st.columns(2)
            with c1:
                air_temp = st.number_input("Air temperature [K]", value=float(numeric_ranges["Air temperature [K]"][2]), step=0.1, format="%.2f")
                process_temp = st.number_input("Process temperature [K]", value=float(numeric_ranges["Process temperature [K]"][2]), step=0.1, format="%.2f")
                rotational_speed = st.number_input("Rotational speed [rpm]", value=float(numeric_ranges["Rotational speed [rpm]"][2]), step=1.0, format="%.2f")
            with c2:
                torque = st.number_input("Torque [Nm]", value=float(numeric_ranges["Torque [Nm]"][2]), step=0.1, format="%.2f")
                tool_wear = st.number_input("Tool wear [min]", value=float(numeric_ranges["Tool wear [min]"][2]), step=1.0, format="%.2f")
                st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

            st.markdown("##### Cost of Feature Changes")
            st.caption("Higher values mean the feature is harder, more expensive, or less practical to modify.")
            cost_col1, cost_col2 = st.columns(2)
            with cost_col1:
                air_cost = st.number_input("Cost: Air temperature", min_value=0.0, value=1.0, step=0.1)
                process_cost = st.number_input("Cost: Process temperature", min_value=0.0, value=1.0, step=0.1)
                speed_cost = st.number_input("Cost: Rotational speed", min_value=0.0, value=2.0, step=0.1)
            with cost_col2:
                torque_cost = st.number_input("Cost: Torque", min_value=0.0, value=2.0, step=0.1)
                tool_wear_cost = st.number_input("Cost: Tool wear", min_value=0.0, value=3.0, step=0.1)

            submit = st.form_submit_button("Run Prediction")


        st.markdown(
            dedent(
                """
                <div class="glass-card" style="margin-top:1rem;">
                    <div class="section-title" style="font-size:1.05rem;">Input Feature Definitions</div>
                    <div class="small-note">
                        <strong>Machine Type:</strong> Product quality category used by the dataset.
                        <br><strong>Air temperature:</strong> Ambient air temperature around the machine.
                        <br><strong>Process temperature:</strong> Temperature generated during the machine process.
                        <br><strong>Rotational speed:</strong> Shaft rotation speed measured in revolutions per minute.
                        <br><strong>Torque:</strong> Rotational force applied during operation.
                        <br><strong>Tool wear:</strong> Accumulated tool usage time measured in minutes.
                    </div>
                    <div class="small-note" style="margin-top:0.85rem;">
                        Counterfactual recommendations only allow the tool wear value to be maintained
                        or reduced; the system does not recommend increasing tool wear.
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with right:
        if not submit:
            st.markdown(
                dedent(
                    """
                    <div class="glass-card" style="min-height: 520px; display:flex; flex-direction:column; justify-content:center;">
                        <div class="section-title" style="font-size:1.45rem;">Ready for Live Prediction</div>
                        <div class="small-note">
                            Enter the machine readings and feature-change costs on the left, then run the prediction.
                            The app will display the prediction result, risk interpretation, anchor explanation,
                            and cost-ranked counterfactual recommendations.
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            return

        input_df = pd.DataFrame([
            {
                "Type": machine_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            }
        ])

        user_cost_weights = {
            "Air temperature [K]": air_cost,
            "Process temperature [K]": process_cost,
            "Rotational speed [rpm]": speed_cost,
            "Torque [Nm]": torque_cost,
            "Tool wear [min]": tool_wear_cost,
        }

        failure_prob = float(binary_model.predict_proba(input_df)[0][1])
        failure_pred = int(binary_model.predict(input_df)[0])
        risk_level, risk_css, risk_message = infer_risk(failure_prob)

        failure_type_label = None
        if failure_pred == 1:
            failure_type_label = get_failure_type_label(failure_type_model.predict(input_df)[0], label_encoder)

        st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)
        risk_banner(risk_level, risk_css, risk_message)

        m1, m2, m3 = st.columns(3)
        with m1:
            metric_card("Prediction", "Failure" if failure_pred == 1 else "No Failure", "Binary machine state")
        with m2:
            metric_card("Failure Type", failure_type_label or "Not Applicable", "Conditional multiclass output")
        with m3:
            metric_card("Risk Level", risk_level, "Probability-based interpretation")

        c1, c2 = st.columns([1.0, 1.1], gap="large")
        with c1:
            st.plotly_chart(build_probability_gauge(failure_prob), use_container_width=True)
        with c2:
            pill_color = "pill-green" if risk_level == "Low Risk" else ("pill-amber" if risk_level == "Medium Risk" else "pill-red")
            pills = '<span class="pill {}">{}</span>'.format(pill_color, risk_level)
            pills += '<span class="pill pill-blue">{:.1f}% Probability</span>'.format(failure_prob * 100)
            if failure_type_label:
                pills += '<span class="pill pill-red">{}</span>'.format(failure_type_label)
            else:
                pills += '<span class="pill pill-slate">No Failure Type</span>'

            st.markdown(
                dedent(
                    """
                    <div class="glass-card" style="min-height: 320px;">
                        <div class="section-title">Decision Interpretation</div>
                        <div style="margin-bottom: 12px;">{}</div>
                        <div class="small-note" style="margin-bottom: 10px;">{}</div>
                        <div class="small-note">
                            Recommended priority:
                            <ul>
                                <li>Low Risk → Continue standard monitoring</li>
                                <li>Medium Risk → Review the settings before the next cycle</li>
                                <li>High Risk → Correct the settings immediately</li>
                            </ul>
                        </div>
                    </div>
                    """.format(pills, recommendation_text(risk_level, failure_type_label))
                ),
                unsafe_allow_html=True,
            )

        anchor_info = explain_anchor_for_input(input_df, threshold=0.95)
        render_anchor_explanation(anchor_info, show_metrics=False)

        st.markdown('<div class="section-title">Counterfactual Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">All three methods are shown below. Solutions are ranked according to the feature-change costs you entered.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="small-note" style="margin-bottom:0.75rem;">Tool wear is treated as an accumulated-usage feature: counterfactual recommendations may keep it unchanged or reduce it, but they cannot increase it.</div>',
            unsafe_allow_html=True,
        )

        if not DICE_AVAILABLE:
            st.warning("Counterfactual analysis requires the dice-ml package in your environment.")
            return

        if failure_pred == 1:
            desired_class = 0
            st.markdown(
                '<div class="small-note" style="margin-bottom: 0.75rem;">The current input is predicted as <strong>Failure</strong>, so the target for all methods is the safer <strong>No Failure</strong> class.</div>',
                unsafe_allow_html=True,
            )
        else:
            desired_class = "opposite"
            st.info("The current input is already predicted as No Failure. The app generates opposite-class stress-test counterfactuals for analysis.")

        render_method_outputs(
            input_df=input_df,
            dataset=dataset,
            desired_class=desired_class,
            key_prefix="user",
            cost_weights=user_cost_weights,
        )


# ==========================================
# Developer Mode
# ==========================================
def render_confusion_heatmap(matrix_values: List[List[int]], x_labels: List[str], y_labels: List[str], title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(z=matrix_values, x=x_labels, y=y_labels, colorscale="Blues", text=matrix_values, texttemplate="%{text}")
    )
    fig.update_layout(title=title, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=360, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def render_data_explorer(dataset: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Select columns, filter the dataset, browse rows, and generate custom visualizations.</div>', unsafe_allow_html=True)

    all_columns = dataset.columns.tolist()
    default_columns = [col for col in FEATURES + ["Machine failure"] if col in all_columns]
    c1, c2, c3, c4 = st.columns([1.25, 1.0, 0.9, 0.9], gap="large")
    with c1:
        selected_columns = st.multiselect("Columns to display", options=all_columns, default=default_columns)
    with c2:
        type_options = sorted(dataset["Type"].dropna().unique().tolist())
        chosen_types = st.multiselect("Machine type filter", options=type_options, default=type_options)
    with c3:
        failure_filter = st.multiselect("Failure state", options=[0, 1], default=[0, 1], format_func=lambda x: "Failure" if x == 1 else "No Failure")
    with c4:
        row_limit = st.slider("Rows to show", min_value=10, max_value=min(1000, len(dataset)), value=min(80, len(dataset)), step=10)

    filtered_df = dataset[dataset["Type"].isin(chosen_types)].copy()
    filtered_df = filtered_df[filtered_df["Machine failure"].isin(failure_filter)].copy()
    display_columns = selected_columns if selected_columns else all_columns
    display_df = filtered_df[display_columns].head(row_limit)

    st.dataframe(display_df, use_container_width=True, height=420)
    st.download_button("Download current table as CSV", data=display_df.to_csv(index=False).encode("utf-8"), file_name="filtered_machine_data.csv", mime="text/csv")

    st.markdown('<div class="section-title" style="margin-top:1rem;">Visualization Builder</div>', unsafe_allow_html=True)
    numeric_columns = [col for col in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[col])]
    if len(numeric_columns) < 1:
        st.info("No numeric columns are available for visualization.")
        return

    v1, v2, v3 = st.columns(3)
    with v1:
        chart_type = st.selectbox("Chart type", ["Histogram", "Box Plot", "Scatter Plot", "Line Plot"])
    with v2:
        x_column = st.selectbox("X axis", options=numeric_columns)
    with v3:
        y_column = None
        if chart_type in ["Scatter Plot", "Line Plot"]:
            y_column = st.selectbox("Y axis", options=numeric_columns, index=min(1, len(numeric_columns) - 1))

    fig = None
    if chart_type == "Histogram":
        fig = px.histogram(filtered_df, x=x_column, color="Type", nbins=30, title="Histogram of {}".format(x_column))
    elif chart_type == "Box Plot":
        fig = px.box(filtered_df, x="Type", y=x_column, color="Type", title="{} by Machine Type".format(x_column))
    elif chart_type == "Scatter Plot" and y_column:
        fig = px.scatter(filtered_df, x=x_column, y=y_column, color=filtered_df["Machine failure"].map({0: "No Failure", 1: "Failure"}), hover_data=FEATURES, title="{} vs {}".format(y_column, x_column))
    elif chart_type == "Line Plot" and y_column:
        temp_df = filtered_df.reset_index(drop=False)
        fig = px.line(temp_df, x=x_column, y=y_column, color="Type", title="{} vs {}".format(y_column, x_column))

    if fig is not None:
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), legend=dict(orientation="h", y=1.1, x=0), margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)


def render_developer_row_lab(dataset: pd.DataFrame, binary_model, failure_type_model, label_encoder) -> None:
    st.markdown('<div class="section-title">Row Counterfactual Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Choose any row, inspect anchor precision and coverage, and compare counterfactual methods with charts and solution tables.</div>', unsafe_allow_html=True)

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        row_index = st.number_input("Dataset row index", min_value=0, max_value=len(dataset) - 1, value=0, step=1)
        target_mode = st.radio("Counterfactual target", options=["Safer State (No Failure)", "Opposite Prediction"], horizontal=False)
    with right:
        selected_row = dataset.iloc[int(row_index)][FEATURES].copy()
        input_df = pd.DataFrame([selected_row.to_dict()])
        st.dataframe(input_df, use_container_width=True)

    probability = float(binary_model.predict_proba(input_df)[0][1])
    prediction = int(binary_model.predict(input_df)[0])
    failure_type_label = None
    if prediction == 1:
        failure_type_label = get_failure_type_label(failure_type_model.predict(input_df)[0], label_encoder)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Selected Row Prediction", "Failure" if prediction == 1 else "No Failure", "Binary model result")
    with c2:
        metric_card("Failure Probability", "{:.2f}%".format(probability * 100), "Binary probability")
    with c3:
        metric_card("Failure Type", failure_type_label or "Not Applicable", "Conditional multiclass output")

    anchor_info = explain_anchor_for_input(input_df, threshold=0.95)
    render_anchor_explanation(anchor_info, show_metrics=True)

    if not DICE_AVAILABLE:
        st.warning("Counterfactual analysis requires the dice-ml package in your environment.")
        return

    desired_class = 0 if target_mode == "Safer State (No Failure)" else "opposite"
    if target_mode == "Safer State (No Failure)" and prediction == 0:
        st.info("This row is already predicted as No Failure. The safer-state target is already satisfied, so the methods may return minimal or no changes.")

    render_method_outputs(input_df=input_df, dataset=dataset, desired_class=desired_class, key_prefix="developer_row")


def render_model_metrics(metrics: Dict) -> None:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Binary Classification Metrics</div>', unsafe_allow_html=True)
        st.json(metrics.get("binary", {}), expanded=False)
        cm = metrics.get("binary", {}).get("confusion_matrix", [])
        if cm:
            st.plotly_chart(render_confusion_heatmap(cm, ["Predicted No Failure", "Predicted Failure"], ["Actual No Failure", "Actual Failure"], "Binary Confusion Matrix"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Failure Type Metrics</div>', unsafe_allow_html=True)
        st.json(metrics.get("failure_type", {}), expanded=False)
        cm2 = metrics.get("failure_type", {}).get("confusion_matrix", [])
        classes = metrics.get("failure_type", {}).get("classes", [])
        if cm2 and classes:
            st.plotly_chart(render_confusion_heatmap(cm2, classes, classes, "Failure Type Confusion Matrix"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


def render_feature_analysis(binary_model, failure_type_model) -> None:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Logistic Regression Coefficients</div>', unsafe_allow_html=True)
        try:
            feature_names = binary_model.named_steps["preprocessor"].get_feature_names_out()
            coefficients = binary_model.named_steps["classifier"].coef_[0]
            coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients}).sort_values("Coefficient", ascending=False)
            fig = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", title="Binary Model Coefficients")
            fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=20, r=20, t=50, b=20), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Coefficient table"):
                st.dataframe(coef_df, use_container_width=True)
        except Exception as exc:
            st.warning("Could not extract logistic coefficients: {}".format(exc))
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">XGBoost Feature Importance</div>', unsafe_allow_html=True)
        try:
            xgb_classifier = failure_type_model.named_steps["classifier"]
            xgb_features = failure_type_model.named_steps["preprocessor"].get_feature_names_out()
            importance_df = pd.DataFrame({"Feature": xgb_features, "Importance": xgb_classifier.feature_importances_}).sort_values("Importance", ascending=False)
            fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h", title="Failure Type Model Importance")
            fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=20, r=20, t=50, b=20), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Importance table"):
                st.dataframe(importance_df, use_container_width=True)
        except Exception as exc:
            st.warning("Could not extract XGBoost feature importances: {}".format(exc))
        st.markdown('</div>', unsafe_allow_html=True)


def render_developer_mode() -> None:
    render_header(
        "Developer Mode",
        "Browse the dataset, inspect summary metrics, analyze feature behavior, choose any row, view anchor precision and coverage, and compare counterfactual methods.",
    )

    dataset = load_dataset()
    binary_model, failure_type_model, label_encoder, metrics, metadata = load_or_train_artifacts()

    st.markdown(
        '<div class="small-note" style="margin-bottom:0.9rem;">Dataset source: {}</div>'.format(
            get_dataset_path_text()
        ),
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        metric_card(
            "Dataset Rows",
            "{:,}".format(len(dataset)),
            "Total observations"
        )

    with m2:
        metric_card(
            "Binary Test Accuracy",
            "{:.3f}".format(metrics.get("binary", {}).get("test_accuracy", 0.0)),
            "Holdout evaluation"
        )

    with m3:
        metric_card(
            "Failure Type Test Accuracy",
            "{:.3f}".format(metrics.get("failure_type", {}).get("test_accuracy", 0.0)),
            "Multiclass evaluation"
        )

    with m4:
        engines = []
        engines.append("DiCE" if DICE_AVAILABLE else "No DiCE")
        engines.append("Anchor" if ANCHOR_AVAILABLE else "No Anchor")
        metric_card(
            "Explainability Engine",
            " + ".join(engines),
            "Counterfactual and rule-based XAI"
        )

    tabs = st.tabs([
        "Dataset Explorer",
        "Feature Analysis",
        "Row Counterfactual Lab"
    ])

    with tabs[0]:
        render_data_explorer(dataset)

    with tabs[1]:
        render_feature_analysis(binary_model, failure_type_model)

    with tabs[2]:
        render_developer_row_lab(
            dataset,
            binary_model,
            failure_type_model,
            label_encoder
        )


# ==========================================
# Footer and app entrypoint
# ==========================================
def render_footer() -> None:
    st.markdown('<div class="footer-note">Machine Failure Prediction and Explainability Dashboard • Streamlit Interface • User and Developer Workspaces</div>', unsafe_allow_html=True)


def main() -> None:
    inject_global_css()
    route = get_route()
    if route == "user":
        render_user_mode()
    elif route == "developer":
        render_developer_mode()
    else:
        render_cover_page()
    render_footer()


if __name__ == "__main__":
    main()
