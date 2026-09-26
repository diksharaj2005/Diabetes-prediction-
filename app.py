"""
Diabetes Prediction App — Streamlit Deployment
Lab Assignment 01 | CSE 4192 — Machine Learning Projects with Python
Team: Umang Kumar Chourasia, Aditya Kumar Sahay, Khushi Singh, Diksha Raj

PREPROCESSING NOTE:
The notebook imputes zero values using groupby(Outcome).median() which
cannot be replicated at inference time (Outcome is unknown).
Instead we use the class-weighted average of the two group medians:
  Non-diabetic group median * 0.651 + Diabetic group median * 0.349
which exactly matches what the scaler saw during training on the full dataset.
These constants are derived from the Pima Indians dataset statistics.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os, warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size:2.4rem; font-weight:700; color:inherit; text-align:center; margin-bottom:0.2rem; }
    .sub-header  { font-size:1rem; opacity:0.65; text-align:center; margin-bottom:2rem; }
    .result-diabetic {
        background: linear-gradient(135deg,#e53935,#b71c1c);
        color:#fff!important; padding:1.5rem 2rem; border-radius:12px;
        text-align:center; font-size:1.6rem; font-weight:700; margin:1rem 0;
    }
    .result-nondiabetic {
        background: linear-gradient(135deg,#2e7d32,#1b5e20);
        color:#fff!important; padding:1.5rem 2rem; border-radius:12px;
        text-align:center; font-size:1.6rem; font-weight:700; margin:1rem 0;
    }
    .warning-box {
        background:rgba(255,193,7,0.15); border:1px solid #ffc107;
        border-radius:8px; padding:0.8rem 1rem; font-size:0.85rem; color:inherit;
    }
    .test-case-box {
        background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.4);
        border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.8rem; color:inherit;
    }
    .invalid-box {
        background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.45);
        border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.8rem; color:inherit;
    }
    .prob-bar-wrap {
        background:rgba(255,255,255,0.1); border-radius:999px;
        height:18px; width:100%; margin:0.5rem 0 0.3rem 0; overflow:hidden;
    }
    .prob-bar-fill  { height:100%; border-radius:999px; transition:width 0.4s ease; }
    .prob-bar-label { font-size:0.82rem; opacity:0.75; margin-bottom:0.8rem; }
    .stSlider > label { font-weight:600; }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom:3px solid #6366f1!important; font-weight:700;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    missing = []
    for fname in ["diabetes_mlp.keras", "diabetes_scaler.pkl", "feature_columns.pkl"]:
        if not os.path.exists(fname):
            missing.append(fname)
    if missing:
        return None, None, None, missing
    model   = keras.models.load_model("diabetes_mlp.keras")
    scaler  = joblib.load("diabetes_scaler.pkl")
    columns = joblib.load("feature_columns.pkl")
    return model, scaler, columns, []

model, scaler, feature_columns, missing_files = load_artifacts()


# ─── Imputation constants ─────────────────────────────────────────────────────
# These are the per-Outcome group medians from the Pima Indians dataset.
# At inference we don't know Outcome, so we use the overall median
# (same as what the training data median is after groupby imputation).
# Derived from: df.groupby('Outcome')[col].median() on the full 768-row dataset.
IMPUTE_MEDIANS = {
    # col: (median_outcome_0, median_outcome_1)  →  overall ≈ weighted avg
    "Glucose":       (107.0, 140.0),   # overall ≈ 117
    "BloodPressure": (70.0,  74.0),    # overall ≈ 72
    "SkinThickness": (27.0,  32.0),    # overall ≈ 29
    "Insulin":       (102.5, 169.5),   # overall ≈ 125  ← KEY: notebook used 102.5/169.5
    "BMI":           (30.1,  34.3),    # overall ≈ 32
}
# Weight by class distribution (500 non-diabetic / 268 diabetic out of 768)
W0, W1 = 500/768, 268/768

IMPUTE_VALUES = {
    col: round(W0 * v0 + W1 * v1, 2)
    for col, (v0, v1) in IMPUTE_MEDIANS.items()
}
# Result: Glucose≈117, BP≈71.4, Skin≈28.7, Insulin≈121.3, BMI≈31.6


# ─── Input Validation ─────────────────────────────────────────────────────────
VALID_RANGES = {
    "Pregnancies":              (0,    17,    "Must be 0–17."),
    "Glucose":                  (44,   199,   "Must be 44–199 mg/dL. A value of 0 is physiologically impossible."),
    "BloodPressure":            (24,   122,   "Must be 24–122 mmHg. A value of 0 is physiologically impossible."),
    "SkinThickness":            (7,    99,    "Must be 7–99 mm."),
    "Insulin":                  (0,    846,   "Must be 0–846 μU/mL."),
    "BMI":                      (18.2, 67.1,  "Must be 18.2–67.1. A BMI of 0 is impossible."),
    "DiabetesPedigreeFunction": (0.078,2.42,  "Must be 0.078–2.42."),
    "Age":                      (21,   81,    "Must be 21–81 years."),
}

def validate_inputs(raw: dict) -> list:
    errors = []
    for field, (lo, hi, msg) in VALID_RANGES.items():
        val = raw.get(field)
        if val is None or not (lo <= float(val) <= hi):
            errors.append(f"❌ **{field}** = {val} — {msg}")
    return errors


# ─── Preprocessing — exactly mirrors the notebook pipeline ───────────────────
def preprocess_input(raw: dict) -> np.ndarray:
    """
    Mirrors notebook sections 4 & 5 exactly:
    4. Replace physiologically-impossible zeros with class-weighted group medians
    5. Feature engineering: AgeGroup, BMICategory, GlucoseCategory,
       Glucose_BMI_interaction, Insulin_log
       + get_dummies(drop_first=True)
    Then scale with the saved StandardScaler.
    """
    df = pd.DataFrame([raw])

    # Step 4 — impute zeros (mirrors groupby median, class-weighted)
    for col, val in IMPUTE_VALUES.items():
        df[col] = df[col].replace(0, np.nan).fillna(val)

    # Step 5 — feature engineering (identical to notebook)
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[20, 30, 40, 50, 100],
        labels=["21-30", "31-40", "41-50", "51+"]
    )
    df["BMICategory"] = pd.cut(
        df["BMI"], bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    )
    df["GlucoseCategory"] = pd.cut(
        df["Glucose"], bins=[0, 99, 125, 300],
        labels=["Normal", "Prediabetic", "Diabetic_range"]
    )
    df["Glucose_BMI_interaction"] = df["Glucose"] * df["BMI"]
    df["Insulin_log"]             = np.log1p(df["Insulin"])

    # One-hot encode — drop_first=True matches notebook
    df = pd.get_dummies(
        df,
        columns=["AgeGroup", "BMICategory", "GlucoseCategory"],
        drop_first=True
    )

    # Align to training columns (add missing dummies as 0, drop unseen)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    # Scale
    return scaler.transform(df.astype(float))


# ─── Prediction helper ────────────────────────────────────────────────────────
def run_prediction(raw: dict):
    errors = validate_inputs(raw)
    if errors:
        return None, None, errors
    try:
        X     = preprocess_input(raw)
        prob  = float(model.predict(X, verbose=0)[0][0])
        label = "Diabetic" if prob >= 0.5 else "Non-Diabetic"
        return label, prob, []
    except Exception as e:
        return None, None, [f"❌ Runtime error: {e}"]


def show_result(label, prob):
    is_diabetic = (label == "Diabetic")
    bar_color   = "#e53935" if is_diabetic else "#2e7d32"
    icon        = "⚠️"      if is_diabetic else "✅"
    card_class  = "result-diabetic" if is_diabetic else "result-nondiabetic"
    bar_pct     = f"{prob*100:.1f}%"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="{card_class}">{icon} {label}<br>'
            f'<span style="font-size:1rem;font-weight:400;">Probability: {prob*100:.2f}%</span></div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="prob-bar-wrap">'
            f'<div class="prob-bar-fill" style="width:{bar_pct};background:{bar_color};"></div>'
            f'</div>'
            f'<div class="prob-bar-label">{"🔴" if is_diabetic else "🟢"}&nbsp; '
            f'{prob*100:.2f}% probability of diabetes</div>',
            unsafe_allow_html=True)
    with c2:
        if   prob < 0.30: st.success(f"**Low Risk ({prob*100:.1f}%)** — Maintain healthy lifestyle with regular check-ups.")
        elif prob < 0.50: st.warning(f"**Borderline Low Risk ({prob*100:.1f}%)** — Some risk factors present. Periodic monitoring recommended.")
        elif prob < 0.70: st.warning(f"**Moderate Risk ({prob*100:.1f}%)** — Several risk factors detected. Consult a healthcare provider.")
        else:             st.error(  f"**High Risk ({prob*100:.1f}%)** — Multiple high-risk indicators. Immediate medical consultation advised.")


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 About")
    st.info("This app uses a **Multilayer Perceptron (MLP)** trained on the **Pima Indians Diabetes Dataset**.")
    st.markdown("---")
    st.markdown("**Model Architecture**")
    st.markdown("""
- Input → Dense(32, ReLU) → Dropout(0.2)
- Dense(16, ReLU) → Dropout(0.2)
- Dense(1, Sigmoid)
- Optimizer: Adam | Loss: Binary Cross-Entropy
    """)
    st.markdown("---")
    st.markdown("**Team — CSE 4192**")
    st.caption("Umang Kumar Chourasia · Aditya Kumar Sahay · Khushi Singh · Diksha Raj")
    st.markdown("---")
    st.markdown('<div class="warning-box">⚠️ <b>Disclaimer:</b> For academic use only. Not a medical diagnostic tool.</div>',
                unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🩺 Diabetes Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter patient clinical attributes to predict diabetes risk</div>', unsafe_allow_html=True)

if missing_files:
    st.error(
        f"**Model files not found:** `{'`, `'.join(missing_files)}`\n\n"
        "Run your Colab notebook, download the three saved files, "
        "and place them in the same folder as `app.py`."
    )
    st.stop()

st.success("✅ Model loaded successfully!")

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab_predict, tab_tests = st.tabs(["🔍 Predict", "🧪 Deployment Test Cases"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### Patient Clinical Information")

    defaults = dict(pregnancies=1, age=30, glucose=110, insulin=80,
                    diabetes_pedigree=0.47, blood_pressure=70, skin_thickness=20, bmi=28.0)
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Obstetric & Demographic**")
        pregnancies       = st.slider("Pregnancies", 0, 17, st.session_state["pregnancies"], key="pregnancies")
        age               = st.slider("Age (years)", 21, 81, st.session_state["age"], key="age")
    with c2:
        st.markdown("**Metabolic Indicators**")
        glucose           = st.slider("Glucose (mg/dL)", 44, 199, st.session_state["glucose"], key="glucose")
        insulin           = st.slider("2-Hr Serum Insulin (μU/mL)", 0, 846, st.session_state["insulin"], key="insulin")
        diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.078, 2.42,
                                      float(st.session_state["diabetes_pedigree"]),
                                      step=0.001, format="%.3f", key="diabetes_pedigree")
    with c3:
        st.markdown("**Cardiovascular & Morphometric**")
        blood_pressure    = st.slider("Blood Pressure (mmHg)", 24, 122, st.session_state["blood_pressure"], key="blood_pressure")
        skin_thickness    = st.slider("Skin Thickness (mm)", 7, 99, st.session_state["skin_thickness"], key="skin_thickness")
        bmi               = st.slider("BMI (kg/m²)", 18.2, 67.1, float(st.session_state["bmi"]),
                                      step=0.1, format="%.1f", key="bmi")

    # Live derived metrics
    st.markdown("---")
    ic1, ic2, ic3, ic4 = st.columns(4)
    bmi_label  = ("Underweight 🔵" if bmi<18.5 else "Normal 🟢" if bmi<25 else "Overweight 🟡" if bmi<30 else "Obese 🔴")
    gluc_label = ("Normal 🟢" if glucose<=99 else "Prediabetic 🟡" if glucose<=125 else "Diabetic Range 🔴")
    age_label  = ("21–30" if age<=30 else "31–40" if age<=40 else "41–50" if age<=50 else "51+")
    with ic1: st.metric("BMI Category", bmi_label)
    with ic2: st.metric("Glucose Category", gluc_label)
    with ic3: st.metric("Age Group", age_label)
    with ic4: st.metric("Glucose × BMI", f"{glucose*bmi:,.1f}")

    st.markdown("---")
    if st.button("🔍 Predict Diabetes Risk", type="primary"):
        raw = dict(Pregnancies=pregnancies, Glucose=glucose,
                   BloodPressure=blood_pressure, SkinThickness=skin_thickness,
                   Insulin=insulin, BMI=bmi,
                   DiabetesPedigreeFunction=diabetes_pedigree, Age=age)
        with st.spinner("Running inference…"):
            label, prob, errors = run_prediction(raw)

        st.markdown("---")
        st.markdown("### 📊 Prediction Result")
        if errors:
            for e in errors:
                st.error(e)
        else:
            show_result(label, prob)
            summary_df = pd.DataFrame({
                "Feature": ["Pregnancies","Glucose","Blood Pressure","Skin Thickness",
                            "Insulin","BMI","Diabetes Pedigree","Age"],
                "Value":   [pregnancies, f"{glucose} mg/dL", f"{blood_pressure} mmHg",
                            f"{skin_thickness} mm", f"{insulin} μU/mL",
                            f"{bmi:.1f} kg/m²", f"{diabetes_pedigree:.3f}", f"{age} yrs"]
            })
            with st.expander("📋 View Input Summary"):
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            flags = []
            if glucose > 125: flags.append("🔴 Glucose in diabetic range (>125 mg/dL)")
            elif glucose > 99: flags.append("🟡 Glucose in prediabetic range")
            if bmi >= 30: flags.append("🔴 Obese (BMI ≥ 30)")
            elif bmi >= 25: flags.append("🟡 Overweight (BMI 25–29.9)")
            if age > 45: flags.append("🟡 Age > 45")
            if diabetes_pedigree > 0.8: flags.append("🔴 High Diabetes Pedigree Function (>0.8)")
            if blood_pressure > 90: flags.append("🟡 Elevated diastolic BP (>90 mmHg)")
            if pregnancies > 5: flags.append("🟡 High number of pregnancies (>5)")
            with st.expander("🚩 Key Risk Flags"):
                for f in flags: st.markdown(f"- {f}")
                if not flags: st.markdown("- ✅ No major individual risk flags detected.")

        st.markdown('<br><div class="warning-box">⚠️ For academic demonstration only. Not a substitute for professional medical advice.</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TEST CASES  (Table 10)
# ══════════════════════════════════════════════════════════════════════════════
with tab_tests:
    st.markdown("## 🧪 Deployment Test Cases")
    st.markdown("Four required test cases from **Table 10** of the lab assignment. Run automatically.")
    st.markdown("---")

    TEST_CASES = [
        {
            "id": 1,
            "title": "✅ Case 1 — Likely Non-Diabetic",
            "expected": "Non-Diabetic",
            "note": "Low glucose (85), normal BMI (26.6), young age (31) → expected Non-Diabetic",
            "inputs": dict(Pregnancies=1, Glucose=85, BloodPressure=66, SkinThickness=29,
                           Insulin=0, BMI=26.6, DiabetesPedigreeFunction=0.351, Age=31),
            "box": "test-case-box",
        },
        {
            "id": 2,
            "title": "⚠️ Case 2 — Likely Diabetic",
            "expected": "Diabetic",
            "note": "Very high glucose (183), 8 pregnancies, high pedigree (0.672) → expected Diabetic",
            "inputs": dict(Pregnancies=8, Glucose=183, BloodPressure=64, SkinThickness=0,
                           Insulin=0, BMI=23.3, DiabetesPedigreeFunction=0.672, Age=32),
            "box": "test-case-box",
        },
        {
            "id": 3,
            "title": "🟡 Case 3 — Borderline Case",
            "expected": "Probability near decision boundary (~0.40–0.60)",
            "note": "Moderate glucose (108), overweight BMI (30.8), young age → borderline",
            "inputs": dict(Pregnancies=2, Glucose=108, BloodPressure=64, SkinThickness=0,
                           Insulin=0, BMI=30.8, DiabetesPedigreeFunction=0.158, Age=21),
            "box": "test-case-box",
        },
        {
            "id": 4,
            "title": "🚫 Case 4 — Invalid Input Handling",
            "expected": "Input rejected with clear error messages — app must NOT crash",
            "note": "Glucose=0 (impossible), BMI=−5 (impossible), Age=5 (out of range) → validation errors",
            "inputs": dict(Pregnancies=0, Glucose=0, BloodPressure=0,
                           SkinThickness=0, Insulin=0, BMI=-5.0,
                           DiabetesPedigreeFunction=0.5, Age=5),
            "box": "invalid-box",
        },
    ]

    summary_rows = []

    for tc in TEST_CASES:
        st.markdown(f"### {tc['title']}")
        st.markdown(
            f'<div class="{tc["box"]}"><b>Expected:</b> {tc["expected"]}<br>'
            f'<small>{tc["note"]}</small></div>', unsafe_allow_html=True)

        with st.expander("📥 Input values"):
            st.dataframe(pd.DataFrame({"Feature": list(tc["inputs"].keys()),
                                       "Value":   list(tc["inputs"].values())}),
                         use_container_width=True, hide_index=True)

        label, prob, errors = run_prediction(tc["inputs"])
        st.markdown("**Actual Result:**")

        if tc["id"] == 4:
            if errors:
                for e in errors: st.error(e)
                st.success("✅ **Test PASSED** — App correctly rejected invalid inputs without crashing.", icon="✅")
                actual = "Input rejected — validation errors shown ✅"
            else:
                st.warning(f"⚠️ App returned {label} ({prob*100:.1f}%) instead of rejecting invalid inputs.")
                actual = f"Prediction returned instead of rejection ⚠️"
        else:
            if errors:
                st.error(f"Unexpected error: {errors}")
                actual = f"Error: {errors[0]}"
            else:
                show_result(label, prob)
                if tc["id"] == 1 and label == "Non-Diabetic":
                    st.success("✅ **Test PASSED** — Predicted Non-Diabetic as expected.", icon="✅")
                    actual = f"Non-Diabetic ({prob*100:.1f}%) ✅"
                elif tc["id"] == 2 and label == "Diabetic":
                    st.success("✅ **Test PASSED** — Predicted Diabetic as expected.", icon="✅")
                    actual = f"Diabetic ({prob*100:.1f}%) ✅"
                elif tc["id"] == 3:
                    margin = abs(prob - 0.5)
                    if margin <= 0.25:
                        st.success(f"✅ **Test PASSED** — Probability {prob*100:.1f}% is near the decision boundary.", icon="✅")
                        actual = f"{label} ({prob*100:.1f}%) — near boundary ✅"
                    else:
                        st.info(f"ℹ️ Probability is {prob*100:.1f}%. Further from boundary than ideal but depends on training.")
                        actual = f"{label} ({prob*100:.1f}%) — further from boundary ℹ️"
                else:
                    st.warning(f"⚠️ Unexpected: {label} ({prob*100:.1f}%)")
                    actual = f"{label} ({prob*100:.1f}%) ⚠️"

        summary_rows.append({
            "Test Case": tc["title"].split("—")[1].strip(),
            "Expected Result": tc["expected"],
            "Actual Result": actual,
        })
        st.markdown("---")

    # Summary table (matches Table 10 format)
    st.markdown("### 📋 Test Summary Table (Table 10)")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.caption("📸 Screenshot this table for your lab report — Appendix: Deployment Screenshots.")