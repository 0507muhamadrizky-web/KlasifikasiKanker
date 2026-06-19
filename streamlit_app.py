"""
============================================================
APLIKASI KLASIFIKASI KANKER PAYUDARA — STREAMLIT VERSION
ULTRA LIGHT — Tanpa TensorFlow, Tanpa Plotly
============================================================
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🩺 Klasifikasi Kanker Payudara",
    page_icon="🩺",
    layout="wide",
)

# ============================================================
# KONSTANTA
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SAMPLE_MALIGNANT = [
    17.99, 10.38, 122.80, 1001.0, 0.11840, 0.27760, 0.30010, 0.14710, 0.24190, 0.07871,
    1.0950, 0.9053, 8.5890, 153.40, 0.006399, 0.049040, 0.053730, 0.015870, 0.030030, 0.006193,
    25.380, 17.33, 184.60, 2019.0, 0.16220, 0.66560, 0.71190, 0.26540, 0.46010, 0.11890,
]
SAMPLE_BENIGN = [
    13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.18850, 0.05766,
    0.26990, 0.7886, 2.0580, 23.56, 0.008462, 0.014600, 0.023870, 0.013150, 0.019800, 0.002300,
    15.110, 19.26, 99.70, 711.2, 0.14400, 0.17730, 0.23900, 0.12880, 0.29770, 0.07259,
]
DEFAULTS = [
    14.127, 19.290, 91.969, 654.889, 0.09636, 0.10430, 0.08880, 0.04892, 0.18120, 0.06280,
    0.40510, 1.21700, 2.8670, 40.337, 0.00704, 0.02548, 0.03189, 0.01172, 0.02054, 0.003795,
    16.269, 25.677, 107.26, 880.583, 0.13240, 0.25420, 0.27270, 0.11480, 0.29010, 0.08394,
]

FEATURE_NAMES = [
    "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
    "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean", "Fractal Dimension Mean",
    "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
    "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
    "Radius Worst", "Texture Worst", "Perimeter Worst", "Area Worst", "Smoothness Worst",
    "Compactness Worst", "Concavity Worst", "Concave Points Worst", "Symmetry Worst", "Fractal Dimension Worst",
]

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    result = {
        "svm": None,
        "rf": None,
        "scaler": None,
    }
    
    try:
        result["svm"] = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
    except Exception as e:
        st.sidebar.error(f"SVM Error: {e}")
    
    try:
        result["rf"] = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    except Exception as e:
        st.sidebar.error(f"RF Error: {e}")
    
    try:
        result["scaler"] = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    except Exception as e:
        st.sidebar.error(f"Scaler Error: {e}")
    
    return result

models = load_models()

# ============================================================
# SESSION STATE
# ============================================================
if "feature_values" not in st.session_state:
    st.session_state.feature_values = DEFAULTS.copy()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🩺 Klasifikasi Kanker")
    st.markdown("---")
    
    # Sample buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔴 Ganas", use_container_width=True):
            st.session_state.feature_values = SAMPLE_MALIGNANT.copy()
            st.rerun()
    with col2:
        if st.button("🟢 Jinak", use_container_width=True):
            st.session_state.feature_values = SAMPLE_BENIGN.copy()
            st.rerun()
    
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.feature_values = DEFAULTS.copy()
        st.rerun()
    
    st.markdown("---")
    st.subheader("📊 Input Fitur")
    
    current_vals = st.session_state.feature_values
    new_vals = list(current_vals)
    
    for i, name in enumerate(FEATURE_NAMES):
        new_vals[i] = st.number_input(
            name,
            value=float(current_vals[i]),
            format="%.5f",
            key=f"feat_{i}",
        )
    
    st.session_state.feature_values = new_vals
    
    st.markdown("---")
    st.subheader("🤖 Status Model")
    st.write(f"{'✅' if models['svm'] else '❌'} SVM")
    st.write(f"{'✅' if models['rf'] else '❌'} Random Forest")
    st.write(f"{'✅' if models['scaler'] else '❌'} Scaler")

# ============================================================
# MAIN CONTENT
# ============================================================
st.title("🩺 Klasifikasi Kanker Payudara")
st.markdown("""
> ⚠️ **Disclaimer:** Aplikasi ini hanya untuk keperluan **edukasi & riset akademis**.  
> Hasil prediksi **bukan** diagnosis medis. Konsultasikan dengan dokter untuk diagnosis resmi.
""")

if st.button("🔬 PREDIKSI SEKARANG", type="primary", use_container_width=True):
    features = np.array(st.session_state.feature_values, dtype=float).reshape(1, -1)
    
    if models["scaler"] is None:
        st.error("❌ Scaler tidak tersedia!")
        st.stop()
    
    features_scaled = models["scaler"].transform(features)
    results = {}
    
    # SVM
    if models["svm"] is not None:
        svm_pred = int(models["svm"].predict(features_scaled)[0])
        svm_prob = models["svm"].predict_proba(features_scaled)[0]
        results["SVM"] = {
            "pred": svm_pred,
            "benign": svm_prob[0] * 100,
            "malignant": svm_prob[1] * 100,
        }
    
    # Random Forest
    if models["rf"] is not None:
        rf_pred = int(models["rf"].predict(features)[0])
        rf_prob = models["rf"].predict_proba(features)[0]
        results["Random Forest"] = {
            "pred": rf_pred,
            "benign": rf_prob[0] * 100,
            "malignant": rf_prob[1] * 100,
        }
    
    if not results:
        st.error("❌ Tidak ada model yang tersedia!")
        st.stop()
    
    # Voting
    preds = [v["pred"] for v in results.values()]
    majority = 1 if sum(preds) > len(preds) / 2 else 0
    
    # Display result
    col_result, col_detail = st.columns([1, 2])
    
    with col_result:
        if majority == 1:
            st.error("## 🔴 MALIGNANT (GANAS)")
            st.metric("Kesimpulan", "Malignant", delta="⚠️ Konsultasi Dokter")
        else:
            st.success("## 🟢 BENIGN (JINAK)")
            st.metric("Kesimpulan", "Benign", delta="✅ Lanjutkan Pemeriksaan")
    
    with col_detail:
        st.write("### Detail Prediksi per Model")
        for model_name, res in results.items():
            pred_text = "🔴 Malignant" if res["pred"] == 1 else "🟢 Benign"
            st.write(f"**{model_name}:** {pred_text}")
            st.progress(res["malignant"] / 100, text=f"Risiko Malignant: {res['malignant']:.1f}%")
            st.progress(res["benign"] / 100, text=f"Risiko Benign: {res['benign']:.1f}%")
            st.write("---")
else:
    st.info("👈 Isi fitur di sidebar, lalu klik **PREDIKSI SEKARANG**")
    
    # Dataset info
    st.markdown("---")
    st.subheader("📊 Dataset Wisconsin Breast Cancer")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sampel", "569")
    col2.metric("Benign (Jinak)", "357")
    col3.metric("Malignant (Ganas)", "212")
    col4.metric("Fitur", "30")
