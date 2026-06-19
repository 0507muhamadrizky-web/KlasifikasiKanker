"""
============================================================
APLIKASI KLASIFIKASI KANKER PAYUDARA — STREAMLIT VERSION
Framework: Streamlit (Lokal / Alternatif)
Model: SVM, Random Forest, ANN (Deep Learning)
Dataset: Wisconsin Breast Cancer Dataset
Jalankan: streamlit run streamlit_app.py
============================================================
"""

import os
import sys

# Tambahkan path Disk D agar Python bisa membaca package yang diinstal di sana
custom_path = r"D:\python_packages"
if os.path.exists(custom_path) and custom_path not in sys.path:
    sys.path.insert(0, custom_path)

import warnings
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🩺 Klasifikasi Kanker Payudara",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://huggingface.co/spaces/kyy080505/KlasifikasiKangkerPayudara",
        "About": "Klasifikasi Kanker Payudara — SVM, Random Forest, ANN",
    },
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Header gradient */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 40%, #0c4a6e 100%);
    padding: 32px 28px;
    border-radius: 20px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}
.main-header h1 {
    font-size: 2.6em;
    font-weight: 800;
    background: linear-gradient(90deg, #93c5fd, #c4b5fd, #86efac);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px;
}
.main-header p { color: #94a3b8; margin: 4px 0; font-size: 1.0em; }

/* Model cards */
.model-card {
    border-radius: 16px;
    padding: 20px;
    margin: 8px 0;
    transition: transform 0.2s, box-shadow 0.2s;
}
.model-card:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.15); }

.card-malignant {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 2px solid #ef4444;
}
.card-benign {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #22c55e;
}
.card-neutral {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
}

/* Prediction label */
.pred-label-malignant { color: #dc2626; font-size: 1.5em; font-weight: 800; }
.pred-label-benign    { color: #16a34a; font-size: 1.5em; font-weight: 800; }

/* Summary banner */
.summary-banner {
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin-bottom: 20px;
}
.summary-malignant { background: linear-gradient(135deg, #1e1b4b, #991b1b); color: white; }
.summary-benign    { background: linear-gradient(135deg, #052e16, #064e3b); color: white; }

/* Warning box */
.warning-box {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border-left: 5px solid #f59e0b;
    border-radius: 10px;
    padding: 12px 16px;
    color: #92400e;
    font-size: 0.88em;
    margin-bottom: 16px;
}

/* Stats boxes */
.stat-box {
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    font-weight: 700;
}

/* Progress bar custom */
.prog-bar-bg {
    background: #e2e8f0;
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
    margin-bottom: 6px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e1b4b) !important;
    color: white;
}
section[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
    font-size: 0.82em !important;
}
section[data-testid="stSidebar"] .stNumberInput input {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: white !important;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

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

FEATURE_GROUPS = {
    "🔵 Mean Features": [
        ("Radius Mean",            0, "%.5f"),
        ("Texture Mean",           1, "%.5f"),
        ("Perimeter Mean",         2, "%.3f"),
        ("Area Mean",              3, "%.2f"),
        ("Smoothness Mean",        4, "%.6f"),
        ("Compactness Mean",       5, "%.6f"),
        ("Concavity Mean",         6, "%.6f"),
        ("Concave Points Mean",    7, "%.6f"),
        ("Symmetry Mean",          8, "%.5f"),
        ("Fractal Dimension Mean", 9, "%.6f"),
    ],
    "🟡 SE Features": [
        ("Radius SE",              10, "%.5f"),
        ("Texture SE",             11, "%.5f"),
        ("Perimeter SE",           12, "%.5f"),
        ("Area SE",                13, "%.4f"),
        ("Smoothness SE",          14, "%.7f"),
        ("Compactness SE",         15, "%.7f"),
        ("Concavity SE",           16, "%.7f"),
        ("Concave Points SE",      17, "%.7f"),
        ("Symmetry SE",            18, "%.7f"),
        ("Fractal Dimension SE",   19, "%.7f"),
    ],
    "🔴 Worst Features": [
        ("Radius Worst",            20, "%.5f"),
        ("Texture Worst",           21, "%.5f"),
        ("Perimeter Worst",         22, "%.3f"),
        ("Area Worst",              23, "%.2f"),
        ("Smoothness Worst",        24, "%.6f"),
        ("Compactness Worst",       25, "%.6f"),
        ("Concavity Worst",         26, "%.6f"),
        ("Concave Points Worst",    27, "%.6f"),
        ("Symmetry Worst",          28, "%.6f"),
        ("Fractal Dimension Worst", 29, "%.6f"),
    ],
}

# ============================================================
# LOAD MODELS (cached)
# ============================================================
@st.cache_resource(show_spinner="⏳ Memuat model ML...")
def load_models():
    result = {
        "svm": None, "rf": None, "ann": None,
        "scaler": None, "feature_names": None,
        "ann_available": False,
    }
    try:
        result["svm"] = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
    except Exception as e:
        st.sidebar.warning(f"SVM: {e}")

    try:
        result["rf"] = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    except Exception as e:
        st.sidebar.warning(f"RF: {e}")

    try:
        result["scaler"] = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    except Exception as e:
        st.sidebar.warning(f"Scaler: {e}")

    try:
        with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "rb") as f:
            result["feature_names"] = pickle.load(f)
    except Exception:
        result["feature_names"] = [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
            "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
            "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
            "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
            "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
            "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
            "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
        ]

    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        ann_paths = [
            os.path.join(MODELS_DIR, "ann_model.h5"),
            os.path.join(MODELS_DIR, "ann_model.keras"),
        ]
        for ann_path in ann_paths:
            if os.path.exists(ann_path):
                result["ann"] = tf.keras.models.load_model(ann_path)
                result["ann_available"] = True
                break
    except Exception as e:
        try:
            import h5py
            class NumpyANN:
                def __init__(self, h5_path):
                    self.weights = {}
                    with h5py.File(h5_path, 'r') as f:
                        w = f['model_weights']
                        self.weights['d1_k'] = w['dense']['sequential']['dense']['kernel'][:]
                        self.weights['d1_b'] = w['dense']['sequential']['dense']['bias'][:]
                        self.weights['bn1_gamma'] = w['batch_normalization']['sequential']['batch_normalization']['gamma'][:]
                        self.weights['bn1_beta'] = w['batch_normalization']['sequential']['batch_normalization']['beta'][:]
                        self.weights['bn1_mm'] = w['batch_normalization']['sequential']['batch_normalization']['moving_mean'][:]
                        self.weights['bn1_mv'] = w['batch_normalization']['sequential']['batch_normalization']['moving_variance'][:]
                        
                        self.weights['d2_k'] = w['dense_1']['sequential']['dense_1']['kernel'][:]
                        self.weights['d2_b'] = w['dense_1']['sequential']['dense_1']['bias'][:]
                        self.weights['bn2_gamma'] = w['batch_normalization_1']['sequential']['batch_normalization_1']['gamma'][:]
                        self.weights['bn2_beta'] = w['batch_normalization_1']['sequential']['batch_normalization_1']['beta'][:]
                        self.weights['bn2_mm'] = w['batch_normalization_1']['sequential']['batch_normalization_1']['moving_mean'][:]
                        self.weights['bn2_mv'] = w['batch_normalization_1']['sequential']['batch_normalization_1']['moving_variance'][:]
                        
                        self.weights['d3_k'] = w['dense_2']['sequential']['dense_2']['kernel'][:]
                        self.weights['d3_b'] = w['dense_2']['sequential']['dense_2']['bias'][:]
                        self.weights['bn3_gamma'] = w['batch_normalization_2']['sequential']['batch_normalization_2']['gamma'][:]
                        self.weights['bn3_beta'] = w['batch_normalization_2']['sequential']['batch_normalization_2']['beta'][:]
                        self.weights['bn3_mm'] = w['batch_normalization_2']['sequential']['batch_normalization_2']['moving_mean'][:]
                        self.weights['bn3_mv'] = w['batch_normalization_2']['sequential']['batch_normalization_2']['moving_variance'][:]
                        
                        self.weights['d4_k'] = w['dense_3']['sequential']['dense_3']['kernel'][:]
                        self.weights['d4_b'] = w['dense_3']['sequential']['dense_3']['bias'][:]

                def predict(self, x, verbose=0):
                    x = np.dot(x, self.weights['d1_k']) + self.weights['d1_b']
                    x = np.maximum(0, x)
                    x = self.weights['bn1_gamma'] * (x - self.weights['bn1_mm']) / np.sqrt(self.weights['bn1_mv'] + 1e-3) + self.weights['bn1_beta']
                    
                    x = np.dot(x, self.weights['d2_k']) + self.weights['d2_b']
                    x = np.maximum(0, x)
                    x = self.weights['bn2_gamma'] * (x - self.weights['bn2_mm']) / np.sqrt(self.weights['bn2_mv'] + 1e-3) + self.weights['bn2_beta']
                    
                    x = np.dot(x, self.weights['d3_k']) + self.weights['d3_b']
                    x = np.maximum(0, x)
                    x = self.weights['bn3_gamma'] * (x - self.weights['bn3_mm']) / np.sqrt(self.weights['bn3_mv'] + 1e-3) + self.weights['bn3_beta']
                    
                    x = np.dot(x, self.weights['d4_k']) + self.weights['d4_b']
                    x = 1 / (1 + np.exp(-x))
                    return np.array([x])
                    
            ann_path = os.path.join(MODELS_DIR, "ann_model.h5")
            if os.path.exists(ann_path):
                result["ann"] = NumpyANN(ann_path)
                result["ann_available"] = True
        except:
            pass

    return result

models = load_models()

# ============================================================
# SESSION STATE UNTUK NILAI INPUT
# ============================================================
if "feature_values" not in st.session_state:
    st.session_state.feature_values = DEFAULTS.copy()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0;">
        <div style="font-size: 3em;">🩺</div>
        <h2 style="color: #93c5fd; margin: 8px 0 4px; font-size: 1.1em;">Klasifikasi Kanker Payudara</h2>
        <p style="color: #64748b; font-size: 0.78em; margin: 0;">Wisconsin Breast Cancer Dataset</p>
    </div>
    <hr style="border-color: #1e293b; margin: 12px 0;">
    """, unsafe_allow_html=True)

    # ── Sample Data Buttons ─────────────────────────────────────
    st.markdown("<p style='color:#94a3b8; font-size:0.85em; margin-bottom:6px;'>📦 Muat Data Sampel</p>", unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🔴 Ganas", use_container_width=True, key="btn_malignant"):
            st.session_state.feature_values = SAMPLE_MALIGNANT.copy()
            st.rerun()
    with col_s2:
        if st.button("🟢 Jinak", use_container_width=True, key="btn_benign"):
            st.session_state.feature_values = SAMPLE_BENIGN.copy()
            st.rerun()
    if st.button("🔄 Reset ke Default", use_container_width=True, key="btn_reset"):
        st.session_state.feature_values = DEFAULTS.copy()
        st.rerun()

    st.markdown("<hr style='border-color: #1e293b; margin: 14px 0;'>", unsafe_allow_html=True)

    # ── Feature Inputs ──────────────────────────────────────────
    st.markdown("<p style='color:#94a3b8; font-size:0.85em;'>📊 Input Fitur (30 fitur)</p>", unsafe_allow_html=True)

    current_vals = st.session_state.feature_values
    new_vals = list(current_vals)

    for group_name, features in FEATURE_GROUPS.items():
        with st.expander(group_name, expanded=(group_name == "🔵 Mean Features")):
            for label, idx, fmt in features:
                new_vals[idx] = st.number_input(
                    label,
                    value=float(current_vals[idx]),
                    format=fmt,
                    key=f"feat_{idx}",
                    step=None,
                )

    st.session_state.feature_values = new_vals

    st.markdown("<hr style='border-color: #1e293b; margin: 14px 0;'>", unsafe_allow_html=True)

    # ── Model Status ────────────────────────────────────────────
    st.markdown("<p style='color:#94a3b8; font-size:0.85em;'>🤖 Status Model</p>", unsafe_allow_html=True)
    st.markdown(f"{'✅' if models['svm'] else '❌'} SVM Model", unsafe_allow_html=True)
    st.markdown(f"{'✅' if models['rf'] else '❌'} Random Forest", unsafe_allow_html=True)
    st.markdown(f"{'✅' if models['ann_available'] else '⚠️'} ANN Model {'(tersedia)' if models['ann_available'] else '(belum di-upload)'}", unsafe_allow_html=True)

# ============================================================
# MAIN CONTENT
# ============================================================

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩺 Klasifikasi Kanker Payudara</h1>
    <p>Breast Cancer Classification — Wisconsin Breast Cancer Dataset</p>
    <p style="color:#64748b; font-size:0.85em;">
        ⚡ Support Vector Machine &nbsp;|&nbsp; 🌲 Random Forest &nbsp;|&nbsp; 🧠 ANN Deep Learning
    </p>
</div>
<div class="warning-box">
    ⚠️ <b>Disclaimer:</b> Aplikasi ini hanya untuk keperluan <b>edukasi & riset akademis</b>.
    Hasil prediksi <u>bukan</u> diagnosis medis. Konsultasikan dengan dokter untuk diagnosis resmi.
</div>
""", unsafe_allow_html=True)

# ── Predict Button ───────────────────────────────────────────
col_btn, col_info = st.columns([1, 3])
with col_btn:
    predict_clicked = st.button(
        "🔬 PREDIKSI SEKARANG",
        use_container_width=True,
        type="primary",
        key="btn_predict",
    )
with col_info:
    st.markdown(
        "<small style='color:#64748b;'>← Klik tombol prediksi setelah mengisi semua fitur di sidebar</small>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ============================================================
# HASIL PREDIKSI
# ============================================================
if predict_clicked:
    features = np.array(st.session_state.feature_values, dtype=float).reshape(1, -1)

    if models["scaler"] is None:
        st.error("❌ Scaler tidak tersedia. Pastikan `scaler.pkl` ada di folder `models/`")
        st.stop()

    features_scaled = models["scaler"].transform(features)
    results = {}

    # SVM
    if models["svm"] is not None:
        svm_pred = int(models["svm"].predict(features_scaled)[0])
        svm_prob = models["svm"].predict_proba(features_scaled)[0]
        results["SVM"] = {"pred": svm_pred, "benign_pct": svm_prob[0]*100, "malignant_pct": svm_prob[1]*100, "icon": "⚡"}

    # Random Forest
    if models["rf"] is not None:
        rf_pred = int(models["rf"].predict(features)[0])
        rf_prob = models["rf"].predict_proba(features)[0]
        results["Random Forest"] = {"pred": rf_pred, "benign_pct": rf_prob[0]*100, "malignant_pct": rf_prob[1]*100, "icon": "🌲"}

    # ANN
    if models["ann"] is not None and models["ann_available"]:
        ann_prob_raw = float(models["ann"].predict(features_scaled, verbose=0)[0][0])
        ann_pred = 1 if ann_prob_raw > 0.5 else 0
        results["ANN"] = {"pred": ann_pred, "benign_pct": (1-ann_prob_raw)*100, "malignant_pct": ann_prob_raw*100, "icon": "🧠"}

    # ── Summary Banner ───────────────────────────────────────────
    if results:
        preds = [v["pred"] for v in results.values()]
        majority = 1 if sum(preds) > len(preds)/2 else 0
        agree = max(sum(preds), len(preds)-sum(preds))
        is_malignant = majority == 1
        banner_cls = "summary-malignant" if is_malignant else "summary-benign"
        banner_icon = "🔴" if is_malignant else "🟢"
        banner_label = "MALIGNANT (GANAS)" if is_malignant else "BENIGN (JINAK)"

        st.markdown(f"""
        <div class="summary-banner {banner_cls}">
            <div style="font-size:0.85em; opacity:0.7; margin-bottom:6px;">
                KESIMPULAN VOTING ({agree}/{len(preds)} model sepakat)
            </div>
            <div style="font-size:2.4em; font-weight:800;">
                {banner_icon} {banner_label}
            </div>
            <div style="font-size:0.8em; opacity:0.5; margin-top:10px;">
                ⚠️ Untuk referensi edukasi saja — bukan diagnosis medis
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Model Cards ──────────────────────────────────────────────
    st.markdown("### 📊 Detail Prediksi Per Model")
    cols = st.columns(max(len(results), 1))

    for i, (model_name, res) in enumerate(results.items()):
        with cols[i]:
            is_mal = res["pred"] == 1
            card_cls = "card-malignant" if is_mal else "card-benign"
            label_cls = "pred-label-malignant" if is_mal else "pred-label-benign"
            label_text = "🔴 MALIGNANT" if is_mal else "🟢 BENIGN"

            st.markdown(f"""
            <div class="model-card {card_cls}">
                <div style="font-size:1.8em; text-align:center;">{res['icon']}</div>
                <h4 style="text-align:center; color:#1e293b; margin:6px 0;">{model_name}</h4>
                <div class="{label_cls}" style="text-align:center;">{label_text}</div>
                <div style="margin-top:14px; font-size:0.84em; color:#475569;">
                    🟢 Benign: <b>{res['benign_pct']:.1f}%</b>
                    <div class="prog-bar-bg">
                        <div style="background:linear-gradient(90deg,#22c55e,#16a34a);
                                     width:{res['benign_pct']}%;height:10px;border-radius:99px;"></div>
                    </div>
                    🔴 Malignant: <b>{res['malignant_pct']:.1f}%</b>
                    <div class="prog-bar-bg">
                        <div style="background:linear-gradient(90deg,#ef4444,#dc2626);
                                     width:{res['malignant_pct']}%;height:10px;border-radius:99px;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── ANN not available notice ─────────────────────────────────
    if not models["ann_available"]:
        st.info(
            "🧠 **ANN (Deep Learning)** belum tersedia. "
            "Jalankan kode Google Colab (`app.md`) untuk training ANN, lalu upload `ann_model.h5` ke folder `models/`"
        )

    # ── Plotly: Confidence Bar Chart ────────────────────────────
    if results:
        st.markdown("### 📈 Visualisasi Confidence Score")
        chart_col1, chart_col2 = st.columns(2)

        # Bar chart — perbandingan semua model
        with chart_col1:
            model_names = list(results.keys())
            benign_pcts = [results[m]["benign_pct"] for m in model_names]
            malignant_pcts = [results[m]["malignant_pct"] for m in model_names]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name="🟢 Benign (Jinak)",
                x=model_names,
                y=benign_pcts,
                marker_color="#22c55e",
                text=[f"{v:.1f}%" for v in benign_pcts],
                textposition="outside",
            ))
            fig_bar.add_trace(go.Bar(
                name="🔴 Malignant (Ganas)",
                x=model_names,
                y=malignant_pcts,
                marker_color="#ef4444",
                text=[f"{v:.1f}%" for v in malignant_pcts],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Perbandingan Probabilitas Semua Model",
                barmode="group",
                yaxis=dict(title="Probabilitas (%)", range=[0, 115]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                height=350,
            )
            fig_bar.update_xaxes(showgrid=False)
            fig_bar.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
            st.plotly_chart(fig_bar, use_container_width=True)

        # Gauge chart — Malignant probability dari model terbaik
        with chart_col2:
            # Pakai rata-rata malignant probability
            avg_malignant = np.mean([results[m]["malignant_pct"] for m in results])

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=avg_malignant,
                title={"text": "Rata-Rata Risiko Malignant (%)", "font": {"size": 14, "family": "Inter"}},
                number={"suffix": "%", "font": {"size": 32}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                    "bar": {"color": "#6366f1"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "#e2e8f0",
                    "steps": [
                        {"range": [0, 30],  "color": "#dcfce7"},
                        {"range": [30, 60], "color": "#fef9c3"},
                        {"range": [60, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#dc2626", "width": 4},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=350,
                margin=dict(t=80, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Tabel Ringkasan ─────────────────────────────────────
        st.markdown("### 📋 Tabel Ringkasan Prediksi")
        table_data = []
        for model_name, res in results.items():
            table_data.append({
                "Model": f"{res['icon']} {model_name}",
                "Prediksi": "🔴 Malignant (Ganas)" if res["pred"] == 1 else "🟢 Benign (Jinak)",
                "Prob. Benign (%)": f"{res['benign_pct']:.2f}",
                "Prob. Malignant (%)": f"{res['malignant_pct']:.2f}",
                "Confidence (%)": f"{max(res['benign_pct'], res['malignant_pct']):.2f}",
            })
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

else:
    # ── Placeholder (belum prediksi) ─────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#94a3b8;">
        <div style="font-size:5em; margin-bottom:16px;">🩺</div>
        <h3 style="color:#64748b; font-weight:600;">Siap untuk Prediksi</h3>
        <p>Isi nilai fitur di <b>sidebar kiri</b>, lalu klik<br>
        <b style="color:#6366f1;">🔬 PREDIKSI SEKARANG</b></p>
        <br>
        <p style="font-size:0.85em;">Atau gunakan tombol <b>🔴 Ganas</b> / <b>🟢 Jinak</b>
        di sidebar untuk muat data contoh</p>
    </div>
    """, unsafe_allow_html=True)

    # Info cards statistik dataset
    st.markdown("---")
    st.markdown("### 📊 Tentang Dataset Wisconsin Breast Cancer")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("""<div class="stat-box" style="background:#dbeafe; color:#1d4ed8;">
            <div style="font-size:2em;">569</div><div style="font-size:0.8em;">Total Sampel</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown("""<div class="stat-box" style="background:#dcfce7; color:#15803d;">
            <div style="font-size:2em;">357</div><div style="font-size:0.8em;">Benign (Jinak)</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown("""<div class="stat-box" style="background:#fee2e2; color:#dc2626;">
            <div style="font-size:2em;">212</div><div style="font-size:0.8em;">Malignant (Ganas)</div>
        </div>""", unsafe_allow_html=True)
    with s4:
        st.markdown("""<div class="stat-box" style="background:#ede9fe; color:#7e22ce;">
            <div style="font-size:2em;">30</div><div style="font-size:0.8em;">Fitur Numerik</div>
        </div>""", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<hr style="margin-top: 32px; border-color: #e2e8f0;">
<div style="text-align:center; color:#94a3b8; font-size:0.82em; padding-bottom:16px;">
    🩺 Klasifikasi Kanker Payudara &nbsp;|&nbsp;
    Dibuat dengan ❤️ menggunakan <b>Streamlit</b> & <b>TensorFlow</b> &nbsp;|&nbsp;
    Dataset: Wisconsin Breast Cancer &nbsp;|&nbsp;
    <a href="https://huggingface.co/spaces/kyy080505/KlasifikasiKangkerPayudara"
       style="color:#6366f1; text-decoration:none;">🤗 HuggingFace Space</a>
</div>
""", unsafe_allow_html=True)
