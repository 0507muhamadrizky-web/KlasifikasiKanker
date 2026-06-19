"""
============================================================
APLIKASI KLASIFIKASI KANKER PAYUDARA
Framework: Gradio (HuggingFace Space)
Model: SVM, Random Forest, ANN (Deep Learning)
Dataset: Wisconsin Breast Cancer Dataset
============================================================
"""

import os
import warnings
import pickle
import numpy as np
import gradio as gr
import joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ============================================================
# PATH KONFIGURASI
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ============================================================
# LOAD SEMUA MODEL
# ============================================================
print("=" * 50)
print("🔄 Memuat semua model...")
print("=" * 50)

svm_model, rf_model, scaler, ann_model = None, None, None, None
ann_available = False

try:
    svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
    print("✅ SVM model berhasil dimuat")
except Exception as e:
    print(f"❌ Gagal memuat SVM: {e}")

try:
    rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    print("✅ Random Forest berhasil dimuat")
except Exception as e:
    print(f"❌ Gagal memuat RF: {e}")

try:
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    print("✅ Scaler berhasil dimuat")
except Exception as e:
    print(f"❌ Gagal memuat Scaler: {e}")

try:
    with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)
    print(f"✅ Feature names dimuat: {len(feature_names)} fitur")
except Exception as e:
    print(f"⚠️ Feature names tidak ditemukan, pakai default")
    feature_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
    ]

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
            return np.array([[x[0][0]]])  # Mengembalikan matriks 2D agar kompatibel dengan output gradionya

    ann_paths = [
        os.path.join(MODELS_DIR, "ann_model.h5"),
        os.path.join(MODELS_DIR, "ann_model.keras"),
    ]
    for ann_path in ann_paths:
        if os.path.exists(ann_path):
            ann_model = NumpyANN(ann_path)
            ann_available = True
            print(f"✅ ANN model berhasil dimuat melalui NumpyANN dari {os.path.basename(ann_path)}")
            break
    if not ann_available:
        print("⚠️ ann_model.h5 tidak ditemukan — prediksi ANN dinonaktifkan")
except Exception as e:
    print(f"⚠️ Tidak dapat memuat ANN: {e}")

print("=" * 50)

# ============================================================
# SAMPLE DATA (Wisconsin Breast Cancer Dataset)
# ============================================================
# Data Malignant (Ganas) - Pasien #1
SAMPLE_MALIGNANT = [
    17.99, 10.38, 122.80, 1001.0, 0.11840, 0.27760, 0.30010, 0.14710, 0.24190, 0.07871,
    1.0950, 0.9053, 8.5890, 153.40, 0.006399, 0.049040, 0.053730, 0.015870, 0.030030, 0.006193,
    25.380, 17.33, 184.60, 2019.0, 0.16220, 0.66560, 0.71190, 0.26540, 0.46010, 0.11890,
]

# Data Benign (Jinak) - Pasien #2
SAMPLE_BENIGN = [
    13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.18850, 0.05766,
    0.26990, 0.7886, 2.0580, 23.56, 0.008462, 0.014600, 0.023870, 0.013150, 0.019800, 0.002300,
    15.110, 19.26, 99.70, 711.2, 0.14400, 0.17730, 0.23900, 0.12880, 0.29770, 0.07259,
]

# Default values (rata-rata dataset)
DEFAULTS = [
    14.127, 19.290, 91.969, 654.889, 0.09636, 0.10430, 0.08880, 0.04892, 0.18120, 0.06280,
    0.40510, 1.21700, 2.8670, 40.337, 0.00704, 0.02548, 0.03189, 0.01172, 0.02054, 0.003795,
    16.269, 25.677, 107.26, 880.583, 0.13240, 0.25420, 0.27270, 0.11480, 0.29010, 0.08394,
]

# ============================================================
# FITUR: LABEL & GROUPING
# ============================================================
FEATURES_MEAN = [
    ("Radius Mean", 0), ("Texture Mean", 1), ("Perimeter Mean", 2),
    ("Area Mean", 3), ("Smoothness Mean", 4), ("Compactness Mean", 5),
    ("Concavity Mean", 6), ("Concave Points Mean", 7),
    ("Symmetry Mean", 8), ("Fractal Dimension Mean", 9),
]
FEATURES_SE = [
    ("Radius SE", 10), ("Texture SE", 11), ("Perimeter SE", 12),
    ("Area SE", 13), ("Smoothness SE", 14), ("Compactness SE", 15),
    ("Concavity SE", 16), ("Concave Points SE", 17),
    ("Symmetry SE", 18), ("Fractal Dimension SE", 19),
]
FEATURES_WORST = [
    ("Radius Worst", 20), ("Texture Worst", 21), ("Perimeter Worst", 22),
    ("Area Worst", 23), ("Smoothness Worst", 24), ("Compactness Worst", 25),
    ("Concavity Worst", 26), ("Concave Points Worst", 27),
    ("Symmetry Worst", 28), ("Fractal Dimension Worst", 29),
]

# ============================================================
# HELPER: BUAT CARD HTML UNTUK SATU MODEL
# ============================================================
def make_model_card(model_name, icon, pred, prob_benign, prob_malignant):
    is_malignant = pred == 1
    color = "#ef4444" if is_malignant else "#22c55e"
    bg = "rgba(239,68,68,0.08)" if is_malignant else "rgba(34,197,94,0.08)"
    label = "MALIGNANT (GANAS) 🔴" if is_malignant else "BENIGN (JINAK) 🟢"
    border_color = "#ef4444" if is_malignant else "#22c55e"

    return f"""
    <div style="border: 2px solid {border_color}; border-radius: 14px; padding: 18px;
                margin: 10px 0; background: {bg}; transition: all 0.3s;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
            <span style="font-size: 1.5em;">{icon}</span>
            <h3 style="margin: 0; color: #1e293b; font-size: 1em;">{model_name}</h3>
        </div>
        <div style="font-size: 1.3em; font-weight: 700; color: {color}; margin-bottom: 14px;">{label}</div>
        <div style="font-size: 0.85em; color: #475569; margin-bottom: 4px;">
            🟢 Benign (Jinak): <b>{prob_benign:.1f}%</b>
        </div>
        <div style="background: #e2e8f0; border-radius: 99px; height: 10px; margin-bottom: 10px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #22c55e, #16a34a);
                         width: {prob_benign}%; height: 100%; border-radius: 99px;
                         transition: width 0.5s;"></div>
        </div>
        <div style="font-size: 0.85em; color: #475569; margin-bottom: 4px;">
            🔴 Malignant (Ganas): <b>{prob_malignant:.1f}%</b>
        </div>
        <div style="background: #e2e8f0; border-radius: 99px; height: 10px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #ef4444, #dc2626);
                         width: {prob_malignant}%; height: 100%; border-radius: 99px;
                         transition: width 0.5s;"></div>
        </div>
    </div>
    """

# ============================================================
# FUNGSI PREDIKSI UTAMA
# ============================================================
def predict(*args):
    try:
        features = np.array(args, dtype=float).reshape(1, -1)

        if scaler is None:
            return "<div style='color:red; padding:16px;'>❌ Scaler tidak tersedia. Pastikan scaler.pkl ada di folder models/</div>"

        features_scaled = scaler.transform(features)
        cards_html = ""
        predictions = []

        # --- SVM ---
        if svm_model is not None:
            svm_pred = int(svm_model.predict(features_scaled)[0])
            svm_prob = svm_model.predict_proba(features_scaled)[0]
            predictions.append(svm_pred)
            cards_html += make_model_card(
                "Support Vector Machine (SVM)", "⚡",
                svm_pred,
                svm_prob[0] * 100,
                svm_prob[1] * 100,
            )
        else:
            cards_html += """<div style='border:2px dashed #cbd5e1; border-radius:12px; padding:16px; color:#94a3b8; text-align:center;'>
                ⚡ SVM — Model tidak tersedia</div>"""

        # --- Random Forest ---
        if rf_model is not None:
            rf_pred = int(rf_model.predict(features)[0])
            rf_prob = rf_model.predict_proba(features)[0]
            predictions.append(rf_pred)
            cards_html += make_model_card(
                "Random Forest", "🌲",
                rf_pred,
                rf_prob[0] * 100,
                rf_prob[1] * 100,
            )
        else:
            cards_html += """<div style='border:2px dashed #cbd5e1; border-radius:12px; padding:16px; color:#94a3b8; text-align:center;'>
                🌲 Random Forest — Model tidak tersedia</div>"""

        # --- ANN ---
        if ann_model is not None and ann_available:
            ann_prob_raw = float(ann_model.predict(features_scaled, verbose=0)[0][0])
            ann_pred = 1 if ann_prob_raw > 0.5 else 0
            predictions.append(ann_pred)
            cards_html += make_model_card(
                "ANN – Deep Learning", "🧠",
                ann_pred,
                (1 - ann_prob_raw) * 100,
                ann_prob_raw * 100,
            )
        else:
            cards_html += """
            <div style="border: 2px dashed #c4b5fd; border-radius: 14px; padding: 18px; margin: 10px 0;
                         background: rgba(139,92,246,0.05); text-align: center;">
                <div style="font-size: 1.5em;">🧠</div>
                <h3 style="color: #7c3aed; margin: 4px 0;">ANN – Deep Learning</h3>
                <p style="color: #94a3b8; font-size: 0.9em; margin: 0;">
                    ⚠️ Model ANN belum tersedia.<br>
                    Jalankan kode Google Colab, lalu upload <b>ann_model.h5</b> ke folder <code>models/</code>
                </p>
            </div>
            """

        # --- Ringkasan Voting ---
        if predictions:
            malignant_votes = sum(predictions)
            benign_votes = len(predictions) - malignant_votes
            majority = 1 if malignant_votes > benign_votes else 0
            consensus = "MALIGNANT (GANAS)" if majority == 1 else "BENIGN (JINAK)"
            consensus_icon = "🔴" if majority == 1 else "🟢"
            consensus_color = "#ef4444" if majority == 1 else "#22c55e"
            consensus_bg = "linear-gradient(135deg, #1e1b4b, #312e81)" if majority == 1 else "linear-gradient(135deg, #052e16, #14532d)"
            agree = max(malignant_votes, benign_votes)

            summary_html = f"""
            <div style="background: {consensus_bg}; color: white; border-radius: 16px;
                         padding: 20px; margin-bottom: 16px; text-align: center;
                         box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
                <div style="font-size: 0.9em; color: #94a3b8; margin-bottom: 6px;">
                    KESIMPULAN ({agree}/{len(predictions)} model sepakat)
                </div>
                <div style="font-size: 2em; font-weight: 800; color: {consensus_color};">
                    {consensus_icon} {consensus}
                </div>
                <div style="font-size: 0.8em; color: #64748b; margin-top: 10px;">
                    ⚠️ Hasil ini hanya untuk referensi edukasi. Konsultasikan dengan dokter.
                </div>
            </div>
            """
        else:
            summary_html = ""

        return summary_html + cards_html

    except Exception as e:
        return f"""
        <div style="background: #fef2f2; border: 2px solid #ef4444; border-radius: 12px;
                     padding: 20px; text-align: center;">
            <h3 style="color: #dc2626;">❌ Error saat Prediksi</h3>
            <code style="color: #b91c1c;">{str(e)}</code>
            <p style="color: #6b7280; font-size: 0.9em; margin-top: 8px;">
                Pastikan semua file model tersedia di folder <code>models/</code>
            </p>
        </div>
        """

# ============================================================
# CUSTOM CSS
# ============================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

.gradio-container { max-width: 1300px !important; margin: 0 auto !important; }

#header-block {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 40%, #0c4a6e 100%);
    color: white;
    padding: 32px 24px;
    border-radius: 20px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
}

#header-block h1 {
    font-size: 2.4em;
    font-weight: 800;
    margin: 0 0 8px;
    background: linear-gradient(90deg, #93c5fd, #c4b5fd, #86efac);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

#header-block p { color: #94a3b8; margin: 4px 0; font-size: 1.0em; }

.warn-block {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border-left: 5px solid #f59e0b;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 16px;
    color: #92400e;
    font-size: 0.9em;
}

.gr-accordion { border-radius: 12px !important; border: 1px solid #e2e8f0 !important; }

.primary-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
    font-weight: 700 !important;
}

footer { display: none !important; }
"""

# ============================================================
# BUILD UI
# ============================================================
with gr.Blocks(css=CUSTOM_CSS, title="🩺 Klasifikasi Kanker Payudara", theme=gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)) as demo:

    # ── Header ──────────────────────────────────────────────────
    gr.HTML("""
    <div id="header-block">
        <h1>🩺 Klasifikasi Kanker Payudara</h1>
        <p>Breast Cancer Classification — Wisconsin Dataset</p>
        <p style="color:#64748b; font-size:0.85em;">
            ⚡ SVM &nbsp;|&nbsp; 🌲 Random Forest &nbsp;|&nbsp; 🧠 ANN (Deep Learning)
        </p>
    </div>
    <div class="warn-block">
        ⚠️ <b>Disclaimer:</b> Aplikasi ini untuk keperluan <b>edukasi & riset</b> saja.
        Bukan pengganti diagnosis dokter profesional.
    </div>
    """)

    with gr.Tabs():

        # ──────────────────────────────────────────────────────────
        # TAB 1: PREDIKSI
        # ──────────────────────────────────────────────────────────
        with gr.Tab("🔬 Prediksi"):
            with gr.Row(equal_height=False):

                # ── KOLOM KIRI: INPUT ──────────────────────────────
                with gr.Column(scale=3, min_width=420):
                    gr.Markdown("### 📊 Input Fitur Pasien")
                    gr.Markdown(
                        "*Masukkan 30 nilai fitur hasil biopsi, atau gunakan tombol contoh di bawah.*"
                    )

                    inputs = []  # kumpulkan semua gr.Number di sini

                    with gr.Accordion("🔵 Mean Features — Nilai Rata-Rata (10 fitur)", open=True):
                        with gr.Row():
                            for label, idx in FEATURES_MEAN[:5]:
                                n = gr.Number(label=label, value=DEFAULTS[idx], precision=5)
                                inputs.append(n)
                        with gr.Row():
                            for label, idx in FEATURES_MEAN[5:]:
                                n = gr.Number(label=label, value=DEFAULTS[idx], precision=5)
                                inputs.append(n)

                    with gr.Accordion("🟡 SE Features — Standard Error (10 fitur)", open=False):
                        with gr.Row():
                            for label, idx in FEATURES_SE[:5]:
                                n = gr.Number(label=label, value=DEFAULTS[idx], precision=5)
                                inputs.append(n)
                        with gr.Row():
                            for label, idx in FEATURES_SE[5:]:
                                n = gr.Number(label=label, value=DEFAULTS[idx], precision=5)
                                inputs.append(n)

                    with gr.Accordion("🔴 Worst Features — Nilai Terburuk (10 fitur)", open=False):
                        with gr.Row():
                            for label, idx in FEATURES_WORST[:5]:
                                n = gr.Number(label=label, value=DEFAULTS[idx], precision=5)
                                inputs.append(n)
                        with gr.Row():
                            for label, idx in FEATURES_WORST[5:]:
                                n = gr.Number(label=label, value=DEFAULTS[idx], precision=5)
                                inputs.append(n)

                    gr.Markdown("---")
                    with gr.Row():
                        btn_predict = gr.Button("🔬 Prediksi Sekarang", variant="primary", size="lg", elem_classes="primary-btn")
                    with gr.Row():
                        btn_malignant = gr.Button("🔴 Muat Contoh: Ganas", variant="secondary", size="sm")
                        btn_benign    = gr.Button("🟢 Muat Contoh: Jinak",  variant="secondary", size="sm")

                # ── KOLOM KANAN: OUTPUT ────────────────────────────
                with gr.Column(scale=2, min_width=320):
                    gr.Markdown("### 📈 Hasil Prediksi")
                    output_html = gr.HTML(
                        value="""
                        <div style="text-align:center; padding:60px 20px; color:#94a3b8;">
                            <div style="font-size:4em; margin-bottom:12px;">🩺</div>
                            <p style="font-size:1.1em;">Isi nilai fitur, lalu klik<br>
                            <b style="color:#6366f1;">🔬 Prediksi Sekarang</b></p>
                        </div>
                        """
                    )

            # ── EVENT HANDLERS ─────────────────────────────────────
            btn_predict.click(fn=predict, inputs=inputs, outputs=output_html)
            btn_malignant.click(fn=lambda: SAMPLE_MALIGNANT, outputs=inputs)
            btn_benign.click(fn=lambda: SAMPLE_BENIGN, outputs=inputs)

        # ──────────────────────────────────────────────────────────
        # TAB 2: PANDUAN
        # ──────────────────────────────────────────────────────────
        with gr.Tab("📖 Panduan Penggunaan"):
            gr.HTML("""
            <div style="padding: 8px 0; max-width: 900px; margin: 0 auto;">
                <h2>📖 Cara Menggunakan Aplikasi</h2>
                <ol style="line-height: 2; color: #374151;">
                    <li><b>Isi nilai fitur</b> pada form di tab Prediksi (30 fitur numerik)</li>
                    <li>Atau klik <b>🔴 Muat Contoh: Ganas</b> / <b>🟢 Muat Contoh: Jinak</b> untuk data sampel</li>
                    <li>Klik <b>🔬 Prediksi Sekarang</b></li>
                    <li>Lihat hasil dari ketiga model beserta <i>confidence score</i></li>
                </ol>

                <h2 style="margin-top: 32px;">📊 Penjelasan Fitur</h2>
                <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
                    <thead>
                        <tr style="background: linear-gradient(135deg, #1e1b4b, #312e81); color:white;">
                            <th style="padding:10px; text-align:left;">Kelompok</th>
                            <th style="padding:10px; text-align:left;">Deskripsi</th>
                            <th style="padding:10px; text-align:left;">Jumlah Fitur</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="background:#eff6ff;">
                            <td style="padding:10px;"><b>🔵 Mean</b></td>
                            <td style="padding:10px;">Nilai rata-rata karakteristik sel</td>
                            <td style="padding:10px; text-align:center;">10</td>
                        </tr>
                        <tr style="background:#fefce8;">
                            <td style="padding:10px;"><b>🟡 SE</b></td>
                            <td style="padding:10px;">Standard error (standar deviasi) karakteristik sel</td>
                            <td style="padding:10px; text-align:center;">10</td>
                        </tr>
                        <tr style="background:#fff1f2;">
                            <td style="padding:10px;"><b>🔴 Worst</b></td>
                            <td style="padding:10px;">Nilai terbesar / terburuk dari 3 inti sel terburuk</td>
                            <td style="padding:10px; text-align:center;">10</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """)

        # ──────────────────────────────────────────────────────────
        # TAB 3: TENTANG MODEL
        # ──────────────────────────────────────────────────────────
        with gr.Tab("🤖 Tentang Model"):
            ann_status = "✅ Tersedia" if ann_available else "⚠️ Belum di-upload (jalankan Google Colab dulu)"
            gr.HTML(f"""
            <div style="padding: 8px 0; max-width: 960px; margin: 0 auto;">
                <h2>🤖 Model Machine Learning & Deep Learning</h2>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                             gap: 16px; margin: 20px 0;">

                    <div style="border: 2px solid #6366f1; border-radius: 16px; padding: 20px;
                                 background: linear-gradient(135deg, #eef2ff, #e0e7ff);">
                        <h3 style="color: #4338ca; margin: 0 0 8px;">⚡ Support Vector Machine</h3>
                        <p style="color: #4b5563; font-size: 0.9em; margin: 0 0 12px;">
                            Algoritma yang mencari hyperplane optimal untuk memisahkan dua kelas.
                        </p>
                        <ul style="color: #374151; font-size: 0.85em; margin: 0; padding-left: 16px;">
                            <li>Kernel: RBF (Radial Basis Function)</li>
                            <li>C = 10, Gamma = scale</li>
                            <li>Probability = True</li>
                            <li>Input: Fitur ter-standardisasi (ScaledScaler)</li>
                        </ul>
                    </div>

                    <div style="border: 2px solid #22c55e; border-radius: 16px; padding: 20px;
                                 background: linear-gradient(135deg, #f0fdf4, #dcfce7);">
                        <h3 style="color: #15803d; margin: 0 0 8px;">🌲 Random Forest</h3>
                        <p style="color: #4b5563; font-size: 0.9em; margin: 0 0 12px;">
                            Ensemble dari ratusan decision tree yang di-voting untuk klasifikasi final.
                        </p>
                        <ul style="color: #374151; font-size: 0.85em; margin: 0; padding-left: 16px;">
                            <li>n_estimators = 200 pohon</li>
                            <li>max_depth = 10</li>
                            <li>min_samples_split = 5</li>
                            <li>Input: Fitur asli (tanpa scaling)</li>
                        </ul>
                    </div>

                    <div style="border: 2px solid #a855f7; border-radius: 16px; padding: 20px;
                                 background: linear-gradient(135deg, #faf5ff, #ede9fe);">
                        <h3 style="color: #7e22ce; margin: 0 0 8px;">🧠 ANN – Deep Learning</h3>
                        <p style="color: #4b5563; font-size: 0.9em; margin: 0 0 12px;">
                            Multi-layer neural network dengan BatchNorm dan Dropout untuk regularisasi.
                        </p>
                        <ul style="color: #374151; font-size: 0.85em; margin: 0; padding-left: 16px;">
                            <li>Arsitektur: 128 → 64 → 32 → 1</li>
                            <li>Aktivasi: ReLU + Sigmoid</li>
                            <li>BatchNormalization + Dropout</li>
                            <li>Optimizer: Adam (lr=0.001)</li>
                            <li>Status: <b>{ann_status}</b></li>
                        </ul>
                    </div>
                </div>

                <h2 style="margin-top: 32px;">📊 Dataset</h2>
                <p style="color: #374151;">
                    <b>Wisconsin Breast Cancer Dataset</b> — dikumpulkan oleh Dr. William H. Wolberg,
                    University of Wisconsin–Madison. Setiap sampel adalah hasil digitalisasi gambar FNA
                    (Fine Needle Aspirate) dari massa payudara.
                </p>
                <div style="display:flex; gap:16px; flex-wrap:wrap; margin:12px 0;">
                    <div style="background:#dbeafe; border-radius:10px; padding:12px 20px; text-align:center;">
                        <div style="font-size:1.8em; font-weight:800; color:#1d4ed8;">569</div>
                        <div style="font-size:0.8em; color:#1e40af;">Total Sampel</div>
                    </div>
                    <div style="background:#dcfce7; border-radius:10px; padding:12px 20px; text-align:center;">
                        <div style="font-size:1.8em; font-weight:800; color:#15803d;">357</div>
                        <div style="font-size:0.8em; color:#166534;">Benign (Jinak)</div>
                    </div>
                    <div style="background:#fee2e2; border-radius:10px; padding:12px 20px; text-align:center;">
                        <div style="font-size:1.8em; font-weight:800; color:#dc2626;">212</div>
                        <div style="font-size:0.8em; color:#b91c1c;">Malignant (Ganas)</div>
                    </div>
                    <div style="background:#ede9fe; border-radius:10px; padding:12px 20px; text-align:center;">
                        <div style="font-size:1.8em; font-weight:800; color:#7e22ce;">30</div>
                        <div style="font-size:0.8em; color:#6b21a8;">Fitur Numerik</div>
                    </div>
                </div>

                <div style="background:#fef2f2; border-left:4px solid #ef4444; border-radius:10px;
                             padding:14px 18px; margin-top: 20px; color: #7f1d1d; font-size: 0.9em;">
                    <b>⚕️ Disclaimer Medis:</b> Aplikasi ini dibuat untuk keperluan akademis (skripsi/tugas akhir).
                    Hasil prediksi <u>tidak menggantikan</u> pemeriksaan dan diagnosis dari dokter profesional.
                    Jika Anda atau keluarga memiliki gejala kanker payudara, segera konsultasikan ke dokter.
                </div>
            </div>
            """)

    # ── Footer ───────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; padding:16px; color:#94a3b8; font-size:0.82em; margin-top:8px;
                border-top: 1px solid #e2e8f0;">
        🩺 Klasifikasi Kanker Payudara &nbsp;|&nbsp;
        Dibuat dengan ❤️ menggunakan <b>Gradio</b> & <b>TensorFlow</b> &nbsp;|&nbsp;
        Dataset: Wisconsin Breast Cancer
    </div>
    """)

# ============================================================
# LAUNCH
# ============================================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
