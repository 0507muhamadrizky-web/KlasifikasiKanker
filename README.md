---
title: Klasifikasi Kanker Payudara
emoji: 🩺
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Breast Cancer Classification dengan SVM, Random Forest & ANN
---

# 🩺 Klasifikasi Kanker Payudara

Aplikasi **Machine Learning & Deep Learning** untuk membantu klasifikasi kanker payudara sebagai **Benign (Jinak)** atau **Malignant (Ganas)** menggunakan tiga model:

| Model | Tipe | Akurasi |
|-------|------|---------|
| ⚡ Support Vector Machine | Traditional ML | >95% |
| 🌲 Random Forest | Ensemble ML | >95% |
| 🧠 ANN (Deep Learning) | Neural Network | >95% |

## 📊 Dataset

**Wisconsin Breast Cancer Dataset** — 569 sampel dengan 30 fitur numerik yang diekstrak dari gambar FNA (Fine Needle Aspirate):
- **Benign (Jinak):** 357 sampel
- **Malignant (Ganas):** 212 sampel
- **Fitur:** 30 fitur numerik (Mean, Standard Error, Worst)

## 🚀 Cara Penggunaan

1. Masukkan nilai 30 fitur pada form input (atau klik tombol **Contoh: Ganas / Jinak**)
2. Klik tombol **🔬 Prediksi**
3. Lihat hasil prediksi dari ketiga model beserta confidence score

## ⚠️ Disclaimer

> Aplikasi ini **HANYA** untuk keperluan **edukasi dan riset akademis**.
> Hasil prediksi **BUKAN** diagnosis medis.
> Selalu konsultasikan dengan dokter profesional untuk diagnosis resmi.

## 🛠️ Tech Stack

- **Frontend:** Gradio (HF Space) + Streamlit (Lokal)
- **ML Models:** scikit-learn (SVM, Random Forest)
- **Deep Learning:** TensorFlow/Keras (ANN)
- **Backend:** Python 3.10

## 📁 Struktur File

```
├── app.py                 # Gradio app (HuggingFace Space)
├── streamlit_app.py       # Streamlit app (Lokal)
├── requirements.txt       # Dependencies
├── Dockerfile             # Docker config
├── models/
│   ├── svm_model.pkl      # Trained SVM model
│   ├── rf_model.pkl       # Trained Random Forest model
│   ├── ann_model.h5       # Trained ANN model (upload dari Google Colab)
│   ├── scaler.pkl         # StandardScaler
│   └── feature_names.pkl  # Feature names
└── dataset/
    └── data.csv           # Dataset Wisconsin Breast Cancer
```

## 👨‍💻 Author

Dibuat untuk keperluan **Tugas Akhir / Skripsi** klasifikasi kanker payudara.
