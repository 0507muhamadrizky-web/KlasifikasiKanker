# ============================================================
# GOOGLE COLAB - TRAINING 3 MODEL KLASIFIKASI KANKER PAYUDARA
# MODEL: SVM, RANDOM FOREST, ANN (DEEP LEARNING)
# ============================================================

# ============================================================
# 1. INSTALL LIBRARY (JIKA PERLU)
# ============================================================
!pip install -q pandas numpy scikit-learn tensorflow joblib matplotlib seaborn

# ============================================================
# 2. IMPOR LIBRARY
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.feature_selection import SelectKBest, f_classif

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

print("="*60)
print("✅ SEMUA LIBRARY BERHASIL DI-IMPORT")
print("="*60)

# ============================================================
# 3. UPLOAD FILE DATA.CSV
# ============================================================
from google.colab import files
print("\n📤 SILAHKAN UPLOAD FILE data.csv:")
print("   (Klik tombol 'Choose Files' dan pilih file data.csv Anda)")
print("-"*40)

uploaded = files.upload()

# Ambil nama file yang diupload
file_name = list(uploaded.keys())[0]
print(f"\n✅ File berhasil diupload: {file_name}")

# ============================================================
# 4. LOAD DAN PREPROCESSING DATA
# ============================================================
print("\n" + "="*60)
print("📊 PREPROCESSING DATA")
print("="*60)

# Load dataset
df = pd.read_csv(file_name)
print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Tampilkan 5 data pertama
print("\n📋 5 Data Pertama:")
print(df.head())

# Informasi dataset
print("\n📋 Informasi Dataset:")
print(df.info())

# Statistik deskriptif
print("\n📋 Statistik Deskriptif:")
print(df.describe())

# ============================================================
# 5. HAPUS KOLOM 'id' DAN 'Unnamed: 32' (JIKA ADA)
# ============================================================
print("\n" + "="*60)
print("🧹 BERSIHKAN KOLOM TIDAK PERLU")
print("="*60)

# Hapus kolom 'id' jika ada
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
    print("✅ Kolom 'id' dihapus")

# Hapus kolom 'Unnamed: 32' jika ada
if 'Unnamed: 32' in df.columns:
    df.drop('Unnamed: 32', axis=1, inplace=True)
    print("✅ Kolom 'Unnamed: 32' dihapus")
else:
    print("☑️ Kolom 'Unnamed: 32' tidak ditemukan.")

# ============================================================
# 6. CEK DAN BERSIHKAN MISSING VALUES (SETELAH HAPUS KOLOM TIDAK PERLU)
# ============================================================
print("\n" + "="*60)
print("🧹 CEK DAN HAPUS MISSING VALUES")
print("="*60)

missing_values_after_drop = df.isnull().sum()
total_missing_after_drop = missing_values_after_drop.sum()
print(f"\nTotal missing values setelah hapus kolom tidak perlu: {total_missing_after_drop}")

if total_missing_after_drop > 0:
    print("\n🔍 Kolom dengan missing values (setelah hapus kolom tidak perlu):")
    print(missing_values_after_drop[missing_values_after_drop > 0])
    df.dropna(inplace=True)
    print(f"\n✅ Setelah drop baris dengan NaN: {df.shape[0]} rows, {df.shape[1]} columns")
else:
    print("\n✅ Tidak ada missing values tersisa.")

# ============================================================
# 7. ENCODE TARGET
# ============================================================
print("\n" + "="*60)
print("🏷️ ENCODE TARGET")
print("="*60)

# Encode target: M=1 (Malignant/Ganas), B=0 (Benign/Jinak)
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])
print(f"\n✅ Target encoding: M=1 (Malignant/Ganas), B=0 (Benign/Jinak)")

# Cek distribusi target
print("\n📊 Distribusi Target:")
print(df['diagnosis'].value_counts())
print(f"   - Benign (0): {df['diagnosis'].value_counts().get(0, 0)} samples")
print(f"   - Malignant (1): {df['diagnosis'].value_counts().get(1, 0)} samples")

# ============================================================
# 8. PISAHKAN FITUR DAN TARGET
# ============================================================
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

print(f"\n✅ Fitur: {X.shape[1]} kolom")
print(f"✅ Target: {y.shape[0]} samples")

# ============================================================
# 9. SPLIT DATA TRAIN-TEST
# ============================================================
print("\n" + "="*60)
print("📊 SPLIT DATA TRAIN-TEST")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Training data: {X_train.shape[0]} samples")
print(f"✅ Testing data: {X_test.shape[0]} samples")
print(f"\n📊 Distribusi target (training):")
print(y_train.value_counts())
print(f"\n📊 Distribusi target (testing):")
print(y_test.value_counts())

# ============================================================
# 10. STANDARDISASI DATA
# ============================================================
print("\n" + "="*60)
print("📊 STANDARDISASI DATA (StandardScaler)")
print("="*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Standardisasi selesai")
print(f"   - Mean training: {X_train_scaled.mean():.10f}")
print(f"   - Std training: {X_train_scaled.std():.10f}")

# ============================================================
# 11. RANKING FITUR TERPENTING (ANOVA F-SCORE)
# ============================================================
print("\n" + "="*60)
print("📊 RANKING FITUR TERPENTING")
print("="*60)

selector = SelectKBest(f_classif, k='all')
selector.fit(X_train, y_train)
feature_scores = pd.DataFrame({
    'Fitur': X.columns,
    'Skor': selector.scores_
}).sort_values('Skor', ascending=False)

print("\n📊 Top 15 Fitur Terpenting:")
print(feature_scores.head(15).to_string(index=False))

# Visualisasi top 10 fitur
plt.figure(figsize=(12, 8))
sns.barplot(x='Skor', y='Fitur', data=feature_scores.head(10), palette='viridis')
plt.title('10 Fitur Terpenting dalam Diagnosis Kanker Payudara', fontsize=16)
plt.xlabel('F-Score', fontsize=12)
plt.ylabel('Fitur', fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================
# 12. TRAINING SVM
# ============================================================
print("\n" + "="*60)
print("🔄 TRAINING SVM (Support Vector Machine)")
print("="*60)

svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True,
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_prob = svm_model.predict_proba(X_test_scaled)[:, 1]

# Metrik SVM
svm_acc = accuracy_score(y_test, svm_pred)
svm_prec = precision_score(y_test, svm_pred)
svm_rec = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
svm_auc = roc_auc_score(y_test, svm_prob)

print(f"\n✅ SVM Training Selesai!")
print(f"\n📊 Metrik SVM:")
print(f"   Akurasi  : {svm_acc:.4f}")
print(f"   Presisi  : {svm_prec:.4f}")
print(f"   Recall   : {svm_rec:.4f}")
print(f"   F1-Score : {svm_f1:.4f}")
print(f"   ROC-AUC  : {svm_auc:.4f}")

# Cross-validation SVM
svm_cv = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
print(f"\n📊 Cross-Validation SVM (5-fold): {svm_cv.mean():.4f} (+/- {svm_cv.std():.4f})")

# ============================================================
# 13. TRAINING RANDOM FOREST
# ============================================================
print("\n" + "="*60)
print("🔄 TRAINING RANDOM FOREST")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Metrik Random Forest
rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred)
rf_rec = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)

print(f"\n✅ Random Forest Training Selesai!")
print(f"\n📊 Metrik Random Forest:")
print(f"   Akurasi  : {rf_acc:.4f}")
print(f"   Presisi  : {rf_prec:.4f}")
print(f"   Recall   : {rf_rec:.4f}")
print(f"   F1-Score : {rf_f1:.4f}")
print(f"   ROC-AUC  : {rf_auc:.4f}")

# Cross-validation RF
rf_cv = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f"\n📊 Cross-Validation RF (5-fold): {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")

# Feature Importance dari Random Forest
feature_importance_rf = pd.DataFrame({
    'Fitur': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Top 15 Feature Importance (Random Forest):")
print(feature_importance_rf.head(15).to_string(index=False))

# ============================================================
# 14. TRAINING ANN (DEEP LEARNING)
# ============================================================
print("\n" + "="*60)
print("🔄 TRAINING ANN (Deep Learning Neural Network)")
print("="*60)

# Bangun arsitektur ANN
ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

# Compile model
ann_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n📊 Arsitektur ANN:")
ann_model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Training
history = ann_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluasi ANN
ann_prob = ann_model.predict(X_test_scaled, verbose=0).flatten()
ann_pred = (ann_prob > 0.5).astype(int)

ann_acc = accuracy_score(y_test, ann_pred)
ann_prec = precision_score(y_test, ann_pred)
ann_rec = recall_score(y_test, ann_pred)
ann_f1 = f1_score(y_test, ann_pred)
ann_auc = roc_auc_score(y_test, ann_prob)

print(f"\n✅ ANN Training Selesai!")
print(f"\n📊 Metrik ANN:")
print(f"   Akurasi  : {ann_acc:.4f}")
print(f"   Presisi  : {ann_prec:.4f}")
print(f"   Recall   : {ann_rec:.4f}")
print(f"   F1-Score : {ann_f1:.4f}")
print(f"   ROC-AUC  : {ann_auc:.4f}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('ANN - Loss', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['accuracy'], label='Training Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_title('ANN - Accuracy', fontsize=14)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# 15. RINGKASAN PERBANDINGAN MODEL
# ============================================================
print("\n" + "="*60)
print("📊 RINGKASAN PERBANDINGAN KETIGA MODEL")
print("="*60)

comparison_data = {
    'Model': ['SVM', 'Random Forest', 'ANN'],
    'Accuracy': [svm_acc, rf_acc, ann_acc],
    'Precision': [svm_prec, rf_prec, ann_prec],
    'Recall': [svm_rec, rf_rec, ann_rec],
    'F1-Score': [svm_f1, rf_f1, ann_f1],
    'ROC-AUC': [svm_auc, rf_auc, ann_auc]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.round(4).to_string(index=False))

# Visualisasi perbandingan
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart metrics
comparison_df_melted = comparison_df.melt(id_vars='Model',
                                          value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
sns.barplot(x='Model', y='value', hue='variable', data=comparison_df_melted, ax=axes[0])
axes[0].set_title('Perbandingan Metrik Klasifikasi', fontsize=14)
axes[0].set_ylabel('Skor')
axes[0].set_ylim(0.85, 1.0)
axes[0].legend(loc='lower right')
axes[0].grid(axis='y', alpha=0.3)

# ROC Curves
def plot_roc_curve(y_test, y_prob, label, color, ax):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f'{label} (AUC = {auc:.4f})', color=color, linewidth=2)

colors = ['#2e86ab', '#28a745', '#764ba2']
models_roc = [
    ('SVM', svm_prob, colors[0]),
    ('Random Forest', rf_prob, colors[1]),
    ('ANN', ann_prob, colors[2])
]

for name, prob, color in models_roc:
    plot_roc_curve(y_test, prob, name, color, axes[1])

axes[1].plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)', linewidth=1)
axes[1].set_title('ROC Curves Comparison', fontsize=14)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 16. CONFUSION MATRIX
# ============================================================
print("\n" + "="*60)
print("📊 CONFUSION MATRIX")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models_cm = [
    ('SVM', svm_pred, svm_acc),
    ('Random Forest', rf_pred, rf_acc),
    ('ANN', ann_pred, ann_acc)
]

for idx, (name, pred, acc) in enumerate(models_cm):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                ax=axes[idx])
    axes[idx].set_title(f'{name}\nAccuracy: {acc:.4f}', fontsize=14)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# ============================================================
# 17. SIMPAN SEMUA MODEL
# ============================================================
print("\n" + "="*60)
print("💾 MENYIMPAN MODEL")
print("="*60)

# Simpan model
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
ann_model.save('ann_model.h5')
joblib.dump(scaler, 'scaler.pkl')

# Simpan feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("\n✅ Semua model berhasil disimpan!")
print("\n📁 File yang disimpan:")
print("   1. svm_model.pkl")
print("   2. rf_model.pkl")
print("   3. ann_model.h5")
print("   4. scaler.pkl")
print("   5. feature_names.pkl")

# ============================================================
# 18. CEK UKURAN FILE
# ============================================================
print("\n" + "="*60)
print("📊 UKURAN FILE")
print("="*60)

file_list = ['svm_model.pkl', 'rf_model.pkl', 'ann_model.h5', 'scaler.pkl', 'feature_names.pkl']
for f in file_list:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"   {f}: {size:.1f} KB")

# ============================================================
# 19. TEST PREDIKSI DENGAN DATA SAMPLE
# ============================================================
print("\n" + "="*60)
print("🧪 TEST PREDIKSI DENGAN DATA SAMPLE")
print("="*60)

# Ambil 1 data dari test set
sample_idx = 0
sample_data = X_test.iloc[[sample_idx]]
sample_scaled = scaler.transform(sample_data)
sample_true = y_test.iloc[sample_idx]

print(f"\n📊 Data sample ke-{sample_idx+1}:")
print(f"   Target sebenarnya: {'Malignant (Ganas)' if sample_true == 1 else 'Benign (Jinak)'}")

# Prediksi dari semua model
svm_pred_sample = svm_model.predict(sample_scaled)[0]
svm_prob_sample = svm_model.predict_proba(sample_scaled)[0][1]

rf_pred_sample = rf_model.predict(sample_data)[0]
rf_prob_sample = rf_model.predict_proba(sample_data)[0][1]

ann_prob_sample = ann_model.predict(sample_scaled, verbose=0)[0][0]
ann_pred_sample = 1 if ann_prob_sample > 0.5 else 0

print(f"\n📊 Hasil Prediksi:")
print(f"   SVM            : {'Malignant' if svm_pred_sample == 1 else 'Benign'} (Prob: {svm_prob_sample:.4f})")
print(f"   Random Forest  : {'Malignant' if rf_pred_sample == 1 else 'Benign'} (Prob: {rf_prob_sample:.4f})")
print(f"   ANN            : {'Malignant' if ann_pred_sample == 1 else 'Benign'} (Prob: {ann_prob_sample:.4f})")

# ============================================================
# 20. DOWNLOAD SEMUA FILE
# ============================================================
print("\n" + "="*60)
print("📥 MENDOWNLOAD FILE")
print("="*60)

print("\n⚠️ File akan di-download satu per satu ke komputer Anda")
print("   (Siapkan folder untuk menyimpan file-file ini)")

for f in file_list:
    if os.path.exists(f):
        files.download(f)
        print(f"✅ {f} berhasil di-download")

print("\n" + "="*60)
print("✅ SEMUA PROSES SELESAI!")
print("="*60)

print("\n📌 FILE YANG HARUS DI-UPLOAD KE HUGGING FACE:")
print("   1. svm_model.pkl")
print("   2. rf_model.pkl")
print("   3. ann_model.h5")
print("   4. scaler.pkl")
print("   5. feature_names.pkl")
print("\n📌 Juga upload file-file berikut:")
print("   6. Dockerfile")
print("   7. requirements.txt")
print("   8. app.py")
print("   9. README.md")
print("\n🚀 SELAMAT, TUAN! APLIKASI SIAP DI-DEPLOY!")
print("="*60)
