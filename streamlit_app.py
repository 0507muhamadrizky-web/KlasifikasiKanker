import streamlit as st

st.set_page_config(page_title="Test App", layout="wide")

st.title("✅ TEST APLIKASI BERJALAN")
st.write("Jika Anda melihat ini, Streamlit berhasil berjalan!")

st.info("Sekarang kita akan tambahkan model satu per satu.")

# Cek folder models
import os
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if os.path.exists(models_dir):
    st.success(f"✅ Folder 'models' ditemukan di: {models_dir}")
    files = os.listdir(models_dir)
    st.write(f"File di dalam folder: {files}")
else:
    st.error(f"❌ Folder 'models' TIDAK ditemukan di: {models_dir}")
    st.write("Buat folder 'models' dan upload file model di dalamnya.")
