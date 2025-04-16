import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load model
@st.cache_resource
def load_garbage_model():
    model = load_model('model_garbage_classification.h5')
    return model

model = load_garbage_model()

# Kelas dan label mapping
label_mapping = {
    "battery": "Baterai",
    "biological": "Sampah Biologis",
    "brown-glass": "Kaca Coklat",
    "cardboard": "Kardus",
    "clothes": "Kain/Pakaian",
    "green-glass": "Kaca Hijau",
    "metal": "Logam",
    "paper": "Kertas",
    "plastic": "Plastik",
    "shoes": "Sepatu",
    "trash": "Sampah Campuran",
    "white-glass": "Kaca Putih"
}

kategori_organik = ["Sampah Biologis", "Kertas", "Kardus"]
kategori_non_organik = ["Baterai", "Kaca Coklat", "Kaca Hijau", "Kaca Putih", "Plastik", "Logam", "Kain/Pakaian", "Sepatu", "Sampah Campuran"]

deskripsi_sampah = {
    "Baterai": "Mengandung bahan kimia berbahaya dan harus dibuang di tempat khusus.",
    "Sampah Biologis": "Sampah organik seperti sisa makanan, daun, dan bahan alami lainnya.",
    "Kaca Coklat": "Jenis kaca yang biasanya digunakan untuk botol bir atau minuman lainnya.",
    "Kardus": "Bahan daur ulang yang umum digunakan untuk kemasan dan harus dibuang ke tempat daur ulang.",
    "Kain/Pakaian": "Bisa didaur ulang atau disumbangkan jika masih layak pakai.",
    "Kaca Hijau": "Biasanya digunakan untuk botol kaca hijau seperti botol minuman.",
    "Logam": "Bisa didaur ulang menjadi berbagai produk baru seperti kaleng atau bagian kendaraan.",
    "Kertas": "Dapat didaur ulang menjadi kertas baru atau produk berbasis kertas lainnya.",
    "Plastik": "Bisa didaur ulang, tetapi butuh waktu lama untuk terurai di lingkungan.",
    "Sepatu": "Biasanya terbuat dari campuran bahan yang sulit didaur ulang, bisa disumbangkan jika masih layak pakai.",
    "Sampah Campuran": "Sampah yang tidak dapat dikategorikan ke dalam kelompok spesifik.",
    "Kaca Putih": "Kaca bening yang sering digunakan dalam kemasan makanan dan minuman."
}

class_names = list(label_mapping.keys())

# UI Streamlit
st.title("♻️ Deteksi Jenis Sampah Organik Dan Non Organik")
st.write("Upload gambar sampah untuk mendeteksi jenisnya dan dapatkan informasi detail.")

uploaded_file = st.file_uploader("Unggah gambar sampah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)

    # Proses gambar
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_prob = np.max(predictions)

    THRESHOLD = 0.7

    if predicted_prob < THRESHOLD:
        st.warning("🔍 Gambar tidak dikenali dalam dataset.")
    else:
        kelas_inggris = class_names[predicted_index]
        kelas_indonesia = label_mapping.get(kelas_inggris, "Tidak Diketahui")
        kategori = "Organik" if kelas_indonesia in kategori_organik else "Non-Organik"
        deskripsi = deskripsi_sampah.get(kelas_indonesia, "Tidak ada deskripsi.")

        st.success(f"✅ Prediksi: **{kelas_indonesia}** (Prob: {predicted_prob:.2f})")
        st.info(f"🗑️ Kategori: {kategori}")
        st.markdown(f"📄 **Deskripsi:** {deskripsi}")
