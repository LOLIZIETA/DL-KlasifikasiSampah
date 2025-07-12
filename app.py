import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Fungsi untuk load model
@st.cache_resource
def load_garbage_model():
    model = load_model('model_garbage_classification.h5')
    return model

model = load_garbage_model()

# Label dan deskripsi
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
    "Baterai": "Baterai bekas tergolong limbah B3 (bahan berbahaya dan beracun). Mengandung logam berat seperti merkuri, timbal, dan kadmium yang dapat mencemari lingkungan jika dibuang sembarangan. Harus dibuang di tempat penampungan limbah elektronik.",
    "Sampah Biologis": "Termasuk sisa makanan, daun kering, atau bahan alami lainnya. Sampah ini mudah terurai dan cocok untuk dijadikan kompos guna menyuburkan tanah.",
    "Kaca Coklat": "Kaca ini sering digunakan sebagai botol minuman (seperti bir). Daur ulang kaca coklat membantu mengurangi konsumsi energi dan bahan baku dari alam.",
    "Kardus": "Merupakan bahan kemasan yang umum digunakan. Kardus dapat didaur ulang menjadi kertas baru. Harus dijaga tetap kering agar tidak rusak.",
    "Kain/Pakaian": "Pakaian lama dapat didaur ulang menjadi produk baru seperti kain pel, atau disumbangkan untuk yang membutuhkan. Jangan langsung dibuang jika masih layak pakai.",
    "Kaca Hijau": "Umumnya ditemukan sebagai botol minuman. Seperti jenis kaca lainnya, kaca hijau bisa didaur ulang tanpa kehilangan kualitasnya.",
    "Logam": "Sampah logam seperti kaleng atau besi bisa dilebur dan dibentuk ulang menjadi alat baru. Daur ulang logam menghemat energi dan sumber daya alam.",
    "Kertas": "Kertas bekas bisa didaur ulang menjadi tisu, karton, atau kertas cetak ulang. Namun, pastikan tidak tercampur dengan minyak atau makanan agar tetap bisa didaur ulang.",
    "Plastik": "Plastik sangat sulit terurai dan menjadi ancaman besar bagi lingkungan. Daur ulang plastik dapat mengurangi limbah dan menjaga laut dari pencemaran.",
    "Sepatu": "Sampah sepatu biasanya terdiri dari campuran karet, kain, dan lem. Jika masih layak pakai, lebih baik disumbangkan. Daur ulang sepatu lebih kompleks namun tetap memungkinkan.",
    "Sampah Campuran": "Merupakan kombinasi dari beberapa jenis sampah atau jenis yang tidak dapat dikategorikan. Biasanya sulit untuk dipilah atau didaur ulang secara efisien.",
    "Kaca Putih": "Kaca bening seperti botol atau wadah makanan. Mudah didaur ulang dan bisa diproses menjadi produk kaca baru yang setara kualitasnya."
}

class_names = list(label_mapping.keys())

# Navigasi Sidebar
st.sidebar.title("📌 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Klasifikasi Sampah", "Tentang"])

# Halaman Beranda
if page == "Beranda":
    st.markdown("<h1 style='text-align: center; color: green;'>♻️ Website Deteksi Sampah</h1>", unsafe_allow_html=True)
    st.markdown("### Selamat datang!")
    st.write("Website ini bisa digunakan untuk mengenali jenis sampah dari gambar dan memberi informasi penting seperti:")
    st.markdown("- ✅ Jenis sampah (organik / anorganik)")
    st.markdown("- 📄 Penjelasan kategori sampah")
    st.markdown("- 🧠 Edukasi singkat tentang daur ulang")
    st.image("https://media.istockphoto.com/id/1200963979/id/vektor/ilustrasi-vektor-konsep-daur-ulang-desain-modern-datar-untuk-halaman-web-spanduk-presentasi.jpg?s=612x612&w=0&k=20&c=l8xOrP-TCcQnNeUaixJ04yEGaqyLXMn9aDhHL9hG5JI=", caption="Buanglah sampah pada tempatnya", use_container_width=True)

    st.markdown("### 📷 Contoh Gambar")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://static9.depositphotos.com/1000261/1129/i/450/depositphotos_11293200-stock-photo-waste-cardboard.jpg", caption="Kardus")
    with col2:
        st.image("https://www.dbs.com/spark/index/id_id/site/img/pillars/89/89.jpg", caption="Plastik")
    with col3:
        st.image("https://mmc.tirto.id/image/2019/02/04/ilustrasi-baterai-istockphoto_ratio-16x9.jpg", caption="Baterai")

    st.markdown("### ❓ Apa itu sampah organik dan Anorganik?")
    st.write("""
        **Sampah organik** adalah sampah yang berasal dari makhluk hidup dan bisa terurai secara alami, seperti daun, sisa makanan, atau kertas.

        **Sampah Anorganik** berasal dari benda tak hidup dan sulit terurai, seperti plastik, kaca, logam, dan baterai.
        """)

    st.markdown("---")
    st.markdown("### 🎬 Video Panduan Menggunakan Prediksi Sampah")
    st.write("Tonton video berikut untuk memahami cara menggunakan fitur prediksi sampah pada website ini.")
    st.video("https://youtu.be/lWr8yTJ439s?si=7VTHj3eIxdof-fIR")

# Halaman Prediksi
elif page == "Klasifikasi Sampah":
    st.markdown("## 🧪 Klasifikasi Jenis Sampah Berdasarkan Gambar")
    st.write("Upload gambar sampah untuk mengetahui jenis dan penjelasannya.")

    uploaded_file = st.file_uploader("Unggah gambar sampah...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Gambar yang Diunggah", use_container_width=True)

        try:
            img = image_pil.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions)
            predicted_prob = np.max(predictions)
            persentase = predicted_prob * 100

            THRESHOLD = 0.7

            if predicted_prob < THRESHOLD:
                st.warning("🔍 Gambar tidak dikenali. Pastikan gambar menampilkan satu jenis sampah dengan jelas dan memiliki kualitas yang baik.")
                st.info("📌 Tips: Gunakan gambar dengan pencahayaan yang cukup dan latar belakang yang bersih.")
                st.markdown("🤔 Gambar Anda tidak dikenali? Anda bisa memberi masukan ke kami agar model terus ditingkatkan.")
            else:
                kelas_inggris = class_names[predicted_index]
                kelas_indonesia = label_mapping.get(kelas_inggris, "Tidak Diketahui")
                kategori = "Organik" if kelas_indonesia in kategori_organik else "Anorganik"
                deskripsi = deskripsi_sampah.get(kelas_indonesia, "Tidak ada deskripsi.")

                st.success(f"✅ Jenis Sampah: **{kelas_indonesia}** ({persentase:.2f}%)")
                st.info(f"🗑️ Kategori: {kategori}")
                st.markdown(f"📄 **Deskripsi:** {deskripsi}")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

# Halaman Tentang
elif page == "Tentang":
    st.markdown("## ℹ️ Tentang Website")
    st.write("""
    Website ini menggunakan model deep learning berbasis Convolutional Neural Network (CNN) untuk mengenali jenis sampah dari gambar.

    Model telah dilatih menggunakan dataset dari berbagai kategori sampah, baik organik maupun Anorganik.

    **Tujuan**:
    - Meningkatkan kesadaran memilah sampah
    - Membantu masyarakat dalam edukasi daur ulang
    - Mengurangi pencemaran dengan pemilahan yang benar
    """)
    st.markdown("🧑‍💻 Dibuat oleh: **Anugerah Bakti Prasisto**")

# Footer
st.markdown("---")
st.markdown("<center>© 2025 - Website Deteksi Sampah oleh Anugerah Bakti Prasisto</center>", unsafe_allow_html=True)
