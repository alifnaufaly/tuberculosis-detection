import streamlit as st
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# 1. Konfigurasi Halaman Web
st.set_page_config(
    page_title="Deteksi TBC Paru-Paru",
    page_icon="🫁",
    layout="centered"
)

# 2. Fungsi Load Model + Scaler + LabelEncoder (pakai PKL)
@st.cache_resource
def load_artifacts():
    ann_model_path = "model_tbc_ann.h5"
    rf_model_path = "model_tbc_rf.pkl"
    scaler_path = "scaler_tbc.pkl"
    le_path = "labelencoder_tbc.pkl"

    missing_files = []
    if not os.path.exists(ann_model_path):
        missing_files.append(ann_model_path)
    if not os.path.exists(rf_model_path):
        missing_files.append(rf_model_path)
    if not os.path.exists(scaler_path):
        missing_files.append(scaler_path)
    if not os.path.exists(le_path):
        missing_files.append(le_path)

    if missing_files:
        return None, None, None, None, missing_files

    # Load model, scaler, dan label encoder
    ann_model = tf.keras.models.load_model(ann_model_path)
    rf_model = joblib.load(rf_model_path)
    scaler = joblib.load(scaler_path)
    labelencoder = joblib.load(le_path)

    return ann_model, rf_model, scaler, labelencoder, []

# Memanggil fungsi load artifacts
ann_model, rf_model, scaler, labelencoder, missing_files = load_artifacts()

# 3. Tampilan Judul
st.title("🫁 Prediksi Penyakit TBC Paru")
st.write("Aplikasi berbasis Deep Learning (ANN) dan Random Forest untuk mendeteksi risiko TBC.")
st.divider()

# Jika ada file yang hilang, tampilkan pesan error dan hentikan app
if ann_model is None or rf_model is None or scaler is None or labelencoder is None:
    st.error("⚠️ FILE MODEL / SCALER / LABELENCODER TIDAK DITEMUKAN!")
    st.warning(
        "Pastikan file berikut ada di folder yang sama dengan file 'app.py':\n"
        + "\n".join(f"- {f}" for f in missing_files)
    )
    st.stop()

# 4. Sidebar yang rapi
with st.sidebar:
    st.markdown("## 🫁 TBC Detector")
    st.markdown("---")
    menu = st.radio(
        "",
        ["Prediksi", "Evaluasi Model"],
        index=0
    )
    st.markdown("---")
    st.caption("Mini Project • Deteksi TBC Paru")

# Daftar nama fitur (untuk evaluasi & kesesuaian dataset)
FEATURE_COLS = ["CO", "NS", "BD", "FV", "CP", "SP", "IS", "LP",
                "CH", "LC", "IR", "LA", "LE", "LN", "SB", "BMI"]


# ============================
# HALAMAN 1: MENU PREDIKSI
# ============================
if menu == "Prediksi":
    st.subheader("📋 Masukkan Data Gejala")

    # Penjelasan umum (dekat judul)
    st.markdown(
        """
        Menu ini digunakan untuk **memprediksi risiko TBC paru** berdasarkan gejala klinis pasien.

        **Cara penggunaan:**
        1. Pilih terlebih dahulu **model prediksi**:
           - 🧠 **ANN (Deep Learning)** : model jaringan saraf tiruan yang mampu mempelajari pola kompleks.
           - 🌲 **Random Forest** : model _ensemble_ yang menggabungkan banyak decision tree.
        2. Isi setiap gejala sesuai kondisi pasien saat ini.
        3. Klik tombol **"Analisa Risiko"** untuk menjalankan model.
        4. Hasil akan menampilkan status **Positif / Negatif TBC** beserta **persentase probabilitas**.

        > ⚠️ **Catatan penting:**  
        > Hasil prediksi ini **bukan diagnosis medis**.  
        > Keputusan akhir tetap harus dikonfirmasi oleh **dokter atau tenaga kesehatan**.
        """
    )

    # Penjelasan khusus skala 0–3
    st.markdown(
        """
        **Skala Penilaian Gejala (0–3):**

        - 🟢 **0 = Tidak ada gejala**  
        - 🟡 **1 = Gejala ringan**  
        - 🟠 **2 = Gejala sedang**  
        - 🔴 **3 = Gejala berat**

        Gunakan skala ini saat memilih nilai untuk setiap gejala di bawah.
        """
    )

    # Pilih model yang mau dipakai
    model_choice = st.radio(
        "Pilih Model Prediksi:",
        ["ANN (Deep Learning)", "Random Forest"],
        horizontal=True
    )

    # Form input gejala
    col1, col2 = st.columns(2)

    with col1:
        co = st.selectbox(
            "🗣️ Batuk (CO)",
            [0, 1, 2, 3],
            help="0 = Tidak batuk, 1 = Batuk ringan, 2 = Batuk sedang, 3 = Batuk berat"
        )
        ns = st.selectbox(
            "💧 Keringat Malam (NS)",
            [0, 1, 2],
            help="0 = Tidak berkeringat malam, 1 = Kadang, 2 = Sering / berat"
        )
        bd = st.selectbox(
            "😮‍💨 Sesak Napas (BD)",
            [0, 1, 2],
            help="0 = Tidak sesak, 1 = Sesak ringan, 2 = Sesak berat"
        )
        fv = st.selectbox(
            "🌡️ Demam (FV)",
            [0, 1, 2],
            help="0 = Tidak demam, 1 = Demam ringan, 2 = Demam tinggi"
        )
        cp = st.selectbox(
            "❤️ Nyeri Dada (CP)",
            [0, 1, 2],
            help="0 = Tidak nyeri, 1 = Nyeri ringan, 2 = Nyeri berat"
        )
        sp = st.selectbox(
            "🫧 Produksi Dahak (SP)",
            [0, 1, 2],
            help="0 = Tidak ada dahak, 1 = Dahak sedikit, 2 = Dahak banyak"
        )
        jis = st.selectbox(
            "🍽️ Nafsu Makan (IS)",
            [0, 1, 2, 3],
            help="0 = Normal, 1 = Sedikit turun, 2 = Turun sedang, 3 = Sangat turun"
        )
        lp = st.selectbox(
            "🦵 Nyeri Sendi (LP)",
            [0, 1, 2],
            help="0 = Tidak nyeri, 1 = Nyeri ringan, 2 = Nyeri berat"
        )

    with col2:
        ch = st.selectbox(
            "🥶 Menggigil (CH)",
            [0, 1, 2],
            help="0 = Tidak menggigil, 1 = Menggigil ringan, 2 = Menggigil berat"
        )
        lc = st.selectbox(
            "⚡ Kelelahan (LC)",
            [0, 1, 2],
            help="0 = Tidak lelah, 1 = Lelah ringan, 2 = Lelah berat"
        )
        ir = st.selectbox(
            "😠 Iritabilitas (IR)",
            [0, 1, 2],
            help="0 = Tidak mudah marah, 1 = Mudah marah, 2 = Sangat mudah marah"
        )
        la = st.selectbox(
            "😓 Lesu (LA)",
            [0, 1, 2],
            help="0 = Tidak lesu, 1 = Lesu ringan, 2 = Lesu berat"
        )
        le = st.selectbox(
            "🦠 Pembengkakan Kelenjar (LE)",
            [0, 1, 2, 3],
            help="0 = Tidak bengkak, 1–3 = tingkat pembengkakan dari ringan sampai berat"
        )
        ln = st.selectbox(
            "⚖️ Penurunan Berat Badan (LN)",
            [0, 1, 2],
            help="0 = Tidak turun BB, 1 = Turun ringan, 2 = Turun signifikan"
        )
        sb = st.selectbox(
            "🩸 Batuk Darah (SB)",
            [0, 1, 2],
            help="0 = Tidak ada darah, 1 = Sedikit darah, 2 = Darah cukup banyak"
        )
        bmi = st.selectbox(
            "📊 BMI (Body Mass Index)",
            [0, 1, 2],
            help="0 = Normal, 1 = Risiko gizi kurang/lebih ringan, 2 = Risiko lebih berat"
        )

    # Tombol Prediksi
    if st.button("🔍 Analisa Risiko", type="primary"):
        input_data = np.array([[co, ns, bd, fv, cp, sp, jis, lp,
                                ch, lc, ir, la, le, ln, sb, bmi]])

        input_data_scaled = scaler.transform(input_data)

        if model_choice.startswith("ANN"):
            prediksi_raw = ann_model.predict(input_data_scaled)
            probabilitas = float(prediksi_raw[0][0])
            pred_label_int = int(probabilitas > 0.5)
            nama_model = "ANN (Deep Learning)"
        else:
            rf_pred = rf_model.predict(input_data_scaled)
            pred_label_int = int(rf_pred[0])

            if hasattr(rf_model, "predict_proba"):
                prob_rf = rf_model.predict_proba(input_data_scaled)[0][1]
                probabilitas = float(prob_rf)
            else:
                probabilitas = float(pred_label_int)
            nama_model = "Random Forest"

        pred_label_text = labelencoder.inverse_transform([pred_label_int])[0]

        st.divider()
        st.subheader(f"Hasil Diagnosa AI ({nama_model}):")

        if pred_label_text == "Ya":
            st.error(f"🚨 POSITIF TBC (Probabilitas: {probabilitas*100:.1f}%)")
            st.write("Saran: Segera lakukan pemeriksaan lebih lanjut ke fasilitas kesehatan atau dokter spesialis paru.")
        else:
            st.success(f"✅ NEGATIF TBC (Probabilitas TBC: {probabilitas*100:.1f}%)")
            st.write("Saran: Tetap jaga kesehatan dan pantau gejala. Lakukan pemeriksaan bila keluhan muncul atau memburuk.")


# ============================
# HALAMAN 2: MENU EVALUASI MODEL
# ============================
elif menu == "Evaluasi Model":
    st.subheader("📊 Evaluasi & Perbandingan Kinerja Model")

    st.markdown(
        """
        Menu ini digunakan untuk **mengevaluasi performa dua model** yang digunakan pada aplikasi, yaitu:
        - **ANN (Deep Learning)**
        - **Random Forest**

        Dengan evaluasi ini, kita dapat melihat:
        - Seberapa sering model memberikan prediksi yang **benar** (akurasi),
        - Seberapa tepat model saat menyatakan **pasien TBC** (presisi),
        - Seberapa banyak kasus TBC yang berhasil **terdeteksi** oleh model (recall),
        - Keseimbangan antara presisi dan recall (F1-Score),
        - Serta **Confusion Matrix** yang menggambarkan perbandingan label asli dan hasil prediksi.

        **Cara penggunaan:**
        1. Siapkan **dataset uji** yang sudah memiliki kolom label/target (misalnya kolom `Prediksi` berisi 0 = Tidak TBC, 1 = TBC atau 'Tidak' / 'Ya').
        2. Upload file CSV tersebut pada form di bawah.
        3. Pilih kolom yang berfungsi sebagai **label/target**.
        4. Klik tombol **"Hitung Evaluasi Model"** untuk melihat hasil perbandingan antara ANN dan Random Forest.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload Dataset Uji (format .csv)",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=";")

        st.write("Preview Dataset Uji:")
        st.dataframe(df.head())

        st.write(f"Jumlah baris: **{df.shape[0]}**, Jumlah kolom: **{df.shape[1]}**")

        # Pilih kolom label/target
        if "Prediksi" in df.columns:
            default_index = list(df.columns).index("Prediksi")
        else:
            default_index = 0

        target_col = st.selectbox(
            "Pilih kolom label/target (0 = Tidak TBC, 1 = TBC):",
            df.columns,
            index=default_index
        )

        st.markdown(
            """
            Setelah memilih kolom target, klik tombol di bawah untuk menjalankan evaluasi.  
            Hasil evaluasi akan menampilkan metrik untuk **masing-masing model** serta perbandingannya.
            """
        )

        if st.button("📊 Hitung Evaluasi Model"):
            try:
                # Pisahkan fitur & label
                X_df = df.drop(columns=[target_col])
                y_true = df[target_col]

                if y_true.dtype == "O":
                    y_true = labelencoder.transform(y_true)
                else:
                    y_true = y_true.values

                if all(col in X_df.columns for col in FEATURE_COLS):
                    X_df = X_df[FEATURE_COLS]

                X = X_df.astype(float).values

                if X.shape[1] != 16:
                    st.error(f"Model mengharapkan 16 fitur, tapi dataset punya {X.shape[1]} fitur.")
                    st.stop()

                X_scaled = scaler.transform(X)

                # Prediksi dua model
                y_prob_ann = ann_model.predict(X_scaled).reshape(-1)
                y_pred_ann = (y_prob_ann > 0.5).astype(int)

                y_pred_rf = rf_model.predict(X_scaled)

                def compute_metrics(y_true_, y_pred_):
                    acc_ = accuracy_score(y_true_, y_pred_)
                    prec_ = precision_score(y_true_, y_pred_, zero_division=0)
                    rec_ = recall_score(y_true_, y_pred_, zero_division=0)
                    f1_ = f1_score(y_true_, y_pred_, zero_division=0)
                    return acc_, prec_, rec_, f1_

                acc_ann, prec_ann, rec_ann, f1_ann = compute_metrics(y_true, y_pred_ann)
                acc_rf,  prec_rf,  rec_rf,  f1_rf  = compute_metrics(y_true, y_pred_rf)

                jumlah_data = len(y_true)

                st.divider()
                st.write("### Ringkasan Metrik (Perbandingan ANN vs Random Forest)")

                # Penjelasan metrik singkat
                with st.expander("ℹ️ Penjelasan Metrik Evaluasi"):
                    st.markdown(
                        """
                        - **Akurasi** : Persentase seluruh prediksi yang **benar** dari semua data uji.  
                        - **Presisi** : Dari semua pasien yang **diprediksi TBC**, berapa persen yang **benar-benar TBC**.  
                        - **Recall (Sensitivitas)** : Dari semua pasien yang **benar-benar TBC**, berapa persen yang berhasil **terdeteksi model**.  
                        - **F1-Score** : Rata-rata harmonis antara Presisi dan Recall. Cocok untuk data yang tidak seimbang.  
                        """
                    )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Jumlah Data Uji", jumlah_data)
                with col2:
                    st.metric("Akurasi ANN", f"{acc_ann:.2%}")
                    st.metric("Presisi ANN", f"{prec_ann:.2%}")
                with col3:
                    st.metric("Akurasi RF", f"{acc_rf:.2%}")
                    st.metric("Presisi RF", f"{prec_rf:.2%}")

                # Tabel metrik perbandingan
                st.write("### Tabel Metrik Per Model")
                df_metrics_compare = pd.DataFrame({
                    "Metrik": ["Akurasi", "Presisi", "Recall", "F1-Score"],
                    "ANN (Deep Learning)": [acc_ann, prec_ann, rec_ann, f1_ann],
                    "Random Forest": [acc_rf, prec_rf, rec_rf, f1_rf],
                })
                st.table(df_metrics_compare.style.format({
                    "ANN (Deep Learning)": "{:.4f}",
                    "Random Forest": "{:.4f}",
                }))

                # Plot visual perbandingan metrik
                st.write("### Visualisasi Perbandingan Metrik")
                df_plot = pd.DataFrame({
                    "ANN (Deep Learning)": [acc_ann, prec_ann, rec_ann, f1_ann],
                    "Random Forest": [acc_rf, prec_rf, rec_rf, f1_rf],
                }, index=["Akurasi", "Presisi", "Recall", "F1-Score"])
                st.bar_chart(df_plot)

                st.write(
                    """
                    Grafik di atas menunjukkan perbandingan setiap metrik antara model ANN dan Random Forest.  
                    Semakin tinggi batang pada suatu metrik, semakin baik performa model pada aspek tersebut.
                    """
                )

                # Confusion Matrix
                st.write("### Confusion Matrix")

                cm_ann = confusion_matrix(y_true, y_pred_ann)
                cm_rf = confusion_matrix(y_true, y_pred_rf)

                cm_index = ["Actual Tidak TBC (0)", "Actual TBC (1)"]
                cm_columns = ["Pred Tidak TBC (0)", "Pred TBC (1)"]

                col_cm1, col_cm2 = st.columns(2)
                with col_cm1:
                    st.markdown("**ANN (Deep Learning)**")
                    cm_ann_df = pd.DataFrame(cm_ann, index=cm_index, columns=cm_columns)
                    st.table(cm_ann_df)
                with col_cm2:
                    st.markdown("**Random Forest**")
                    cm_rf_df = pd.DataFrame(cm_rf, index=cm_index, columns=cm_columns)
                    st.table(cm_rf_df)

                st.markdown(
                    """
                    **Cara membaca confusion matrix (per model):**
                    - Baris = **label asli** (Actual), Kolom = **hasil prediksi** (Pred).  
                    - Nilai diagonal menunjukkan jumlah data yang diprediksi **benar**.  
                    - Nilai di luar diagonal menunjukkan jumlah data yang diprediksi **salah**.
                    """
                )

                # ================================
                # CLASSIFICATION REPORT – RAPI & ADA PENJELASAN
                # ================================
                st.write("## 📄 Classification Report")
                st.markdown("""
Bagian ini menunjukkan performa model ditinjau dari beberapa metrik evaluasi.

**Ringkasan metrik:**
- **Precision** → Dari semua prediksi *positif*, berapa banyak yang benar-benar positif  
- **Recall (Sensitivity)** → Dari semua data *positif asli*, berapa banyak yang berhasil dikenali model  
- **F1-Score** → Kombinasi Precision & Recall (semakin tinggi semakin baik)  
- **Support** → Jumlah data aktual per kelas  
""")

                # ANN report sebagai DataFrame
                st.subheader("🧠 ANN (Deep Learning) — Classification Report")
                report_ann_dict = classification_report(
                    y_true, y_pred_ann,
                    target_names=list(labelencoder.classes_),
                    zero_division=0,
                    output_dict=True
                )
                df_ann_report = pd.DataFrame(report_ann_dict).transpose()
                st.dataframe(df_ann_report.style.format("{:.2f}"))

                st.markdown("""
**Penjelasan:**
- Baris **'Tidak'** = performa model dalam mendeteksi pasien *tidak TBC*  
- Baris **'Ya'** = performa model dalam mendeteksi *pasien TBC*  
- **macro avg** = rata-rata sederhana dari semua kelas  
- **weighted avg** = rata-rata yang mempertimbangkan jumlah data tiap kelas  
""")

                # Random Forest report sebagai DataFrame
                st.subheader("🌲 Random Forest — Classification Report")
                report_rf_dict = classification_report(
                    y_true, y_pred_rf,
                    target_names=list(labelencoder.classes_),
                    zero_division=0,
                    output_dict=True
                )
                df_rf_report = pd.DataFrame(report_rf_dict).transpose()
                st.dataframe(df_rf_report.style.format("{:.2f}"))

                st.markdown("""
Model Random Forest sering memberikan performa yang stabil karena menggunakan banyak decision tree.

**Tips membaca:**
- Jika **Recall untuk kelas 'Ya'** tinggi → model sangat baik mendeteksi kasus TBC.  
- Jika **Precision untuk kelas 'Ya'** tinggi → model jarang salah menandai orang sehat sebagai TBC.  
- Perhatikan juga **accuracy**, **macro avg**, dan **weighted avg** sebagai gambaran umum performa model.  
""")

            except Exception as e:
                st.error("Terjadi error saat menghitung evaluasi.")
                st.exception(e)

    else:
        st.info("Silakan upload file CSV data uji terlebih dahulu untuk melihat evaluasi dan perbandingan model.")
