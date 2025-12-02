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
    st.caption("Deteksi TBC Paru • alifnaufalyr")

# Daftar nama fitur (untuk evaluasi & kesesuaian dataset)
FEATURE_COLS = ["CO", "NS", "BD", "FV", "CP", "SP", "IS", "LP",
                "CH", "LC", "IR", "LA", "LE", "LN", "SB", "BMI"]


# ============================
# UTIL: Membaca CSV dengan robust handling
# ============================
def safe_read_csv(path_or_buffer):
    """
    Membaca CSV dari path atau buffer dengan deteksi separator ',' atau ';'.
    Jika file terbaca sebagai 1 kolom (semua elemen digabung), lakukan split pada ';' dan
    beri header sementara.
    """
    # pertama coba read default (pandas akan auto-detect delimiter tetapi default sep=',')
    try:
        df = pd.read_csv(path_or_buffer, encoding='utf-8')
    except Exception:
        try:
            df = pd.read_csv(path_or_buffer, encoding='latin1')
        except Exception as e:
            raise e

    # Jika hanya 1 kolom dan nampaknya berisi separator semicolon, split manual
    if df.shape[1] == 1:
        # ambil nama kolom (jika ada header yang ikut terbaca, kita cek)
        single_col = df.columns[0]
        # jika isi baris ada ';', maka split
        if df.iloc[:, 0].astype(str).str.contains(";").any():
            splitted = df.iloc[:, 0].astype(str).str.split(";", expand=True)
            # Jika nampak header di baris pertama (mis. ada 'CO;NS;BD;...'), gunakan row pertama sebagai header
            first_row_vals = splitted.iloc[0].tolist()
            # Heuristik: jika first_row_vals banyak string non-numeric, treat as header
            non_numeric_count = sum([not str(x).strip().replace('.', '', 1).isdigit() for x in first_row_vals])
            if non_numeric_count >= len(first_row_vals) // 2:
                # treat first row as header
                splitted.columns = splitted.iloc[0]
                splitted = splitted.drop(index=0).reset_index(drop=True)
            else:
                # buat header generic, asumsi kolom terakhir label
                ncols = splitted.shape[1]
                col_names = [f"col{i}" for i in range(ncols-1)] + ["Prediksi"]
                splitted.columns = col_names
            df = splitted

    return df


def normalize_and_map_label(df, label_col_candidates=['Prediksi', 'label', 'target']):
    """
    Mencari kolom label (prioritas kandidat), normalisasi nilai label
    ('Ya'/'Tidak'/'Yes'/'No' -> 1/0) dan mengembalikan df serta nama kolom label.
    Jika label tidak ditemukan, asumsi kolom terakhir adalah label.
    """
    # Tentukan kolom label
    label_col = None
    for cand in label_col_candidates:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        # gunakan kolom terakhir
        label_col = df.columns[-1]

    # bersihkan nilai string
    s = df[label_col].astype(str).str.strip()
    # beberapa varian kapitalisasi
    s_lower = s.str.lower()

    mapping = {
        "ya": 1, "yes": 1, "1": 1,
        "tidak": 0, "no": 0, "0": 0
    }

    # apply mapping where possible
    s_mapped = s_lower.replace(mapping)

    # jika masih ada yang bukan 0/1 (mis. 'Positif', 'Negatif', atau angka lain), coba more heuristics
    def try_numeric(x):
        try:
            xi = float(x)
            # treat >=0.5 as 1, else 0 (only if values look like probabilities)
            return 1 if xi >= 0.5 else 0
        except Exception:
            return x

    s_mapped = s_mapped.apply(lambda x: try_numeric(x) if not pd.isnull(x) else x)

    # jika masih ada non 0/1, kita print warning dan drop baris yang tidak ter-map
    mask_valid = s_mapped.isin([0, 1])
    if not mask_valid.all():
        invalid_vals = s_mapped[~mask_valid].unique().tolist()
        st.warning(f"Ada nilai label yang tidak dikenali dan akan di-drop: {invalid_vals}")
        df = df[mask_valid].copy()
        s_mapped = s_mapped[mask_valid].copy()

    # assign kembali sebagai int
    df[label_col] = s_mapped.astype(int)
    return df, label_col


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
           - 🧠 **ANN (Deep Learning)** : model jaringan saraf tiruan.
           - 🌲 **Random Forest** : model _ensemble_.
        2. Isi setiap gejala sesuai kondisi pasien saat ini.
        3. Klik tombol **"Analisa Risiko"** untuk menjalankan model.
        4. Hasil akan menampilkan status **Positif / Negatif TBC** beserta **persentase probabilitas**.
        """
    )

    # Penjelasan khusus skala 0–3
    st.markdown(
        """
        **Skala Penilaian Gejala (0–3):**

        - 0 = Tidak ada gejala
        - 1 = Gejala ringan
        - 2 = Gejala sedang
        - 3 = Gejala berat
        """
    )

    # Pilih model yang mau dipakai
    model_choice = st.radio(
        "Pilih Model Prediksi:",
        ["ANN (Deep Learning)", "Random Forest"],
        horizontal=True
    )

    # Daftar penjelasan untuk setiap gejala
    symptoms_help = {
        "CO": "🗣️ Batuk",
        "NS": "💧 Keringat Malam",
        "BD": "😮‍💨 Sesak Napas",
        "FV": "🌡️ Demam",
        "CP": "❤️ Nyeri Dada",
        "SP": "🫧 Produksi Dahak",
        "IS": "🍽️ Nafsu Makan",
        "LP": "🦵 Nyeri Sendi",
        "CH": "🥶 Menggigil",
        "LC": "⚡ Kelelahan",
        "IR": "😠 Iritabilitas",
        "LA": "😓 Lesu",
        "LE": "🦠 Pembengkakan Kelenjar",
        "LN": "⚖️ Penurunan Berat Badan",
        "SB": "🩸 Batuk Darah",
        "BMI": "📊 BMI (Body Mass Index)"
    }
    
    symptoms_descriptions = {
        "CO": "**0**: Tidak ada batuk | **1**: Batuk ringan (sesekali) | **2**: Batuk sedang (sering) | **3**: Batuk berat (hampir terus-menerus)",
        "NS": "**0**: Tidak berkeringat di malam hari | **1**: Keringat ringan | **2**: Keringat berlebih di malam hari",
        "BD": "**0**: Pernapasan normal | **1**: Sesak napas ringan (saat aktivitas berat) | **2**: Sesak napas sedang-berat (saat aktivitas ringan)",
        "FV": "**0**: Suhu tubuh normal (≤36.5°C) | **1**: Demam ringan (36.5-38°C) | **2**: Demam tinggi (>38°C)",
        "CP": "**0**: Tidak ada nyeri | **1**: Nyeri ringan | **2**: Nyeri sedang-berat",
        "SP": "**0**: Tidak ada produksi dahak | **1**: Produksi dahak ringan | **2**: Produksi dahak berlebih",
        "IS": "**0**: Nafsu makan sangat menurun/tidak makan | **1**: Nafsu makan berkurang | **2**: Nafsu makan agak menurun | **3**: Nafsu makan normal",
        "LP": "**0**: Tidak ada nyeri sendi | **1**: Nyeri ringan | **2**: Nyeri sedang-berat",
        "CH": "**0**: Tidak menggigil | **1**: Menggigil ringan | **2**: Menggigil sering",
        "LC": "**0**: Tidak lelah | **1**: Lelah ringan | **2**: Lelah berkepanjangan",
        "IR": "**0**: Tidak mudah marah | **1**: Mudah marah ringan | **2**: Mudah marah/hostile",
        "LA": "**0**: Tidak lesu/normal | **1**: Lesu ringan | **2**: Lesu berat (malas beraktivitas)",
        "LE": "**0**: Tidak ada pembengkakan | **1**: Pembengkakan ringan | **2**: Pembengkakan sedang | **3**: Pembengkakan berat",
        "LN": "**0**: Berat badan stabil | **1**: Penurunan ringan (1-3 kg/bulan) | **2**: Penurunan berat (>3 kg/bulan)",
        "SB": "**0**: Tidak ada batuk darah | **1**: Batuk darah ringan (sedikit) | **2**: Batuk darah berat (banyak)",
        "BMI": "**0**: BMI <18.5 (kurus) atau >30 (gemuk) | **1**: BMI 18.5-24.9 (normal) | **2**: BMI 25-30 (overweight)"
    }

    # Form input gejala dengan penjelasan
    col1, col2 = st.columns(2)

    with col1:
        st.caption("**Kolom Kiri**")
        with st.expander(symptoms_help["CO"]):
            st.info(symptoms_descriptions["CO"])
        co = st.selectbox("🗣️ Batuk (CO)", [0, 1, 2, 3], key="co_select")
        
        with st.expander(symptoms_help["NS"]):
            st.info(symptoms_descriptions["NS"])
        ns = st.selectbox("💧 Keringat Malam (NS)", [0, 1, 2], key="ns_select")
        
        with st.expander(symptoms_help["BD"]):
            st.info(symptoms_descriptions["BD"])
        bd = st.selectbox("😮‍💨 Sesak Napas (BD)", [0, 1, 2], key="bd_select")
        
        with st.expander(symptoms_help["FV"]):
            st.info(symptoms_descriptions["FV"])
        fv = st.selectbox("🌡️ Demam (FV)", [0, 1, 2], key="fv_select")
        
        with st.expander(symptoms_help["CP"]):
            st.info(symptoms_descriptions["CP"])
        cp = st.selectbox("❤️ Nyeri Dada (CP)", [0, 1, 2], key="cp_select")
        
        with st.expander(symptoms_help["SP"]):
            st.info(symptoms_descriptions["SP"])
        sp = st.selectbox("🫧 Produksi Dahak (SP)", [0, 1, 2], key="sp_select")
        
        with st.expander(symptoms_help["IS"]):
            st.info(symptoms_descriptions["IS"])
        jis = st.selectbox("🍽️ Nafsu Makan (IS)", [0, 1, 2, 3], key="is_select")
        
        with st.expander(symptoms_help["LP"]):
            st.info(symptoms_descriptions["LP"])
        lp = st.selectbox("🦵 Nyeri Sendi (LP)", [0, 1, 2], key="lp_select")

    with col2:
        st.caption("**Kolom Kanan**")
        with st.expander(symptoms_help["CH"]):
            st.info(symptoms_descriptions["CH"])
        ch = st.selectbox("🥶 Menggigil (CH)", [0, 1, 2], key="ch_select")
        
        with st.expander(symptoms_help["LC"]):
            st.info(symptoms_descriptions["LC"])
        lc = st.selectbox("⚡ Kelelahan (LC)", [0, 1, 2], key="lc_select")
        
        with st.expander(symptoms_help["IR"]):
            st.info(symptoms_descriptions["IR"])
        ir = st.selectbox("😠 Iritabilitas (IR)", [0, 1, 2], key="ir_select")
        
        with st.expander(symptoms_help["LA"]):
            st.info(symptoms_descriptions["LA"])
        la = st.selectbox("😓 Lesu (LA)", [0, 1, 2], key="la_select")
        
        with st.expander(symptoms_help["LE"]):
            st.info(symptoms_descriptions["LE"])
        le = st.selectbox("🦠 Pembengkakan Kelenjar (LE)", [0, 1, 2, 3], key="le_select")
        
        with st.expander(symptoms_help["LN"]):
            st.info(symptoms_descriptions["LN"])
        ln = st.selectbox("⚖️ Penurunan Berat Badan (LN)", [0, 1, 2], key="ln_select")
        
        with st.expander(symptoms_help["SB"]):
            st.info(symptoms_descriptions["SB"])
        sb = st.selectbox("🩸 Batuk Darah (SB)", [0, 1, 2], key="sb_select")
        
        with st.expander(symptoms_help["BMI"]):
            st.info(symptoms_descriptions["BMI"])
        bmi = st.selectbox("📊 BMI (Body Mass Index)", [0, 1, 2], key="bmi_select")

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
# HALAMAN 2: MENU EVALUASI MODEL (tanpa fitur split)
# ============================
elif menu == "Evaluasi Model":
    st.subheader("📊 Evaluasi & Perbandingan Kinerja Model")

    st.markdown(
        """
        Menu ini digunakan untuk **mengevaluasi performa dua model**:
        - **ANN (Deep Learning)**
        - **Random Forest**

        **Instruksi singkat:**  
        1. Pilih mode: **Peneliti (Upload CSV)** untuk upload file data uji, atau **Pengguna/Pasien (Pilih file)** untuk menggunakan file CSV yang ada di folder `datasets/`.  
        2. Setelah file muncul di preview, pilih kolom label (target).  
        3. Klik **Hitung Evaluasi Model** untuk melihat metrik.  

        _Catatan: fitur pembagian dataset (train/test) telah dihapus dari UI sesuai permintaan — seluruh file yang dipilih akan dipakai untuk evaluasi (pastikan file sudah berisi label ground-truth)._
        """
    )

    # Pilihan mode evaluasi: Peneliti (upload) atau Pengguna/Pasien (pakai dataset yang tersedia)
    eval_mode = st.radio("Mode Evaluasi:", ("Peneliti (Upload CSV)", "Pengguna/Pasien (Pilih dataset tersedia)"))

    # Tampilkan uploader segera kalau mode Peneliti dipilih
    if eval_mode == "Peneliti (Upload CSV)":
        uploaded_file = st.file_uploader("Upload Dataset Uji (format .csv)", type=["csv"])
    else:
        uploaded_file = None

    DATASETS_DIR = "datasets"
    os.makedirs(DATASETS_DIR, exist_ok=True)

    # Inisialisasi variabel
    df = None
    target_col_local = None
    chosen = None

    # Fungsi: preview + pilih kolom target; kembalikan (df, target_col)
    def prepare_df_ui(df_input):
        st.write("Preview Dataset Uji:")
        st.dataframe(df_input.head())
        st.write(f"Jumlah baris: **{df_input.shape[0]}**, Jumlah kolom: **{df_input.shape[1]}**")

        # default pilihan target: 'Prediksi' jika ada, kalau tidak pilih kolom terakhir
        if "Prediksi" in df_input.columns:
            default_index = list(df_input.columns).index("Prediksi")
        else:
            default_index = len(df_input.columns) - 1

        target_col = st.selectbox(
            "Pilih kolom label/target (0 = Tidak TBC, 1 = TBC):",
            df_input.columns,
            index=default_index
        )

        st.markdown(
            "Setelah memilih kolom target, klik **Hitung Evaluasi Model** untuk menjalankan evaluasi."
        )

        return df_input, target_col

    # Fungsi evaluasi yang menjalankan seluruh perhitungan dan menampilkan hasil
    def run_full_evaluation(df_eval, target_col_name):
        try:
            # Salin df agar aman
            df_use = df_eval.copy()

            # Normalize & map label (ke 0/1)
            df_use, mapped_label_col = normalize_and_map_label(df_use, label_col_candidates=[target_col_name, 'Prediksi', 'label', 'target'])

            # Pisahkan fitur & label
            X_df = df_use.drop(columns=[mapped_label_col])
            y_true = df_use[mapped_label_col].astype(int).values

            # Jika dataset berisi kolom fitur lebih, ambil FEATURE_COLS bila tersedia
            if all(col in X_df.columns for col in FEATURE_COLS):
                X_df = X_df[FEATURE_COLS]

            # Pastikan jumlah fitur sesuai
            X = X_df.astype(float).values

            if X.shape[1] != len(FEATURE_COLS):
                st.error(f"Model mengharapkan {len(FEATURE_COLS)} fitur, tapi dataset punya {X.shape[1]} fitur.")
                return

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
                    - **Akurasi** : Persentase seluruh prediksi yang **benar**.  
                    - **Presisi** : Dari semua pasien yang **diprediksi TBC**, berapa persen yang benar-benar TBC.  
                    - **Recall (Sensitivitas)** : Dari semua pasien yang benar-benar TBC, berapa persen yang terdeteksi model.  
                    - **F1-Score** : Rata-rata harmonis antara Presisi dan Recall.
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

            # Visualisasi
            st.write("### Visualisasi Perbandingan Metrik")
            df_plot = pd.DataFrame({
                "ANN (Deep Learning)": [acc_ann, prec_ann, rec_ann, f1_ann],
                "Random Forest": [acc_rf, prec_rf, rec_rf, f1_rf],
            }, index=["Akurasi", "Presisi", "Recall", "F1-Score"])
            st.bar_chart(df_plot)

            # Confusion Matrix
            st.write("### Confusion Matrix")
            
            # Penjelasan Confusion Matrix
            with st.expander("📖 Cara Membaca Confusion Matrix"):
                st.markdown(
                    """
                    **Confusion Matrix** menunjukkan perbandingan antara **prediksi model** dan **hasil sebenarnya (ground truth)**.
                    
                    Terdapat 4 komponen utama:
                    
                    1. **True Negative (TN)** - Kiri Atas
                       - Pasien **tidak ada TBC** dan model **benar memprediksi tidak ada TBC** ✅
                       - Ini adalah **hasil yang baik**
                    
                    2. **False Positive (FP)** - Kanan Atas
                       - Pasien **tidak ada TBC** tapi model **salah memprediksi ada TBC** ❌
                       - Hasil positif palsu (alarm palsu)
                    
                    3. **False Negative (FN)** - Kiri Bawah
                       - Pasien **ada TBC** tapi model **salah memprediksi tidak ada TBC** ❌
                       - **Ini adalah kasus paling berbahaya!** (pasien TBC tidak terdeteksi)
                    
                    4. **True Positive (TP)** - Kanan Bawah
                       - Pasien **ada TBC** dan model **benar memprediksi ada TBC** ✅
                       - Ini adalah **hasil yang baik**
                    
                    **Target yang ideal:** Maksimalkan TN dan TP (diagonal utama), minimalkan FP dan FN.
                    """
                )
            
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

            # Classification Reports
            st.write("## 📄 Classification Report")
            
            # Penjelasan Classification Report
            with st.expander("📖 Cara Membaca Classification Report"):
                st.markdown(
                    """
                    **Classification Report** menampilkan metrik detail untuk setiap kelas prediksi (Tidak TBC dan TBC).
                    
                    **Penjelasan Kolom:**
                    
                    - **Precision (Presisi)**
                      - Dari semua yang **diprediksi sebagai kelompok ini**, berapa persen yang **benar**?
                      - Formula: `TP / (TP + FP)`
                      - Contoh (TBC): Dari 100 pasien yang diprediksi TBC, 98 adalah benar-benar TBC = Presisi 98%
                      - **Penting untuk:** Mengurangi alarm palsu (False Positive)
                    
                    - **Recall (Sensitivitas)**
                      - Dari semua yang **benar-benar termasuk kelompok ini**, berapa persen yang **terdeteksi**?
                      - Formula: `TP / (TP + FN)`
                      - Contoh (TBC): Dari 100 pasien yang benar-benar TBC, 97 terdeteksi = Recall 97%
                      - **Penting untuk:** Tidak melewatkan kasus (False Negative) — **sangat kritis untuk TBC!**
                    
                    - **F1-Score**
                      - **Rata-rata harmonis** antara Precision dan Recall
                      - Formula: `2 × (Precision × Recall) / (Precision + Recall)`
                      - Berguna ketika ingin **balance** antara kedua metrik
                      - Range: 0 hingga 1 (1 = sempurna)
                    
                    - **Support**
                      - Jumlah sampel **sebenarnya** untuk setiap kelas
                      - Membantu memahami apakah dataset **balanced** atau **imbalanced**
                    
                    **Interpretasi Hasil:**
                    - **Precision tinggi, Recall rendah** = Model konservatif, jarang memprediksi TBC (banyak False Negative)
                    - **Precision rendah, Recall tinggi** = Model agresif, sering memprediksi TBC (banyak False Positive)
                    - **Keduanya tinggi** = Model ideal ✅
                    
                    **Untuk kasus TBC, lebih penting Recall tinggi** (tidak boleh melewatkan pasien TBC), meskipun ada False Positive.
                    """
                )
            
            st.subheader("🧠 ANN (Deep Learning) — Classification Report")
            report_ann_dict = classification_report(
                y_true, y_pred_ann,
                target_names=list(labelencoder.classes_),
                zero_division=0,
                output_dict=True
            )
            df_ann_report = pd.DataFrame(report_ann_dict).transpose()
            st.dataframe(df_ann_report.style.format("{:.2f}"))

            st.subheader("🌲 Random Forest — Classification Report")
            report_rf_dict = classification_report(
                y_true, y_pred_rf,
                target_names=list(labelencoder.classes_),
                zero_division=0,
                output_dict=True
            )
            df_rf_report = pd.DataFrame(report_rf_dict).transpose()
            st.dataframe(df_rf_report.style.format("{:.2f}"))

        except Exception as e:
            st.error("Terjadi error saat menghitung evaluasi.")
            st.exception(e)

    # ---------------------------
    # MODE: Peneliti (Upload)
    # ---------------------------
    if eval_mode == "Peneliti (Upload CSV)":
        if uploaded_file is not None:
            try:
                df = safe_read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Gagal membaca file CSV: {e}")
                df = None

            if df is not None:
                df, target_col_local = prepare_df_ui(df)

                # Tombol Evaluasi (tidak ada split anymore)
                if st.button("📊 Hitung Evaluasi Model"):
                    run_full_evaluation(df, target_col_local)
        else:
            st.info("Silakan upload file CSV data uji untuk melihat evaluasi dan perbandingan model.")

    # ---------------------------
    # MODE: Pengguna/Pasien (Pilih file di server)
    # ---------------------------
    else:
        csv_files = [f for f in os.listdir(DATASETS_DIR) if f.lower().endswith(".csv")]
        if not csv_files:
            st.warning(f"Tidak ada file CSV di folder {DATASETS_DIR}. Letakkan file CSV di folder tersebut agar bisa dipilih.")
        else:
            chosen = st.selectbox("Pilih dataset uji dari server:", csv_files)
            if chosen:
                path = os.path.join(DATASETS_DIR, chosen)
                try:
                    df = safe_read_csv(path)
                except Exception as e:
                    st.error(f"Gagal membaca {path}: {e}")
                    df = None

                if df is not None:
                    df, target_col_local = prepare_df_ui(df)

                    if st.button("📊 Hitung Evaluasi Model (dari server file)"):
                        run_full_evaluation(df, target_col_local)
