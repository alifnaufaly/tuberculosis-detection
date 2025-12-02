# 🫁 Aplikasi Deteksi TBC Paru-Paru

Aplikasi berbasis **Deep Learning (ANN)** dan **Random Forest** untuk mendeteksi risiko penyakit TBC (Tuberkulosis) Paru-Paru menggunakan gejala klinis pasien.

## 🎯 Fitur Utama

- **Prediksi Risiko TBC**: Menggunakan model ANN (Artificial Neural Network) atau Random Forest
- **Evaluasi Model**: Membandingkan performa dua model dengan metrik lengkap (Akurasi, Presisi, Recall, F1-Score)
- **Interface Interaktif**: Dibangun dengan Streamlit untuk pengalaman pengguna yang mudah
- **Dua Mode Evaluasi**: 
  - Mode Peneliti: Upload dataset custom
  - Mode Pengguna/Pasien: Gunakan dataset yang tersedia di server

## 📊 Model Machine Learning

1. **ANN (Deep Learning)**: Jaringan saraf tiruan untuk prediksi non-linear
2. **Random Forest**: Ensemble learning method untuk hasil yang robust

## 🛠️ Teknologi yang Digunakan

- Python 3.x
- Streamlit (Web Framework)
- TensorFlow/Keras (Deep Learning)
- Scikit-learn (Machine Learning)
- NumPy & Pandas (Data Processing)
- Joblib (Model Serialization)

## 📋 Fitur Input (16 Gejala)

| Kode | Gejala | Skala |
|------|--------|-------|
| CO | Batuk | 0-3 |
| NS | Keringat Malam | 0-2 |
| BD | Sesak Napas | 0-2 |
| FV | Demam | 0-2 |
| CP | Nyeri Dada | 0-2 |
| SP | Produksi Dahak | 0-2 |
| IS | Nafsu Makan | 0-3 |
| LP | Nyeri Sendi | 0-2 |
| CH | Menggigil | 0-2 |
| LC | Kelelahan | 0-2 |
| IR | Iritabilitas | 0-2 |
| LA | Lesu | 0-2 |
| LE | Pembengkakan Kelenjar | 0-3 |
| LN | Penurunan Berat Badan | 0-2 |
| SB | Batuk Darah | 0-2 |
| BMI | Body Mass Index | 0-2 |

## 🚀 Cara Menjalankan

### 1. Setup Environment

```bash
# Buat virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# atau untuk Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📁 Struktur File

```
.
├── app.py                              # Main Streamlit application
├── training_model_tbc_ann.ipynb       # Training notebook ANN model
├── training_model_tbc_ann_final_vs_code.ipynb  # Final training ANN
├── model_tbc_ann.h5                   # Trained ANN model
├── model_tbc_rf.pkl                   # Trained Random Forest model
├── scaler_tbc.pkl                     # Data scaler
├── labelencoder_tbc.pkl               # Label encoder
├── tuberculosis_labeled.csv           # Training dataset
├── datasets/                          # Folder untuk dataset uji
│   └── tuberculosis_labeled.csv
├── filebackup/                        # Backup file
└── README.md                          # Dokumentasi ini
```

## ⚙️ File Model yang Diperlukan

Pastikan file berikut ada di folder yang sama dengan `app.py`:
- `model_tbc_ann.h5` - Model ANN
- `model_tbc_rf.pkl` - Model Random Forest
- `scaler_tbc.pkl` - StandardScaler
- `labelencoder_tbc.pkl` - LabelEncoder untuk target variable

## 📊 Dataset Format

Dataset harus berformat CSV dengan kolom:
- Kolom fitur: CO, NS, BD, FV, CP, SP, IS, LP, CH, LC, IR, LA, LE, LN, SB, BMI
- Kolom target: Prediksi (atau label/target) dengan nilai 0/1 atau "Tidak"/"Ya"

Contoh:
```
CO,NS,BD,FV,CP,SP,IS,LP,CH,LC,IR,LA,LE,LN,SB,BMI,Prediksi
1,0,1,1,0,1,2,1,0,1,0,1,1,1,0,1,1
0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0
...
```

## 🎓 Metrik Evaluasi

- **Akurasi**: Persentase seluruh prediksi yang benar
- **Presisi**: Dari pasien yang diprediksi TBC, berapa % yang benar-benar TBC
- **Recall (Sensitivitas)**: Dari pasien dengan TBC, berapa % yang terdeteksi
- **F1-Score**: Rata-rata harmonis antara Presisi dan Recall
- **Confusion Matrix**: Visualisasi True Positives, False Positives, dll

## 👥 Pengguna Target

- Tenaga medis/peneliti untuk evaluasi model
- Fasilitas kesehatan untuk screening awal pasien
- Peneliti untuk pengembangan lebih lanjut

## ⚠️ Disclaimer

Aplikasi ini adalah alat bantu screening berbasis AI dan **BUKAN pengganti diagnosis dokter profesional**. Hasil prediksi harus dikonfirmasi dengan pemeriksaan medis lebih lanjut.

## 📝 License

Project ini dibuat untuk tujuan pendidikan dan penelitian.

## 👨‍💻 Pengembang

Created by: Alifnaufaly

---

**Catatan**: Untuk menggunakan aplikasi ini, pastikan semua model dan file pickle sudah dilatih dan tersimpan dengan benar.
