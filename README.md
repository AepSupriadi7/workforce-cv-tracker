# Workforce CV Tracker: Real-Time Employee Monitoring System using Hybrid SOTA Architecture

Sistem pemantau kehadiran dan penghitung akumulasi jam kerja karyawan secara *real-time* berbasis *Edge Computing*. Proyek ini merupakan implementasi tingkat lanjut dari mata kuliah **Advanced Computer Vision (Magister Teknik Informatika)** yang menggabungkan efisiensi deteksi lokasional dan ketangguhan pengenalan identitas berbasis vektor geometri.

---

## 🎯 Pergeseran Paradigma: Klasik ke Deep Learning

Proyek ini bermigrasi dari pendekatan *Computer Vision* klasik (**Haar Cascade + LBPH**) menuju arsitektur *State-of-the-Art (SOTA) Deep Learning* (**YOLOv8-Nano + Dlib ResNet-34**). 

**Mengapa Migrasi Ini Dilakukan?**
* **Mengatasi Batasan Radius (Spatial Constraints):** Algoritma klasik seperti LBPH sangat rentan terhadap *pixel drop* ketika objek menjauh dari kamera (di atas 1 meter). Integrasi YOLOv8-Nano berhasil mengunci posisi wajah secara stabil pada radius **2 meter hingga lebih**.
* **One-Shot Learning Paradigm:** Berbeda dengan LBPH yang membutuhkan puluhan sampel foto per user untuk *training* ulang, arsitektur baru ini hanya membutuhkan **1 foto master** per karyawan untuk ekstraksi *Face Embedding*.

---

## 🏗️ Arsitektur Sistem (Two-Stage Pipeline)

Sistem bekerja secara hibrida melalui dua tahapan utama pada setiap *frame* video yang ditangkap:

1. **Stage 1: Face Detection (YOLOv8-Nano Wajah)**
   Menggunakan model Jaringan Saraf Konvolusional (CNN) yang telah dioptimasi untuk mendeteksi koordinat lokasi wajah global secara cepat dan presisi tinggi meskipun dalam kondisi *low-resolution* karena jarak jauh.
2. **Stage 2: Face Recognition (Dlib ResNet-34)**
   Koordinat hasil deteksi YOLOv8 dikonversi ke format lokasi CSS, lalu diekstrak oleh model ResNet-34 menjadi **128-Dimensional Face Embedding Vectors**. Sistem menghitung kedekatan matriks menggunakan **Jarak Euclidean (Euclidean Distance)** dengan batas toleransi ketat (**Threshold = 0.48**) untuk meminimalkan angka *False Positive* (salah tebak objek).

---

## 📁 Struktur Direktori Proyek

```text
workforce-cv-tracker/
│
├── foto_karyawan/            # Database foto master karyawan (1 foto per user)
│   ├── 1_Aep Supriadi.jpg    # Format penamaan: [ID]_[Nama_Karyawan].jpg
│   └── 2_Anindya Relbi.jpg
        3_Faishal Fahmi.jpg
│
├── assets/
│   └── face_yolov8n.pt       # Pre-trained Weights YOLOv8 Face Detection
│
├── src/
│   ├── deteksi.py            # Core AI Inference (YOLOv8 + ResNet Vector Mapping)
│   └── hitung_waktu.py       # Logika logika Time-Tracker & Absensi Checkpoint
│
├── logs/
│   └── data_absensi.csv      # Output rekaman durasi kerja otomatis (Fail-Safe)
│
└── main.py                   # Main Controller & Heads-Up Display (HUD) Interface