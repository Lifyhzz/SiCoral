# SiCoral: Sistem Deteksi Kesehatan Terumbu Karang 🪸

SiCoral adalah aplikasi berbasis web yang dikembangkan menggunakan prinsip **Pengolahan Citra Digital (PCD)** konvensional untuk mendeteksi tingkat kesehatan terumbu karang secara otomatis. Sistem ini dirancang untuk membedakan antara terumbu karang yang sehat (*Healthy*), mengalami stres (*Warning/Pucat*), dan yang mengalami pemutihan (*Bleached/Sakit*).

## 🚀 Fitur Utama
- **Dashboard Analisis Real-time**: Antarmuka modern untuk pemantauan hasil ekstraksi ciri secara instan.
- **Multistage Preprocessing**: Visualisasi setiap tahapan pengolahan citra dari raw hingga siap ekstraksi.
- **Advanced Feature Extraction**: Ekstraksi parameter statistik (Mean Hue, Saturation, Value) dan analisis tekstur (Standard Deviation).
- **Rule-based Classification**: Logika pengambilan keputusan transparan berbasis ambang batas (Thresholding) tanpa menggunakan Black-box AI/CNN.
- **Technical Diagnostic Report**: Laporan mendalam yang menjelaskan alasan sistem mengambil keputusan berdasarkan data ekstraksi.

## 🛠️ Metodologi Pengolahan Citra
SiCoral menggunakan pipeline PCD murni dengan tahapan sebagai berikut:
1. **Acquisition & Preprocessing**: Resize (224x224), Grayscale Conversion, dan Gaussian Blur (5x5) untuk mereduksi noise (seperti bintik air/pasir).
2. **Color Transformation**: Konversi ruang warna citra dari BGR ke **HSV** untuk persiapan masking warna laut dan mengukur kekuatan pigmentasi.
3. **Adaptive Segmentation**: Menggabungkan **Otsu's Thresholding** (berdasarkan intensitas) dengan **HSV Masking** (membuang rentang warna biru air). Setelah itu, dilakukan **Morphological Operations** (Close & Open) serta pemisahan area terbesar (*Largest Contour Isolation*) untuk mengekstrak objek utama.
4. **Feature Extraction**: Ekstraksi 5 vektor ciri (5-Features) utama: nilai rata-rata (Mean) Hue, Saturation, Value; standar deviasi intensitas (Tekstur ROI); dan perhitungan **Canny Edge Density** untuk validasi kekasaran permukaan.
5. **Heuristic Classification (*Whitelisting*)**: Klasifikasi 100% berbasis *Rule-based / Thresholding Logic*, di mana sistem secara ketat mem-filter objek non-karang (*whitelisting*) sebelum memvonis kesehatan menjadi Sehat (*Healthy*), Pucat (*Warning*), atau Sakit (*Bleached*).

## 💻 Teknologi yang Digunakan
- **Backend**: Python 3.x, Flask Framework.
- **Image Processing**: OpenCV (Open Source Computer Vision Library), NumPy.
- **Frontend**: HTML5, Vanilla JavaScript, Tailwind CSS (Modern UI Framework).
- **Icons**: Lucide Icons & FontAwesome.

## 📂 Struktur Proyek
```text
SICORAL/
├── app.py                 # Entry point aplikasi Flask & routing
├── core/
│   └── image_processing.py# Core engine logika PCD & ekstraksi ciri
├── static/
│   └── uploads/           # Penyimpanan sementara gambar yang diunggah
├── templates/
│   ├── landing.html       # Halaman utama (Portal)
│   └── index.html         # Dashboard analisis utama
└── README.md
```

## ⚙️ Cara Menjalankan Aplikasi
1. **Clone Repository**
   ```bash
   git clone https://github.com/Lifyhzz/SiCoral.git
   cd SiCoral
   ```

2. **Install Dependensi**
   Pastikan Anda telah menginstal Python, lalu jalankan:
   ```bash
   pip install flask opencv-python numpy
   ```

3. **Jalankan Server**
   ```bash
   python app.py
   ```

4. **Akses Aplikasi**
   Buka browser dan kunjungi `http://127.0.0.1:5000`

---
**Catatan Penting:** Proyek ini dikembangkan untuk memenuhi kebutuhan tugas mata kuliah Pengolahan Citra Digital (PCD) dengan fokus pada implementasi algoritma pengolahan citra konvensional.
