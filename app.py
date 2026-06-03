"""
app.py
------
File utama aplikasi Flask (Web Server).
Bertugas menangani sistem routing navigasi (UI), menerima request upload citra dari user,
dan menghubungkannya dengan modul backend (image_processing.py) untuk klasifikasi.
"""
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Import fungsionalitas inti (backend pemrosesan citra)
from core.image_processing import extract_features, predict_coral_health

app = Flask(__name__)
app.secret_key = "sicoral_secret_key"

# Konfigurasi folder upload
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder upload tersedia
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- FUNGSI UTILITAS PENDUKUNG ---
# Verifikasi format ekstensi file yang diunggah agar aman (hanya menerima gambar)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- DEFINISI ROUTING UI (PAGES) ---
# Endpoint Root (/): Menampilkan halaman awal "Landing Page" dengan bagian 'overview'
@app.route('/', methods=['GET'])
def landing():
    return render_template('landing.html', stage='overview')

# Endpoint Info (/info/...): Menampilkan rincian setiap tahap di landing page (Dynamic Routing)
# Mengirim data dictionary spesifik sesuai dengan URL yang diklik user ('input', 'analisis', atau 'hasil').
@app.route('/info/<stage>', methods=['GET'])
def landing_info(stage):
    # Mapping untuk data teks edukasi di UI
    info_data = {
        'input': {
            'title': 'Tahap 01: Input Citra',
            'desc': 'Proses awal dimulai dengan pengunggahan citra digital terumbu karang.',
            'detail': 'Sistem mendukung format citra standar seperti JPG, JPEG, dan PNG. Untuk hasil optimal, citra sebaiknya diambil dengan pencahayaan yang cukup dan resolusi minimal 720p. Citra akan melalui proses normalisasi ukuran sebelum masuk ke tahap prapemrosesan.'
        },
        'analisis': {
            'title': 'Tahap 02: Analisis PCD',
            'desc': 'Digital Signal Processing & Feature Extraction.',
            'detail': 'Sistem melakukan pemisahan objek menggunakan Otsu Thresholding untuk akurasi segmentasi maksimal. Setelah itu, ekstraksi fitur dilakukan pada ruang warna HSV untuk mendeteksi saturasi warna terumbu karang secara presisi, yang menjadi basis logika klasifikasi kesehatan ekosistem laut.'
        },
        'hasil': {
            'title': 'Tahap 03: Klasifikasi Hasil',
            'desc': 'Dapatkan hasil analisis instan dengan tingkat kepercayaan tinggi.',
            'detail': 'Sistem memberikan klasifikasi akhir: Sehat, Pucat (Warning), Terjadi Pemutihan (Bleached), atau Bukan Terumbu Karang. Selain status, Anda akan mendapatkan nilai ekstraksi fitur secara mendetail dan keterangan sistem berdasarkan logika inferensi yang telah diuji.'
        }
    }
    
    if stage not in info_data:
        return redirect(url_for('landing'))
        
    return render_template('landing.html', stage=stage, data=info_data[stage])

# Endpoint Dashboard Utama (/dashboard): Halaman workspace tempat user mengunggah foto
@app.route('/dashboard', methods=['GET'])
def index():
    # Data dummy evaluasi model untuk ditampilkan di sidebar UI
    stats = {
        'accuracy': '95.2%',
        'f1_score': '0.95',
        'latency': '14ms'
    }
    return render_template('index.html', stats=stats)

# --- ENDPOINT INTI (PROSES & PREDIKSI VERSI BACKEND) ---
# Endpoint Deteksi (/detect): Mengolah hasil POST submit berupa file gambar.
@app.route('/detect', methods=['POST'])
def detect():
    # Verifikasi unggahan file
    if 'file' not in request.files:
        flash('Tidak ada file yang diunggah.')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('Tidak ada gambar yang dipilih.')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Penyimpanan citra sementara
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 1. Pipeline Pemrosesan Citra: Melakukan Pra-pemrosesan & Ekstraksi Fitur dari gambar tersebut.
        result_data = extract_features(filepath) 
        
        if result_data:
            features = result_data['features'] # List/array berisi matriks nilai konkrit
            
            # 2. Pipeline Inferensi: Evaluasi hasil nilai menggunakan logika Heuristic
            status_prediksi, confidence_score = predict_coral_health(features)
            
            # Opsional: Memisahkan detail/penjelasan khusus bila prediksi menolak citra (Bukan Karang).
            status_title = status_prediksi
            status_detail = ""
            if "Bukan Terumbu Karang" in status_prediksi and "(" in status_prediksi:
                parts = status_prediksi.split(" (", 1)
                status_title = parts[0]
                status_detail = parts[1].rstrip(")")
            
            # Format akhir output array untuk dilempar dan dirender di index.html (frontend)
            result = {
                'status': status_title,
                'detail': status_detail,
                'confidence': f"{confidence_score}%",
                'image_url': f'uploads/{filename}',
                'features': {
                    'Mean Hue': round(features[0], 2),
                    'Mean Saturation': round(features[1], 2),
                    'Mean Value': round(features[2], 2),
                    'Edge Density': round(features[4], 4),
                    'Std Dev (Texture)': round(features[3], 2)
                },
                'steps': result_data['steps'],
                'filters': result_data['filters']
            }
        else:
            flash('Gagal memproses gambar.')
            return redirect(url_for('index'))
        
        stats = {
            'accuracy': '95.2%',
            'f1_score': '0.95',
            'latency': '14ms'
        }
        
        return render_template('index.html', result=result, stats=stats)
    
    flash('Format file tidak diizinkan. Gunakan JPG atau PNG.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)