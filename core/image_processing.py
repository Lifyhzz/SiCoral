import cv2
import numpy as np
import base64

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def extract_features(image_path):
    """
    Multistage Image Processing: Normalisasi, Segmentasi, dan Ekstraksi Vektor Ciri.
    """
    # Membaca data matriks mentah dari gambar
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_original_b64 = image_to_base64(img)
    
    # --- TAHAP PREPROCESSING ---
    # Metode 1: Image Resizing (Normalisasi Ukuran)
    # Tujuan: Menyamakan semua resolusi foto menjadi 224x224 piksel dengan interpolasi agar komputer mudah memprosesnya.
    img_resized = cv2.resize(img, (224, 224))
    img_resized_b64 = image_to_base64(img_resized)
    
    # Metode 2: Grayscale Conversion (Konversi Skala Abu-abu)
    # Tujuan: Menghapus warna (R,G,B) menjadi 1 nilai terang-gelap untuk memudahkan analisis kepadatan tekstur.
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_gray_b64 = image_to_base64(gray)
    
    # Metode 3: Gaussian Blur (Pengurangan Noise / Gangguan Visual)
    # Tujuan: Meratakan/memburamkan piksel kasat mata menggunakan matriks 5x5 untuk menghilangkan bintik gangguan (noise pasir/air).
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    img_norm_b64 = image_to_base64(img_blurred)
    
    # --- TAHAP ANALISIS FITUR ---
    # Metode 4: Transformasi Ruang Warna (BGR ke HSV)
    # Tujuan: Mengubah format RGB ke format Hue, Saturation, Value agar sistem kebal terhadap perubahan silau cahaya matahari laut.
    hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)
    
    # Metode 5: Segmentasi Otomatis (Otsu Thresholding)
    # Tujuan: Mencari angka pembatas (ambang) secara otomatis untuk membedakan piksel terumbu karang (putih) dengan latar air (hitam).
    gray_for_otsu = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray_for_otsu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Metode 6: Morphological Operation (Memperbaiki Hasil Masking / Opening)
    # Tujuan: Melakukan operasi Erosi lalu Dilasi untuk menghapus sisa bintik noise keputihan yang salah terdeteksi oleh Otsu.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # (Koreksi Tambahan) Pengecekan warna air laut, jika warnanya air maka balikkan mask (invert)
    if np.any(mask == 255):
        mean_h_initial = np.mean(hsv[:, :, 0][mask == 255])
        if 90 < mean_h_initial < 140:
            mask = cv2.bitwise_not(mask)
            
    foreground_pixels = hsv[mask == 255]
    
    # Metode 7: Ekstraksi Fitur Statistik (Rata-rata HSV & Standar Deviasi Tekstur)
    # Tujuan: Menjumlahkan seluruh piksel objek terumbu karang lalu dihitung rata-rata warnanya dan sebaran kekasaran teksturnya.
    if len(foreground_pixels) == 0:
        mean_h = np.mean(hsv[:, :, 0])
        mean_s = np.mean(hsv[:, :, 1])
        mean_v = np.mean(hsv[:, :, 2])
        std_gray = np.std(gray_for_otsu)
    else:
        mean_h = np.mean(foreground_pixels[:, 0])
        mean_s = np.mean(foreground_pixels[:, 1])
        mean_v = np.mean(foreground_pixels[:, 2])
        gray_foreground = gray_for_otsu[mask == 255]
        std_gray = np.std(gray_foreground)
    
    # Menyusun Vektor Ciri: [Statistik Terpusat H, S, V & Analisis Dispersi Tekstur]
    numeric_features = [float(mean_h), float(mean_s), float(mean_v), float(std_gray)]
    
    # --- MEMBUAT 10 FILTER UNTUK VISUALISASI ---
    filters = []
    
    # Mendapatkan status dinamis untuk memfokuskan teks penjelasan
    status_label, _ = predict_coral_health(numeric_features)
    
    if "Sakit" in status_label or "Bleached" in status_label:
        hue_focus = "Gambar didominasi rona pucat/kebiruan (warna dasar), mengonfirmasi ketiadaan sel pigmen Zooxanthellae pada karang yang sakit (Bleached)."
        sat_focus = "Keseluruhan gambar tampak <strong>sangat redup dan pudar</strong>. Ini membuktikan bahwa pigmen sel pelindung karang sudah hilang."
        val_focus = "Ditemukan ledakan titik <strong>Cahaya Sangat Terang / Menyilaukan</strong> akibat terumbu karang mati yang rentan memantulkan mentah sinar UV luar."
        mask_focus = "Sistem secara terik mengunci area isolasi blok putih ini sebagai wujud persis <strong>Karang Sakit (Bleached)</strong> untuk mencegah perhitungan noise lautan."
        morph_focus = "Siluet pinggiran mengungkap struktur rangka keras yang telah terekspos karena telah ditinggalkan oleh alga berharganya."
        edge_focus = "Garis tepi terlihat terlalu halus atau luntur keputihan. Pantulan cahaya ke batu mati menyingkirkan detail tekstur alaminya (garis nihil)."
    elif "Pucat" in status_label or "Warning" in status_label:
        hue_focus = "Warna alami (cokelat) mulai tenggelam dan <strong>bercampur dengan kepucatan</strong>. Mengindikasikan karang menanggung stres luar (Pucat / Warning)."
        sat_focus = "Kepadatan warna merosot drastis memperbanyak titik abu-abu pudar. Ini membuktikan penurunan kritis ketebalan sel warna karang."
        val_focus = "Mulai maraknya bercak terang yang menutupi pigmen aslinya — gejala krusial pemicu awal fase pemutihan yang lebih berat."
        mask_focus = "Bentuk blok putih ini dimanfaatkan untuk mengkunci kalkulasi status karang yang tengah melemah <strong>(Pucat)</strong> dengan presisi tinggi."
        morph_focus = "Batas kontur menyoroti fisik murni terumbu batu yang sedang berjuang dalam fase pergeseran struktur transisi pucat."
        edge_focus = "Garis kerutan halus masih rimbun namun setengahnya mulai pecah karena bercak gradasi dari karang yang telah luntur sebagian."
    else: # SEHAT
        hue_focus = "Area secara konsisten dikuasai oleh <strong>Cokelat Kehijauan / Rona Hangat</strong>. Bukti melimpahnya pigmen warna alami karang yang Sehat."
        sat_focus = "Muncul titik permukaan yang didominasi warna <strong>Ekstra Pekat (Tinggi intensitas)</strong>. Hanya bisa dijumpai dari tebalnya polip terumbu karang yang sehat murni."
        val_focus = "Terpantau merata dengan level redup aman. Karang meredam cahaya, tidak ditemukannya kecerahan silau pantul yang berlebih."
        mask_focus = "Gambar siluet putih utuh mewakili luasan total <strong>Karang Sehat</strong> yang sukses diisolasi bersih dan ditarik keluar dari background laut."
        morph_focus = "Shape garis terluar (batas outline) meramu postur fisik solid tanpa kerentanan yang terkuak maupun distorsi pada rangkanya."
        edge_focus = "Diliputi <strong>Garis Tekstur Kasar/Rimbun</strong> di sekujur badan. Bukti nyata bahwa terumbu memiliki arsitektur alam berpori (menandakan sangat sehat)."

    # --- FILTER 1: Hue Channel (komponen warna) ---
    f1 = hsv[:, :, 0] # Mengambil Layer ke-0 (Hue/Jenis Warna) dari matriks HSV
    desc_f1 = f'''<p class="mb-2"><strong>Fokus Deteksi:</strong> {hue_focus}</p>
    <div class="bg-indigo-50 border border-indigo-100 p-3 rounded-lg mt-3">
    <p class="text-[10px] font-black text-indigo-700 mb-1 uppercase tracking-wide">Fungsi Channel Hue (H):</p>
    <p class="text-[9px] text-slate-600 leading-relaxed">Menampilkan jenis warna sejati pada gambar tanpa terpengaruh bayangan gelap-terang air laut.</p>
    </div>'''
    filters.append({'name': 'Hue Map', 'img': image_to_base64(f1), 'desc': desc_f1})
    
    # --- FILTER 2: Saturation Channel (kepekatan warna) ---
    f2 = hsv[:, :, 1] # Mengambil Layer ke-1 (Saturation/Kepekatan) dari matriks HSV
    desc_f2 = f'''<p class="mb-2"><strong>Fokus Deteksi:</strong> {sat_focus}</p>
    <div class="bg-indigo-50 border border-indigo-100 p-3 rounded-lg mt-3">
    <p class="text-[10px] font-black text-indigo-700 mb-1 uppercase tracking-wide">Fungsi Channel Saturation (S):</p>
    <p class="text-[9px] text-slate-600 leading-relaxed">Mengukur ketebalan atau kepekatan warna. Semakin tinggi nilainya, pigmen karang tersebut semakin kuat.</p>
    </div>'''
    filters.append({'name': 'Sat. Map', 'img': image_to_base64(f2), 'desc': desc_f2})
    
    # --- FILTER 3: Value Channel (kecerahan) ---
    f3 = hsv[:, :, 2] # Mengambil Layer ke-2 (Value/Kecerahan) dari matriks HSV
    desc_f3 = f'''<p class="mb-2"><strong>Fokus Deteksi:</strong> {val_focus}</p>
    <div class="bg-indigo-50 border border-indigo-100 p-3 rounded-lg mt-3">
    <p class="text-[10px] font-black text-indigo-700 mb-1 uppercase tracking-wide">Fungsi Channel Value (V):</p>
    <p class="text-[9px] text-slate-600 leading-relaxed">Mengukur intensitas cahaya (kecerahan). Sangat krusial untuk mendeteksi warna tulang putih karang mati yang rentan silau.</p>
    </div>'''
    filters.append({'name': 'Val. Map', 'img': image_to_base64(f3), 'desc': desc_f3})
    
    # --- FILTER 4: Object Mask (Otsu + Morphological Opening) ---
    desc_f4 = f'''<p class="mb-2"><strong>Fokus Deteksi:</strong> {mask_focus}</p>
    <div class="bg-indigo-50 border border-indigo-100 p-3 rounded-lg mt-3">
    <p class="text-[10px] font-black text-indigo-700 mb-1 uppercase tracking-wide">Fungsi Object Mask:</p>
    <p class="text-[9px] text-slate-600 leading-relaxed">Pemisahan otomatis terumbu karang dari latar air laut menggunakan algoritma Otsu, disusul pembersihan bintik kotoran dengan Filter Morfologi.</p>
    </div>'''
    filters.append({'name': 'Obj. Mask', 'img': image_to_base64(mask), 'desc': desc_f4})
    
    # --- FILTER 5: Morphological Gradient ---
    f5 = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel) # Mencari sisa garis tepi dari rumus matematika: Gambar Dilasi (Ditebalkan) dikurangi Gambar Erosi (Dikikis)
    desc_f5 = f'''<p class="mb-2"><strong>Fokus Deteksi:</strong> {morph_focus}</p>
    <div class="bg-indigo-50 border border-indigo-100 p-3 rounded-lg mt-3">
    <p class="text-[10px] font-black text-indigo-700 mb-1 uppercase tracking-wide">Fungsi Gradien Morfologi:</p>
    <p class="text-[9px] text-slate-600 leading-relaxed">Menghitung selisih matematis antara citra yang ditempel margin (Dilasi) dan dikikis (Erosi) sehingga yang tersisa hanya murni batas kontur luarnya.</p>
    </div>'''
    filters.append({'name': 'Morph Grad', 'img': image_to_base64(f5), 'desc': desc_f5})
    
    # --- FILTER 6: Canny Edge Detection ---
    f6 = cv2.Canny(gray, 100, 200) # Melacak garis tekstur tajam dengan mengabaikan pixel bernilai di bawah 100 dan memperkuat pixel di atas 200
    desc_f6 = f'''<p class="mb-2"><strong>Fokus Deteksi:</strong> {edge_focus}</p>
    <div class="bg-indigo-50 border border-indigo-100 p-3 rounded-lg mt-3">
    <p class="text-[10px] font-black text-indigo-700 mb-1 uppercase tracking-wide">Fungsi Canny Edge Det.:</p>
    <p class="text-[9px] text-slate-600 leading-relaxed">Teknologi deteksi sudut tajam untuk melacak serabut/tekstur permukaan batu karang seraya secara paksa membuang garis bias (noise) pencahayaan dasar laut.</p>
    </div>'''
    filters.append({'name': 'Edge Det.', 'img': image_to_base64(f6), 'desc': desc_f6})
    
    return {
        'features': numeric_features,
        'steps': [
            {'name': 'Original', 'img': img_original_b64, 'desc': 'Citra asli terumbu karang sebelum diproses.'},
            {'name': 'Resize', 'img': img_resized_b64, 'desc': 'Mengubah dimensi menjadi 224x224 piksel untuk input model.'},
            {'name': 'Grayscale', 'img': img_gray_b64, 'desc': 'Konversi ke skala abu-abu untuk analisis tekstur (visualisasi).'},
            {'name': 'Normalisasi', 'img': img_norm_b64, 'desc': 'Aplikasi Gaussian Blur & Normalisasi rentang [0, 1].'}
        ],
        'filters': filters
    }

# Modul Klasifikasi berbasis Aturan Logika (Heuristic Rule-Based Reasoning)
def predict_coral_health(features):
    if features is None:
        return "Gagal", 0
    
    h = features[0] # Hue (Sifat Warna, rentang 0-180)
    s = features[1] # Saturation (Kepekatan Warna, rentang 0-255)
    v = features[2] # Value / Brightness (Kecerahan, rentang 0-255)
    t = features[3] # Tekstur (Standar Deviasi)
    
    # LOGIKA PENDETEKSIAN:
    # Terumbu Sakit (Bleached) = Berwarna Putih (Kecerahan V Tinggi, Warna S Rendah)
    # Terumbu Sehat = Berwarna Cokelat/Hijau (Hue Rendah, Saturasi S Cukup)
    
    # Deteksi "Sehat Palsu" (Warna Biru Air)
    # Jika warna dominan adalah Biru/Sian (Hue 90-140), itu kemungkinan warna air laut, bukan pigmen terumbu karang.
    is_water_color = 85 < h < 145
    
    # KRITERIA BLEACHED (SAKIT)
    # Jika sangat terang (Putih) ATAU warnanya Pudar bercampur biru air laut
    if v > 165 and (s < 80 or is_water_color):
        status = "Sakit (Bleached)" # Klasifikasi: Kecerahan Tinggi & Kehilangan Pigmen
        conf = min(99.0, (v / 255) * 100 + 20)
        return status, round(conf, 1)

    # ANALISIS KONDISI OPTIMAL (SEHAT)
    if s > 60 and v < 175 and not is_water_color and t > 10:
        status = "Sehat (Healthy)" # Klasifikasi: Kepekatan Pigmen Optimal & Tekstur Terumbu Aktif
        conf = min(99.0, s + 10)
        return status, round(conf, 1)

    # SECARA DEFAULT (TIDAK MEMENUHI KEDUANYA): PUCAT / PERINGATAN DINI
    status = "Pucat (Warning)"
    return status, 75.0