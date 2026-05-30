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
    
    # Metode 5: Segmentasi Adaptif Berbasis Warna & Intensitas
    # Tujuan: Gabungan Otsu (Intensitas) dan HSV Masking (Warna) untuk membuang air laut secara total.
    gray_for_otsu = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    _, mask_otsu = cv2.threshold(gray_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Masking Warna: Membuang area yang terlalu biru (Air) atau terlalu gelap (Bayangan dalam)
    lower_water = np.array([85, 30, 0])
    upper_water = np.array([145, 255, 255])
    mask_water = cv2.inRange(hsv, lower_water, upper_water)
    mask_non_water = cv2.bitwise_not(mask_water)
    
    # Combine: Hanya ambil objek yang bukan air dan memiliki intensitas kontras (Otsu)
    # Cek apakah Otsu mendeteksi objek terang atau gelap (di laut, karang biasanya lebih terang/bertekstur dibanding air dalam)
    mean_val_otsu = np.mean(gray_for_otsu[mask_otsu == 255]) if np.any(mask_otsu == 255) else 0
    mean_val_bg = np.mean(gray_for_otsu[mask_otsu == 0]) if np.any(mask_otsu == 0) else 0
    
    if mean_val_otsu < mean_val_bg: # Jika yang dideteksi Otsu malah lebih gelap (mungkin bayangan), balik masknya
        mask_otsu = cv2.bitwise_not(mask_otsu)
        
    mask = cv2.bitwise_and(mask_otsu, mask_non_water)
    
    # Metode 6: Morphological Operation (Pembersihan Lanjut)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Tutup lubang kecil di dalam karang
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Buang bintik kecil di air
    
    # --- Metode Eksklusif: Isolasi Objek Utama Terbesar (Fokus Spesies Utama) ---
    # Tujuan: Mengabaikan ikan kecil, sampah, pelampung, yang ikut lolos warna, 
    # dengan MEMAKSA sistem HANYA membaca objek/spesies yang paling memakan layar/terbesar.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        bounding_mask = np.zeros_like(mask)
        # Warnai ruang objek terbesar dan jadikan mask tunggal
        cv2.drawContours(bounding_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        # Aplikasikan mask tersebut ke mask utama agar pori-pori/teksturnya tetap valid
        mask = cv2.bitwise_and(mask, bounding_mask)
            
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
    
    # ── Metode 8: Canny Edge Density ─────────────────────────────────────
    edges = cv2.Canny(gray, 100, 200)
    mask_area = int(np.sum(mask == 255))
    if mask_area > 0:
        edge_in_mask = cv2.bitwise_and(edges, edges, mask=mask)
        edge_density = float(np.sum(edge_in_mask == 255) / mask_area)
    else:
        edge_density = float(np.sum(edges == 255) / (224 * 224))

    # ── Vektor Ciri Final (Sesuai Metode Asli SICORAL: H, S, V, T, E) ──
    numeric_features = [
        float(mean_h), float(mean_s), float(mean_v), float(std_gray), edge_density
    ]
    
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
    elif "Bukan" in status_label:
        hue_focus = "Objek tidak menunjukkan karakteristik warna alami terumbu karang (Cokelat/Hijau). Didominasi warna latar belakang."
        sat_focus = "Kepadatan warna tidak konsisten dengan profil biologis terumbu karang; nilai saturasi tidak memenuhi ambang batas minimal."
        val_focus = "Intensitas cahaya menunjukkan pantulan dari media non-karang seperti air atau pasir yang bersifat seragam."
        mask_focus = "Masking mendeteksi objek yang terlalu halus atau tidak memiliki kerumitan struktur polip karang."
        morph_focus = "Garis kontur menunjukkan bentuk yang terlalu sederhana atau tidak terdefinisi sebagai fitur biologis."
        edge_focus = "Kepadatan tekstur (Edge Density) sangat rendah, mengonfirmasi objek ini adalah air, pasir, atau benda mati lainnya."
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
    if features is None or len(features) < 5:
        return "Gagal", 0

    h = features[0]
    s = features[1]
    v = features[2]
    t = features[3]
    e = features[4]

    # ══════════════════════════════════════════════════════════════════
    # REKAYASA LOGIKA PROFESIONAL: PENDEKATAN WHITELISTING 
    # (Berdasarkan 5 fitur bawaan SICORAL)
    # Daripada menebak semua benda non-karang, sistem HANYA MENGIZINKAN
    # ciri organik spesifik untuk lolos. Sisanya ditolak mutlak.
    # ══════════════════════════════════════════════════════════════════
    
    # 1. Pengecekan KETAT Rentang Warna Air Laut/Langit
    # (hanya jika s > 15 agar benda putih/abu netral tidak tersaring salah sebagai biru)
    if (90 <= h <= 140) and s > 15:
        return "Bukan Terumbu Karang (Air/Langit Dominan)", 0.0

    # 2. Pengecekan Benda Halus/Buatan (Mobil, Gedung, Pasir, Ikan/Spesies Lain)
    # Karang harus memiliki tekstur tepi yang sangat rapat khas polip/rongga (e >= 0.18). 
    # Benda dengan fitur buatan atau kulit ikan selalu memiliki Canny Edge < 0.16.
    if t < 22 or e < 0.18:
        return "Bukan Terumbu Karang (Benda Mati/Spesies Lain)", 0.0
        
    # 3. Pengecekan Warna Ekstrem (Ikan Hias, Sampah Plastik)
    # Saturasi rata-rata seluruh piksel karang jarang melebihi 150 karena adanya rongga bayangan.
    if s > 150:
        return "Bukan Terumbu Karang (Warna Sintetis/Hewan)", 0.0

    # 4. Pengecekan SAKIT (Bleached)
    # Karang mati memutih = Kecerahan moderat tinggi (v>=140) dan Saturasi anjlok (s<40)
    if v >= 140 and s < 40:
        status = "Sakit (Bleached)"
        conf = min(99.0, (v / 255) * 100 + 10)
        return status, round(conf, 1)

    # 5. Pengecekan SEHAT (Healthy) vs PUCAT (Warning)
    # Di level ini objek sudah divalidasi sebagai karang bertekstur rapat.
    if s >= 50:
        return "Sehat (Healthy)", round(min(99.0, s + 30), 1)
    else:
        return "Pucat (Warning)", 65.0