import cv2
import numpy as np
import base64

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def extract_features(image_path):
    """
    Memuat gambar, melakukan preprocessing, dan ekstraksi ciri.
    Mengembalikan fitur numerik dan data visual (base64) untuk dashboard.
    """
    # 1. Baca Citra (Original)
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_original_b64 = image_to_base64(img)
    
    # 2. Resize (Step 2)
    img_resized = cv2.resize(img, (224, 224))
    img_resized_b64 = image_to_base64(img_resized)
    
    # 3. Grayscale (Step 3)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_gray_b64 = image_to_base64(gray)
    
    # 4. Normalisasi / Blur (Step 4 - Visualisasi Blur/Preprocessed)
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    img_norm_b64 = image_to_base64(img_blurred)
    
    # --- PROSES EKSTRAKSI FITUR UNTUK LOGIKA ---
    # (Menggunakan img_blurred untuk perhitungan fitur)
    hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)
    
    # Segmentasi (Background Removal - Otsu)
    gray_for_otsu = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray_for_otsu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    if np.any(mask == 255):
        mean_h_initial = np.mean(hsv[:, :, 0][mask == 255])
        if 90 < mean_h_initial < 140:
            mask = cv2.bitwise_not(mask)
            
    foreground_pixels = hsv[mask == 255]
    
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
    
    numeric_features = [float(mean_h), float(mean_s), float(mean_v), float(std_gray)]
    
    # --- GENERATE 8 FILTERS FOR VISUALIZATION ---
    filters = []
    
    # Filter 1: Canny Edge
    f1 = cv2.Canny(gray, 100, 200)
    filters.append({'name': 'Edge Det.', 'img': image_to_base64(f1)})
    
    # Filter 2: Sobel X
    f2 = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    f2 = cv2.convertScaleAbs(f2)
    filters.append({'name': 'Sobel X', 'img': image_to_base64(f2)})
    
    # Filter 3: Laplacian
    f3 = cv2.Laplacian(gray, cv2.CV_64F)
    f3 = cv2.convertScaleAbs(f3)
    filters.append({'name': 'Laplacian', 'img': image_to_base64(f3)})
    
    # Filter 4: Hue Channel
    f4 = hsv[:, :, 0]
    filters.append({'name': 'Hue Map', 'img': image_to_base64(f4)})
    
    # Filter 5: Saturation Channel
    f5 = hsv[:, :, 1]
    filters.append({'name': 'Sat. Map', 'img': image_to_base64(f5)})
    
    # Filter 6: Value Channel
    f6 = hsv[:, :, 2]
    filters.append({'name': 'Val. Map', 'img': image_to_base64(f6)})
    
    # Filter 7: Morphological Gradient
    f7 = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    filters.append({'name': 'Morph Grad', 'img': image_to_base64(f7)})
    
    # Filter 8: Threshold Mask
    filters.append({'name': 'Obj. Mask', 'img': image_to_base64(mask)})
    
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

def predict_coral_health(features):
    if features is None:
        return "Gagal", 0
    
    h = features[0] # Hue (0-180)
    s = features[1] # Saturation (0-255)
    v = features[2] # Brightness (0-255)
    t = features[3] # Texture (Std Dev)
    
    # LOGIKA PERBAIKAN:
    # Terumbu Sakit (Bleached) = Berwarna Putih (Kecerahan V Tinggi, Warna S Rendah)
    # Terumbu Sehat = Berwarna Cokelat/Hijau (Hue Rendah, Saturasi S Cukup)
    
    # 1. Deteksi "Fake Healthy" (Warna Biru Air)
    # Jika warna dominan adalah Biru/Sian (Hue 90-140), itu kemungkinan air, bukan pigmen karang.
    is_water_color = 85 < h < 145
    
    # 2. KRITERIA BLEACHED (SAKIT)
    # Jika sangat terang (Putih) ATAU Pudar dengan warna air
    if v > 165 and (s < 80 or is_water_color):
        status = "Sakit (Bleached)"
        conf = min(99.0, (v / 255) * 100 + 20)
        return status, round(conf, 1)

    # 3. KRITERIA SEHAT (HEALTHY)
    # Harus memiliki Saturasi cukup (>60), Tidak terlalu terang (V < 175),
    # Bukan warna air, dan memiliki Tekstur (>10)
    if s > 60 and v < 175 and not is_water_color and t > 10:
        status = "Sehat (Healthy)"
        conf = min(99.0, s + 10)
        return status, round(conf, 1)

    # 4. DEFAULT: PUCAT / WARNING
    status = "Pucat (Warning)"
    return status, 75.0