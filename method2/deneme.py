import os
import cv2
import dlib
import time
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# =============================================================================
# --- DENEY KONFİGÜRASYONU ---
# =============================================================================

# Test edilecek hedef boyutlar
TARGET_SIZES_TO_TEST = [64, 96, 128]

# Deneyi hızlandırmak için kullanılacak veri yüzdesi (örn: %20'si)
DATA_SUBSET_FRACTION = 0.20

# Ham veri setinin bulunduğu klasörün adı
INPUT_FOLDER_NAME = "fane_data"

# Script'in çalıştığı ana dizin
BASE_DIR = Path().resolve()
INPUT_DATA_PATH = BASE_DIR / INPUT_FOLDER_NAME

# =============================================================================
# --- GEREKLİ FONKSİYONLAR (Kendi içinde) ---
# =============================================================================

face_detector = dlib.get_frontal_face_detector()

def process_and_save_image(input_path, output_path, target_size):
    """
    Tek bir resmi okur, yüzü bulur, işler ve hedefe kaydeder.
    Bu fonksiyon, harici preprocess.py'nin işini yapar.
    """
    img = cv2.imread(str(input_path))
    if img is None: return False

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rects = face_detector(rgb, 1)
    if not rects: return False

    rect = rects[0]
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()

    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2
    new_w, new_h = int(w * 1.5), int(h * 1.5)
    nx1, ny1 = max(cx - new_w // 2, 0), max(cy - new_h // 2, 0)
    nx2, ny2 = min(cx + new_w // 2, img.shape[1]), min(cy + new_h // 2, img.shape[0])
    
    face_crop = img[ny1:ny2, nx1:nx2]
    if face_crop.size == 0: return False

    # CLAHE
    img_yuv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    face_crop_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Resize and Pad
    h_crop, w_crop = face_crop_clahe.shape[:2]
    scale = target_size / max(h_crop, w_crop)
    resized = cv2.resize(face_crop_clahe, (int(w_crop * scale), int(h_crop * scale)))

    final_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = (target_size - final_img.shape[0]) // 2
    x_off = (target_size - final_img.shape[1]) // 2
    canvas[y_off:y_off + final_img.shape[0], x_off:x_off + final_img.shape[1]] = final_img

    os.makedirs(output_path.parent, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return True

# =============================================================================
# --- DENEY SÜRECİ ---
# =============================================================================

results = []
image_extensions = [".jpg", ".jpeg", ".png"]
all_raw_images = [p for p in INPUT_DATA_PATH.rglob("*") if p.suffix.lower() in image_extensions]

if not all_raw_images:
    print(f"HATA: '{INPUT_DATA_PATH}' klasöründe hiç resim bulunamadı. Deney başlatılamıyor.")
else:
    for target_size in TARGET_SIZES_TO_TEST:
        print("\n" + "="*80)
        print(f"DENEY BAŞLIYOR: BOYUT = {target_size}x{target_size}")
        print("="*80)
        
        start_time_total = time.time()
        temp_output_path = BASE_DIR / f'temp_processed_{target_size}'
        
        try:
            # --- 1. Dahili Ön İşleme ---
            print(f"1. Ön işleme yapılıyor ve '{temp_output_path.name}' klasörüne kaydediliyor...")
            for img_path in tqdm(all_raw_images, desc=f"Ön İşleme ({target_size}x{target_size})"):
                rel_path = img_path.relative_to(INPUT_DATA_PATH)
                out_path = temp_output_path / rel_path.with_suffix('.png')
                process_and_save_image(img_path, out_path, target_size)

            # --- 2. HOG Özelliklerini Çıkarma (Alt Küme Üzerinde) ---
            print(f"\n2. HOG özellikleri çıkarılıyor (Verinin %{int(DATA_SUBSET_FRACTION*100)}'i kullanılacak)...")
            features_list, labels_list = [], []
            
            # Alt kümeyi seçmek için tüm işlenmiş dosyaları ve etiketlerini topla
            processed_files_with_labels = []
            for class_dir in temp_output_path.iterdir():
                if class_dir.is_dir():
                    for img_file in class_dir.glob('*.png'):
                        processed_files_with_labels.append((img_file, class_dir.name))
            
            _, subset_files = train_test_split(
                processed_files_with_labels, 
                test_size=DATA_SUBSET_FRACTION, 
                stratify=[f[1] for f in processed_files_with_labels], 
                random_state=42
            )
            
            for img_path, label in tqdm(subset_files, desc="HOG Çıkarılıyor"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                                       cells_per_block=(2, 2), visualize=False)
                    features_list.append(hog_features)
                    labels_list.append(label)

            X_subset = np.array(features_list)
            y_subset = np.array(labels_list)
            
            if X_subset.shape[0] == 0:
                print("Hata: Hiç özellik çıkarılamadı.")
                continue

            # --- 3. Hızlı Model Eğitimi ---
            print("\n3. Basit bir SVM modeli eğitiliyor...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_subset, y_subset, test_size=0.25, random_state=42, stratify=y_subset
            )
            
            pipe_fast_svm = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(C=1, gamma='scale', random_state=42)) # Basit, sabit parametreler
            ])
            
            pipe_fast_svm.fit(X_train, y_train)
            y_pred = pipe_fast_svm.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            total_duration = time.time() - start_time_total
            
            # --- 4. Sonuçları Kaydet ---
            print(f"\nSONUÇ (BOYUT={target_size}x{target_size}):")
            print(f"  -> Doğruluk: {accuracy:.4f}")
            print(f"  -> Toplam Süre: {total_duration:.2f} saniye")
            print(f"  -> HOG Vektör Boyutu: {X_subset.shape[1]}")
            
            results.append({
                'Boyut': f"{target_size}x{target_size}",
                'Doğruluk': round(accuracy, 4),
                'Toplam Süre (s)': round(total_duration, 2),
                'HOG Vektör Boyutu': X_subset.shape[1]
            })

        finally:
            # --- 5. Temizlik ---
            print(f"\nGeçici klasör '{temp_output_path.name}' siliniyor...")
            if temp_output_path.exists():
                shutil.rmtree(temp_output_path)

# --- Deney Sonu ---
print("\n" + "="*80)
print("DENEY TAMAMLANDI - SONUÇ ÖZETİ")
print("="*80)

if results:
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
else:
    print("Hiçbir deney başarıyla tamamlanamadı.")