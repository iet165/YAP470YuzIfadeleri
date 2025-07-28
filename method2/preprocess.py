import os
import cv2
import dlib
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- AYARLAR ---
SAVE_AS_GRAYSCALE = True
TARGET_SIZE = 128
SCALE_FACTOR = 1.5

# --- YOL TANIMLAMALARI ---
# Bu script'in bulunduğu klasör
try:
    SCRIPT_DIR = Path(__file__).parent.resolve()
except NameError:
    SCRIPT_DIR = Path().resolve() # Jupyter Notebook için fallback

# Girdi ve Çıktı Klasörleri
input_root = SCRIPT_DIR / "fane_data"
if SAVE_AS_GRAYSCALE:
    output_root = SCRIPT_DIR / "processed_dataset_gray"
else:
    output_root = SCRIPT_DIR / "processed_dataset_color"

# --- dlib Yüz Dedektörünü Yükle ---
face_detector = dlib.get_frontal_face_detector()

# Yüzü bulunamayan resimleri saklamak için liste
undetected = []

# --- Fonksiyonlar ---

def apply_clahe(img_bgr):
    """Görüntüye CLAHE kontrast filtresi uygular."""
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def detect_and_center_face(image_path, output_path):
    """
    Bir görüntüdeki yüzü bulur, kırpar, işler ve belirtilen yola kaydeder.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        undetected.append(str(image_path))
        return

    # dlib, RGB formatında çalışır
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rects = face_detector(rgb, 1)

    if not rects:
        undetected.append(str(image_path))
        return

    # Genellikle ilk bulunan yüz en belirgin olanıdır
    rect = rects[0]
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()

    # Kırpma kutusunu genişlet
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2
    new_w, new_h = int(w * SCALE_FACTOR), int(h * SCALE_FACTOR)
    nx1 = max(cx - new_w // 2, 0)
    ny1 = max(cy - new_h // 2, 0)
    nx2 = min(cx + new_w // 2, img.shape[1])
    ny2 = min(cy + new_h // 2, img.shape[0])

    face_crop = img[ny1:ny2, nx1:nx2]
    if face_crop.size == 0:
        undetected.append(str(image_path))
        return
    
    # Kontrastı iyileştir
    face_crop = apply_clahe(face_crop)

    # Orantıyı koruyarak yeniden boyutlandır ve siyah tuvale yerleştir
    h_crop, w_crop = face_crop.shape[:2]
    scale = TARGET_SIZE / max(h_crop, w_crop)
    resized = cv2.resize(face_crop, (int(w_crop * scale), int(h_crop * scale)))

    if SAVE_AS_GRAYSCALE:
        final_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    else:
        final_img = resized
        canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)

    y_off = (TARGET_SIZE - final_img.shape[0]) // 2
    x_off = (TARGET_SIZE - final_img.shape[1]) // 2
    canvas[y_off:y_off + final_img.shape[0], x_off:x_off + final_img.shape[1]] = final_img

    # Çıktı klasörünün var olduğundan emin ol ve resmi kaydet
    os.makedirs(output_path.parent, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)

# --- Ana İşlem Bloğu ---

if __name__ == "__main__":
    print(f"Girdi klasörü: {input_root}")
    print(f"Çıktı klasörü: {output_root}")

    # Girdi klasöründeki tüm resim dosyalarını bul
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]
    all_images = [p for p in input_root.rglob("*") if p.suffix.lower() in image_extensions]
    
    if not all_images:
        print(f"HATA: '{input_root}' klasöründe hiç resim dosyası bulunamadı.")
    else:
        print(f"{len(all_images)} resim bulundu. Ön işleme başlıyor...")
        
        # Her bir resim için işlemi başlat
        for img_path in tqdm(all_images, desc="İşleniyor"):
            # Orijinal dosyanın göreceli yolunu al (örn: "happy/happy340.jpg")
            rel_path = img_path.relative_to(input_root)
            
            # Çıktı yolunu oluştururken uzantıyı tutarlı olarak .png yap
            # Örn: "processed_dataset_gray/happy/happy340.png"
            out_path = output_root / rel_path.with_suffix('.png')
            
            # Ana fonksiyonu çağır
            detect_and_center_face(img_path, out_path)

        # Sonuçları raporla
        print(f"\nİşlem tamamlandı.")
        print(f"{len(undetected)} resimde yüz algılanamadı.")
        
        # Yüz bulunamayan dosyaları bir JSON dosyasına kaydet
        if undetected:
            undetected_file_path = SCRIPT_DIR / "undetected_faces.json"
            with open(undetected_file_path, "w") as f:
                json.dump(undetected, f, indent=2)
            print(f"Detaylar '{undetected_file_path}' dosyasına kaydedildi.")
            
        print(f"İşlenmiş dosyalar şu klasöre kaydedildi: {output_root}")