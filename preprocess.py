import os
import cv2
import dlib
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Klasör ayarları
SCRIPT_DIR = Path(__file__).parent.resolve()

# Klasör ayarları (SCRIPT_DIR'e göre)
input_root = SCRIPT_DIR / "archive" / "dataset"
output_root = SCRIPT_DIR / "processed_dataset"
TARGET_SIZE = 224
SCALE_FACTOR = 1.5


# Yüz dedektörü (dlib)
face_detector = dlib.get_frontal_face_detector()
undetected = []

def apply_clahe(img_bgr):
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def detect_and_center_face(image_path, output_path):
    img = cv2.imread(str(image_path))
    if img is None:
        undetected.append(str(image_path))
        return

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rects = face_detector(rgb, 1)

    if not rects:
        undetected.append(str(image_path))
        return

    rect = rects[0]
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()

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

    face_crop = apply_clahe(face_crop)

    h_crop, w_crop = face_crop.shape[:2]
    scale = TARGET_SIZE / max(h_crop, w_crop)
    resized = cv2.resize(face_crop, (int(w_crop * scale), int(h_crop * scale)))

    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    y_off = (TARGET_SIZE - resized.shape[0]) // 2
    x_off = (TARGET_SIZE - resized.shape[1]) // 2
    canvas[y_off:y_off + resized.shape[0], x_off:x_off + resized.shape[1]] = resized

    os.makedirs(output_path.parent, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)

# Görsel uzantıları
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]
all_images = [p for p in input_root.rglob("*") if p.suffix.lower() in image_extensions]
print(f"{len(all_images)} resim bulundu.")

# İşlem
for img_path in tqdm(all_images, desc="İşleniyor"):
    rel_path = img_path.relative_to(input_root)
    out_path = output_root / rel_path
    detect_and_center_face(img_path, out_path)

# Kayıt
with open("undetected_faces.json", "w") as f:
    json.dump(undetected, f, indent=2)

print(f"\nİşlem tamamlandı. {len(undetected)} resimde yüz algılanamadı.")
print(f"Çıktılar şu klasöre kaydedildi: {output_root}")
