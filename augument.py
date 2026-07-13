import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# ==========================================
# 1. กำหนดพาธโฟลเดอร์ที่ถูกต้อง
# ==========================================
# ชี้ไปที่โฟลเดอร์ image ที่เก็บรูปภาพของคุณไว้
INPUT_FOLDER = "/Users/borfor/Downloads/test_SkinSafe/image"  

# ชี้ไปที่โฟลเดอร์ปลายทาง (จะสร้างโฟลเดอร์ผลลัพธ์ไว้ข้าง ๆ โค้ดของคุณ)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE_FOLDER = os.path.join(CURRENT_DIR, "augmented_dataset_result")

# ==========================================
# 2. นิยามการแปลงรูป
# ==========================================
blur_transform = A.Compose([
    A.OneOf([
        A.MotionBlur(blur_limit=5, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    ], p=1.0)
])

low_light_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.15), contrast_limit=(-0.15, 0), p=1.0),
])

night_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.45, -0.25), contrast_limit=(-0.25, -0.1), p=1.0),
    A.GaussNoise(p=0.5), 
    A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=0.5)
])

indoor_transform = A.Compose([
    A.RGBShift(r_shift_limit=(15, 25), g_shift_limit=(5, 15), b_shift_limit=(-25, -15), p=1.0),
    A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.05), contrast_limit=(-0.05, 0.05), p=1.0),
    A.RandomSunFlare(flare_roi=(0, 0, 1, 1), src_radius=80, p=0.4) 
])

scenarios = {
    "blur": blur_transform,
    "low_light": low_light_transform,
    "night": night_transform,
    "indoor": indoor_transform
}

# ==========================================
# 3. เริ่มรันกระบวนการแปลงรูป
# ==========================================

for scenario_name in scenarios.keys():
    os.makedirs(os.path.join(OUTPUT_BASE_FOLDER, scenario_name), exist_ok=True)

# ดึงไฟล์ทั้งหมดที่อยู่ในโฟลเดอร์เดียวกับโค้ด
all_files = [f for f in os.listdir(INPUT_FOLDER) if os.path.isfile(os.path.join(INPUT_FOLDER, f))]

image_files = []
for f in all_files:
    # ข้ามไฟล์ระบบและไฟล์โค้ดตัวเอง
    if f.startswith('.') or f.endswith('.py'):
        continue
    
    # เช็กว่าเป็นรูปภาพไหม
    test_img = cv2.imread(os.path.join(INPUT_FOLDER, f))
    if test_img is not None:
        image_files.append(f)

print(f"📁 โค้ดกำลังสแกนหาภาพที่อยู่เคียงข้างมันในพาธ: {INPUT_FOLDER}")
print(f"🎉 เจอรูปภาพต้นฉบับที่ใช้งานได้จริงทั้งหมด {len(image_files)} รูป กำลังเริ่มทำ Augmentation...")

for file_name in tqdm(image_files):
    img_path = os.path.join(INPUT_FOLDER, file_name)
    image = cv2.imread(img_path)
    
    if image is None:
        continue
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for scenario_name, transform in scenarios.items():
        augmented = transform(image=image_rgb)
        aug_img_rgb = augmented['image']
        
        aug_img_bgr = cv2.cvtColor(aug_img_rgb, cv2.COLOR_RGB2BGR)
        
        new_file_name = f"{scenario_name}_{file_name}"
        save_path = os.path.join(OUTPUT_BASE_FOLDER, scenario_name, new_file_name)
        
        cv2.imwrite(save_path, aug_img_bgr)

print(f"🚀 เสร็จแล้ว! ตรวจสอบรูปภาพแยกตามสถานการณ์ได้ที่โฟลเดอร์: augmented_dataset_result")