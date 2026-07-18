import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# ==========================================
# 1. กำหนดพาธโฟลเดอร์
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# โฟลเดอร์รูปต้นฉบับ (test_SkinSafe/image ในโปรเจกต์นี้)
INPUT_FOLDER = os.path.join(CURRENT_DIR, "test_SkinSafe", "image")

# ต้องชื่อ "augmented_dataset" ให้ตรงกับ AUGMENTED_DIR ใน ocr_core.py
# (สคริปต์ประเมินผล/preprocess ทั้งหมดอ่านจากพาธนี้)
OUTPUT_BASE_FOLDER = os.path.join(CURRENT_DIR, "augmented_dataset")

# ==========================================
# 2. นิยามการแปลงรูป จำลองสภาพแสง/การถ่ายภาพจริงของผู้ใช้ 4 สถานการณ์
# ==========================================

# มือสั่น/โฟกัสไม่ทัน เวลาถ่ายฉลากใกล้ ๆ ด้วยมือถือ
blur_transform = A.Compose([
    A.OneOf([
        A.MotionBlur(blur_limit=(7, 15), allow_shifted=True, p=0.45),   # มือสั่นตอนกดชัตเตอร์
        A.GaussianBlur(blur_limit=(5, 9), p=0.35),                       # โฟกัสไม่ชัด
        A.Defocus(radius=(3, 6), alias_blur=(0.1, 0.3), p=0.2),          # เลนส์เบลอเวลาโฟกัสใกล้เกินไป
    ], p=1.0),
    A.Downscale(scale_range=(0.5, 0.75), p=0.4),   # รายละเอียดหายเวลากล้องปรับโฟกัสเร็ว ๆ
    A.ImageCompression(quality_range=(40, 70), p=0.6),  # บีบอัด JPEG ตอนส่งรูป
])

# ถ่ายในที่แสงน้อย (ร้าน/บ้านไฟสลัว) แต่ยังไม่มืดสนิทแบบกลางคืน
low_light_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.35, -0.15), contrast_limit=(-0.2, -0.05), p=1.0),
    A.RandomGamma(gamma_limit=(80, 115), p=0.6),
    A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.2, 0.5), p=0.7),  # กล้องดันไอเอสโอสู้แสง
    A.ImageCompression(quality_range=(50, 80), p=0.5),
])

# ถ่ายกลางคืน แสงน้อยมาก มักมีแหล่งแสงจุดเดียว (ไฟถนน/ไฟมือถือ) และมือสั่นเพราะสปีดชัตเตอร์ต่ำ
night_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.55, -0.35), contrast_limit=(-0.3, -0.1), p=1.0),
    A.RandomGamma(gamma_limit=(70, 110), p=0.6),
    A.ISONoise(color_shift=(0.02, 0.06), intensity=(0.4, 0.8), p=0.85),  # ไอเอสโอสูง เกรนหนัก
    A.MotionBlur(blur_limit=(5, 9), p=0.5),                              # สปีดชัตเตอร์ต่ำ มือสั่น
    A.RGBShift(r_shift_limit=(5, 15), g_shift_limit=(-2, 5), b_shift_limit=(-10, 0), p=0.4),  # แสงไฟโซเดียม/ไฟถนนออกส้ม
    A.ImageCompression(quality_range=(35, 60), p=0.6),
])

# ถ่ายในร่ม แสงไฟหลอด (เหลือง/ขาว) แสงไม่สม่ำเสมอ อาจมีเงาจากมือ/สิ่งของบัง
indoor_transform = A.Compose([
    A.RGBShift(r_shift_limit=(10, 25), g_shift_limit=(0, 10), b_shift_limit=(-20, -5), p=1.0),  # แสงหลอดไส้/วอร์มไวท์
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
    A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), shadow_dimension=5, p=0.4),  # เงาบังแสงบางส่วน
    A.ImageCompression(quality_range=(55, 85), p=0.5),
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

all_files = [f for f in os.listdir(INPUT_FOLDER) if os.path.isfile(os.path.join(INPUT_FOLDER, f))]

image_files = []
for f in all_files:
    if f.startswith('.') or f.endswith('.py'):
        continue

    test_img = cv2.imread(os.path.join(INPUT_FOLDER, f))
    if test_img is not None:
        image_files.append(f)

print(f"📁 โค้ดกำลังสแกนหาภาพที่อยู่ในพาธ: {INPUT_FOLDER}")
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

print(f"🚀 เสร็จแล้ว! ตรวจสอบรูปภาพแยกตามสถานการณ์ได้ที่โฟลเดอร์: {OUTPUT_BASE_FOLDER}")
