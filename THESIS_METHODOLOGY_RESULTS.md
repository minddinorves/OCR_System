# รายงานวิธีการทดลองและผลลัพธ์โดยละเอียด: ระบบ OCR สำหรับฉลากส่วนผสมเครื่องสำอาง

เอกสารนี้ขยายความจาก `EXPERIMENT_SUMMARY.md` โดยลงรายละเอียดที่มาที่ไปและวิธีคิด/สูตรคำนวณของทุกขั้นตอน อ้างอิงตรงจากซอร์สโค้ดจริงในโปรเจกต์ เพื่อให้นำไปอ้างอิงเขียนบทระเบียบวิธีวิจัย (methodology) ในวิทยานิพนธ์ได้ครบถ้วน **ไม่รวมผลการทดลอง GOT-OCR2.0** (ดูเรื่องนั้นแยกใน `EXPERIMENT_SUMMARY.md` ข้อ 8)

---

## 1. ที่มาของโจทย์วิจัย

ระบบต้นแบบที่ใช้อ้างอิง (SkinSafe BOT) ออกแบบ pipeline การอ่านฉลากส่วนผสมเครื่องสำอางด้วยแนวคิด "ทำ image preprocessing หนัก ๆ ก่อนส่งเข้า OCR" เพื่อรับมือกับภาพถ่ายจากผู้ใช้ทั่วไปที่มักมีคุณภาพต่ำ (ถ่ายในที่แสงน้อย, สั่น/เบลอ, มุมเอียง) เอกสารต้นฉบับ (PDF ของ SkinSafe BOT) ระบุ pipeline ไว้ 5 ขั้น ได้แก่ deskew, contrast enhancement แบบ adaptive, bicubic upscaling ถึง 800px, tiled OCR, และ 4-tier fallback พร้อม super-resolution

งานทดลองนี้ตั้งคำถามเชิงประจักษ์ว่า **แนวทาง preprocessing หนัก ๆ ตาม spec ดังกล่าวช่วยให้ OCR อ่านฉลากได้แม่นยำขึ้นจริงหรือไม่** เมื่อเทียบกับ (ก) การไม่ทำอะไรเลย (baseline) และ (ข) เทคนิค classical preprocessing อื่น ๆ ที่ใช้กันทั่วไปในงาน OCR (grayscale, threshold, denoise, sharpen, upscale ธรรมดา) จึงได้ทดลองสร้างทุกเทคนิคขึ้นมาเองแล้ววัดผลบนชุดข้อมูลเดียวกันอย่างเป็นธรรม (controlled comparison)

---

## 2. ชุดข้อมูล (Dataset) — ที่มาและวิธีสร้าง

### 2.1 ภาพต้นฉบับและ ground-truth

- **ภาพต้นฉบับ**: `test_SkinSafe/image/*.jpg` — 137 ไฟล์ เป็นภาพถ่ายฉลากส่วนผสมเครื่องสำอางจริงที่ถูก crop มาเฉพาะส่วนที่มีรายชื่อสารเคมี (ไม่ใช่ภาพเต็มขวด) ขนาดภาพหลากหลายและเล็กผิดปกติเมื่อเทียบกับภาพถ่ายทั่วไป: **ความสูงต่ำสุด 66px, สูงสุด 1000px, มัธยฐาน (median) 163px** (n=136 รูป, วัดจริงด้วย `cv2.imread` แล้วอ่าน `shape[0]`)
- **Ground-truth**: `test_SkinSafe/label/label<ชื่อไฟล์>.txt` — 136 ไฟล์ (ขาด 1 คู่: `A26.jpg` ไม่มี label คู่กัน จึงถูกตัดออกจากการคำนวณ CER/WER ในทุกเทคนิคเพื่อให้จำนวนตัวอย่างเท่ากันทุกเทคนิค → **n_scored = 136 ภาพ/scenario เสมอ**) แต่ละไฟล์คือรายชื่อสาร (INCI name) หนึ่งชื่อต่อบรรทัด ได้จากการถอดข้อความด้วยมือ/ผู้ช่วย AI จากภาพฉลากจริง
  - ไฟล์ label รุ่นเก่าบางไฟล์มี prefix ตัวเลขนำหน้าแบบ `1\tIngredient Name` (คั่นด้วย tab) โค้ดจะตรวจและตัด prefix ตัวเลขนี้ทิ้งก่อน หากตัวแรกก่อน tab เป็นตัวเลขล้วน (`ocr_core.py: load_reference_text`)
  - **Reference text ที่ใช้เทียบ**: นำชื่อสารทุกชื่อในไฟล์มาต่อกันด้วย `, ` (comma + space) เป็น string เดียว — **นี่คือจุดสำคัญที่ทำให้ค่า CER/WER สัมบูรณ์สูงกว่างานวิจัย OCR ทั่วไป** เพราะ format นี้ต่างจากการจัดวางตัวอักษรจริงบนฉลาก (ซึ่งอาจตัดบรรทัด ใช้เครื่องหมายวรรคตอนต่างกัน) ดูข้อ 8 (ข้อจำกัด)

### 2.2 ภาพ augmented (จำลองสภาพถ่ายภาพจริงที่แย่)

สร้างด้วย `augument.py` โดยใช้ไลบรารี **albumentations** ผลิต 4 scenario จากภาพต้นฉบับทุกภาพ (137 × 4 = 548 ภาพ) รวมกับภาพต้นฉบับ (original) เป็น **5 scenario ทั้งหมด, รวม 137×5 = 685 ภาพ** (685 หลังตัด A26.jpg คือ 680 ภาพที่ถูกให้คะแนนจริง)

พารามิเตอร์ transform ที่ใช้จริงในโค้ด (ค่า limit เป็นสัดส่วนที่ albumentations สุ่มในช่วงนั้นต่อภาพ):

| Scenario | Transform (`A.Compose`) | รายละเอียดพารามิเตอร์ |
|---|---|---|
| **blur** | `A.OneOf([MotionBlur, GaussianBlur], p=1.0)` | สุ่มเลือก 1 ใน 2 แบบเสมอ (ให้น้ำหนักเท่ากัน p=0.5 ต่อแบบภายใน OneOf): Motion Blur (`blur_limit=5`) หรือ Gaussian Blur (`blur_limit=(3,5)`) จำลองภาพสั่น/โฟกัสไม่ชัด |
| **low_light** | `RandomBrightnessContrast(brightness_limit=(-0.3,-0.15), contrast_limit=(-0.15,0), p=1.0)` | ลดความสว่างลง 15-30% และลดคอนทราสต์ลง 0-15% (สุ่มในช่วงนี้ทุกภาพ) |
| **night** | `RandomBrightnessContrast(brightness_limit=(-0.45,-0.25), contrast_limit=(-0.25,-0.1), p=1.0)` + `GaussNoise(p=0.5)` + `CLAHE(clip_limit=1.5, tile_grid_size=(8,8), p=0.5)` | มืดกว่า low_light ชัดเจน (ลดสว่าง 25-45%) บวก **สุ่ม 50%** ใส่ Gaussian noise และ **สุ่ม 50%** ใส่ CLAHE เบา ๆ — เป็น scenario ที่สภาพแย่ที่สุดในชุดข้อมูล |
| **indoor** | `RGBShift(r_shift=(15,25), g_shift=(5,15), b_shift=(-25,-15), p=1.0)` + `RandomBrightnessContrast(brightness_limit=(-0.05,0.05), contrast_limit=(-0.05,0.05), p=1.0)` + `RandomSunFlare(flare_roi=(0,0,1,1), src_radius=80, p=0.4)` | จำลองแสงในร่ม/หลอดไฟสีเหลือง: เพิ่มช่องแดง/เขียว ลดช่องน้ำเงิน (สีอมเหลือง) + จิตเตอร์ความสว่าง/คอนทราสต์เล็กน้อย + **สุ่ม 40%** มี lens flare ปรากฏในเฟรม |

หมายเหตุ: แต่ละภาพจะสุ่มค่าภายในช่วง limit ที่กำหนดครั้งเดียวตอนรัน (`augmented_dataset/` ที่ใช้จริงถูก generate ไว้ล่วงหน้าแล้ว ไม่ได้สุ่มใหม่ทุกครั้งที่รัน OCR) ดังนั้นทุกเทคนิค OCR ที่ทดสอบจะเห็นภาพ augmented ชุดเดียวกันเป๊ะ ๆ — เปรียบเทียบระหว่างเทคนิคได้แบบ apples-to-apples

---

## 3. เกณฑ์การวัดผล (Evaluation Metrics) — สูตรและวิธีคำนวณละเอียด

โค้ดคำนวณอยู่ใน `ocr_core.py` (`normalize_text`, `compute_cer`, `compute_wer`)

### 3.1 การ normalize ข้อความก่อนเทียบ

```python
def normalize_text(text):
    text = text.lower().strip()          # แปลงเป็นตัวพิมพ์เล็กทั้งหมด + ตัดช่องว่างหัวท้าย
    text = re.sub(r"\s+", " ", text)     # รวม whitespace ติดกันหลายตัวให้เหลือ space เดียว
    return text
```
ทำแบบนี้กับทั้ง OCR output (hypothesis) และ reference text ก่อนคำนวณเสมอ เพื่อไม่ให้ case-sensitivity หรือช่องว่างเกิน (ที่ไม่ใช่ความผิดพลาดจริงของ OCR) กระทบคะแนน

### 3.2 CER (Character Error Rate)

```
CER = Levenshtein_distance(normalize(hypothesis), normalize(reference)) / len(normalize(reference))
```
- **Levenshtein distance** = จำนวนการแก้ไข "ตัวอักษรเดี่ยว" ขั้นต่ำ (insert / delete / substitute) ที่ต้องทำเพื่อเปลี่ยน hypothesis ให้เป็น reference — คำนวณด้วย `rapidfuzz.distance.Levenshtein.distance` (dynamic programming แบบมาตรฐาน)
- หารด้วยความยาว (จำนวนตัวอักษร) ของ reference ที่ normalize แล้ว
- **ค่ายิ่งน้อยยิ่งดี**: 0 = อ่านถูกทุกตัวอักษร, ค่า > 1 เป็นไปได้ถ้า OCR อ่านออกมายาว/ผิดเพี้ยนมากกว่าความยาวจริงของ reference (เช่น อ่านซ้ำ หรือแต่งข้อความยาวเกิน)

### 3.3 WER (Word Error Rate)

```
WER = Levenshtein_distance(normalize(hypothesis).split(), normalize(reference).split()) / len(normalize(reference).split())
```
สูตรเดียวกับ CER ทุกประการ แต่ตัดสตริงเป็น "คำ" (แบ่งด้วย whitespace) ก่อน แล้วคำนวณ edit distance ระดับคำแทนระดับตัวอักษร (คำที่สะกดผิดแม้ 1 ตัวอักษรจะนับเป็น 1 word error เต็ม ไม่ใช่ partial credit — ทำให้ WER มักสูงกว่า CER เสมอในชุดข้อมูลนี้)

### 3.4 การรวมผลลัพธ์

CER/WER คำนวณแยกต่อภาพ 1 ค่า แล้วนำมาหาค่าเฉลี่ยเลขคณิต (arithmetic mean) แยกตาม scenario และแยกตามเทคนิค (`summarize_results.py`) — ไม่ใช่การรวม edit distance รวมทุกภาพก่อนหารทีเดียว (micro-average) แต่เป็นการเฉลี่ยคะแนนต่อภาพ (macro-average) ซึ่งหมายความว่าทุกภาพมีน้ำหนักเท่ากันไม่ว่า reference จะยาวสั้นแค่ไหน

### 3.5 เหตุผลที่ค่า CER/WER สัมบูรณ์ดูสูงผิดปกติ (0.55-0.95) เทียบกับงาน OCR ทั่วไป (มักรายงาน CER < 0.1)

1. Reference text ต่อด้วย comma ต่างจาก line-break/ตัวคั่นจริงบนฉลาก (ข้อ 2.1)
2. ภาพในชุดข้อมูลนี้มีความละเอียดต่ำมาก (median สูง 163px) ต่ำกว่าภาพถ่ายฉลากทั่วไปมาก

ดังนั้น **ตัวเลขเหล่านี้ควรใช้เพื่อเปรียบเทียบ "ระหว่างเทคนิค" บนชุดข้อมูลเดียวกัน** ไม่ใช่นำไปเทียบตรง ๆ กับตัวเลขจากงานวิจัยอื่นที่ใช้ reference format ต่างกัน

---

## 4. เครื่องมือ OCR และการตั้งค่า

- **PaddleOCR 3.7.0**, โมเดล **PP-OCRv6_medium** (มีทั้ง text detection + text recognition ในตัว) รันบน GPU (NVIDIA RTX 4050 Laptop, CUDA 12.6, cuDNN 9.9, paddlepaddle-gpu 3.2.2)
- การตั้งค่า (`ocr_core.py: get_ocr`):
  ```python
  PaddleOCR(
      lang="en",
      use_doc_orientation_classify=False,  # ปิดการหมุนเอกสารอัตโนมัติในตัว
      use_doc_unwarping=False,             # ปิดการยืด/แก้บิดเบี้ยวเอกสารอัตโนมัติในตัว
      use_textline_orientation=False,      # ปิดการตรวจทิศทางบรรทัดข้อความอัตโนมัติในตัว
  )
  ```
  ปิด 3 ฟีเจอร์ในตัวนี้โดยตั้งใจ เพื่อให้ผลของ "preprocessing ที่เราเขียนเอง" (เช่น deskew) เห็นผลชัดเจน ไม่ถูกฟีเจอร์ default ของ PaddleOCR กลบ/ซ้ำซ้อนกัน
- โมเดล PaddleOCR ถูก cache เป็น singleton (`_OCR_INSTANCES`) และใช้ซ้ำข้ามทุกภาพในรอบทดลองเดียวกัน เพราะการโหลดโมเดลใช้เวลานาน — เวลาที่วัด (`ocr_time_sec`) จึงเป็นเวลา **inference ต่อภาพเท่านั้น** ไม่รวมเวลาโหลดโมเดลครั้งแรก
- ผลลัพธ์ดิบจาก `ocr.predict(image)` ให้ list ของ `rec_texts` (ข้อความที่ recognize ได้ต่อกล่องข้อความที่ detect เจอ) — เทคนิคส่วนใหญ่นำมาต่อกันด้วยช่องว่าง (`" ".join(lines)`) เป็น 1 string เพื่อเทียบกับ reference

---

## 5. รายละเอียดเทคนิคทั้งหมด (13 เทคนิค) — อัลกอริทึมและพารามิเตอร์จริงจากโค้ด

### 5.1 Baseline — `ocr_paddle_only.py`
ส่งภาพต้นฉบับ (path ไฟล์ตรง ๆ) เข้า `PaddleOCR.predict()` โดยไม่มีการแต่งภาพใด ๆ ก่อน — ใช้เป็นเส้นฐานเปรียบเทียบทุกเทคนิคอื่น

### 5.2 Fuzzy matching (post-processing) — `ocr_paddle_fuzzy.py`
ทำงาน "หลัง" OCR อ่านเสร็จแล้ว ไม่แตะภาพเลย:
1. รวมทุกบรรทัดที่ OCR อ่านได้ ตัด header เช่น "Ingredients:" ออกด้วย regex `^[^:：]*ingredient[^:：]*[:：]\s*`
2. แบ่งข้อความเป็น token ด้วย regex `,(?!\s*\d)` — ตัดที่ comma แต่ **ไม่ตัด** ถ้า comma ตามด้วยตัวเลข (กันไม่ให้ตัดกลางชื่อสารที่มีตัวเลข เช่น "1,2-Hexanediol")
3. แต่ละ token (ที่ยาว ≥ 3 ตัวอักษร) จะถูกค้นหาชื่อที่ใกล้เคียงที่สุดในฐานข้อมูล INCI (`ingredient_master_dataset_fixed.csv`, 11,091 ชื่อไม่ซ้ำ) ด้วย `rapidfuzz.process.extractOne(token, vocabulary, scorer=fuzz.ratio, score_cutoff=85)` — **threshold ค่า default = 85/100**
4. รับ match นั้นก็ต่อเมื่อผ่านเงื่อนไข "reliable match" เพิ่มเติม: อัตราส่วนความยาว (ยาวกว่า/สั้นกว่า) ระหว่าง token กับชื่อที่ match ต้อง ≤ 1.6 เท่า (กันไม่ให้ token สั้น ๆ ถูกจับคู่ผิดกับชื่อที่ยาวกว่ามากหรือสั้นกว่ามาก)
5. ถ้าไม่ผ่าน threshold หรือ length ratio ก็เก็บ token เดิมไว้ (ไม่แก้)
6. บันทึกทั้ง `cer_raw`/`wer_raw` (ก่อนแก้) และ `cer_fuzzy`/`wer_fuzzy` (หลังแก้) เพื่อเห็นผลต่างชัดเจน

### 5.3 Grayscale — `preprocess_grayscale.py`
`cv2.cvtColor(BGR→GRAY)` แล้วแปลงกลับเป็น 3-channel (`GRAY→BGR`) เพราะ PaddleOCR ต้องการภาพ 3 ช่องสี — ไม่ได้เพิ่มข้อมูลใหม่ เพียงตัด channel สีทิ้ง

### 5.4 Threshold (Otsu Binarization) — `preprocess_threshold.py`
Grayscale แล้ว `cv2.threshold(gray, 0, 255, THRESH_BINARY + THRESH_OTSU)` — Otsu algorithm หาค่า threshold ที่แบ่งภาพเป็นขาว-ดำโดยอัตโนมัติ (maximize between-class variance ของ histogram) ได้ภาพ binary ล้วน

### 5.5 Denoise — `preprocess_denoise.py`
`cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)` — Non-Local Means denoising: เทียบ patch ขนาด 7×7 พิกเซลกับ patch อื่นในหน้าต่างค้นหา 21×21 พิกเซล แล้วเฉลี่ยถ่วงน้ำหนักตามความคล้าย (`h`/`hColor` = filter strength สำหรับ luminance/color)

### 5.6 Sharpen (Unsharp Masking) — `preprocess_sharpen.py`
```
blurred = GaussianBlur(image, sigma=3)
sharpened = 1.5 × image − 0.5 × blurred   (clip เป็น [0,255])
```
สูตร unsharp mask มาตรฐาน: ดึงรายละเอียดความถี่สูงออกมาโดยลบภาพเบลอออกจากภาพจริง แล้วบวกกลับเข้าไปเพิ่มน้ำหนัก

### 5.7 Upscale (bicubic → 480px) — `preprocess_upscale.py`
ถ้าความสูงภาพ < 480px: ขยายด้วย `cv2.resize(..., interpolation=INTER_CUBIC)` ให้ความสูง = 480px (จำกัด scale factor สูงสุด 6.0 เท่า กันภาพเล็กจิ๋วขยายจนเบลอเกินไป) ภาพที่สูงอยู่แล้ว ≥480px ปล่อยผ่านไม่แตะ

### 5.8 Deskew (Projection Profile Variance Maximization) — `preprocess_deskew.py`
อ้างอิงเทคนิคจาก SkinSafe BOT PDF โดยตรง:
1. Grayscale + Otsu inverse binarization (`THRESH_BINARY_INV + THRESH_OTSU`) — ให้ตัวอักษรเป็นสีขาวบนพื้นดำ
2. ลองหมุนภาพ binary ทีละมุมตั้งแต่ -5° ถึง +5° ทีละ 0.5° (รวม 21 มุมที่ทดสอบ) ด้วย `cv2.warpAffine` (bicubic interpolation)
3. แต่ละมุม: sum พิกเซลตามแนวนอนทีละแถว (`rotated.sum(axis=1)`) ได้ profile 1 มิติ แล้วคำนวณ **variance** ของ profile นั้น
4. เลือกมุมที่ให้ variance สูงสุด — หลักการคือถ้าบรรทัดข้อความเรียงแนวนอนพอดี (ไม่เอียง) row-sum จะมีค่าสูงต่ำสลับกันชัดเจน (แถวที่มีตัวอักษรเยอะ vs แถวว่างระหว่างบรรทัด) → variance สูงสุด
5. หมุนภาพสี (ไม่ใช่ binary) จริงด้วยมุมที่เลือก โดย fill พื้นหลังว่างด้วยสีขาว (`borderValue=(255,255,255)`)

### 5.9 Contrast Adaptive (CLAHE/HistEq ตามความสว่าง) — `preprocess_contrast_adaptive.py`
อ้างอิง SkinSafe BOT PDF ขั้นที่ 2: วัดค่าความสว่างเฉลี่ยของภาพ grayscale (`gray.mean()`) — ถ้า **≥ 127** (ภาพสว่าง) ใช้ **CLAHE** (`clipLimit=2.0, tileGridSize=(8,8)`, Contrast Limited Adaptive Histogram Equalization — ปรับ histogram เฉพาะที่เป็น tile ย่อย ๆ พร้อมจำกัด contrast ไม่ให้ noise ถูกขยายเกิน) — ถ้า **< 127** (ภาพมืด) ใช้ **Histogram Equalization ธรรมดา** (`cv2.equalizeHist`, กระจาย histogram ทั้งภาพให้เต็มช่วง 0-255)

### 5.10 Upscale800 (bicubic → 800px ด้านสั้น) — `preprocess_upscale800.py`
อ้างอิง SkinSafe BOT PDF ขั้นที่ 3 ตรง ๆ: ขยายภาพด้วย bicubic ให้ **ด้านที่สั้นกว่า** (ไม่ใช่แค่ความสูง) ≥ 800px (จำกัด scale สูงสุด 12.0 เท่า) — ภาพที่ด้านสั้นอยู่แล้ว ≥800px ปล่อยผ่าน

### 5.11 Tiled OCR — `preprocess_tiled_ocr.py`
อ้างอิง SkinSafe BOT PDF ขั้นที่ 4:
1. แบ่งภาพเป็นแถบแนวนอน สูงแถบละ 150px, ซ้อนทับกัน (overlap) 40px ต่อแถบ (stride = 150−40 = 110px ต่อครั้ง) — **ปรับค่านี้เองจาก spec เดิม** (ซึ่งอิงภาพสูง 800px) ให้เหมาะกับภาพในชุดข้อมูลนี้ที่ median สูงแค่ 163px เพื่อให้ tiling เกิดขึ้นจริงกับภาพส่วนใหญ่
2. รัน OCR แยกแต่ละแถบ เก็บกล่องข้อความ (`rec_boxes`) พร้อมแปลงพิกัด y กลับเป็นพิกัดภาพเต็ม (`by1 + y`, `by2 + y`)
3. **ลบข้อความซ้ำ** จากส่วนที่ overlap กัน ด้วย Intersection-over-Union (IoU) ของกล่อง: เรียงกล่องตาม confidence score จากมากไปน้อย แล้วเก็บกล่องที่ไม่ทับซ้อน (IoU < 0.5) กับกล่องที่เก็บไว้แล้ว
4. เรียงกล่องที่เหลือจากบนลงล่าง/ซ้ายไปขวา (`sort by (y1, x1)`) แล้วต่อข้อความเป็น string เดียว
5. ถ้าภาพสูง ≤ 150px (ไม่เกิน 1 แถบ) จะไม่ tile เลย รัน OCR ปกติครั้งเดียว

### 5.12 Multi-Tier Fallback (4 ระดับ พร้อม EDSR Super-Resolution) — `preprocess_multitier_fallback.py`
อ้างอิง SkinSafe BOT PDF ขั้นที่ 5 ซึ่งเป็นเทคนิคซับซ้อนที่สุด ไล่ระดับ (escalate) เฉพาะเมื่อ tier ก่อนหน้า "อ่านได้น้อยเกินไป" — เกณฑ์นี้ไม่ได้ระบุในเอกสารต้นฉบับ จึงกำหนดขึ้นเองเป็น **น้อยกว่า 3 กล่องข้อความ (MIN_BOXES) หรือน้อยกว่า 20 ตัวอักษรรวม (MIN_CHARS)**:

- **Tier 1**: เลือกวิธี contrast ตามกฎเดียวกับข้อ 5.9 (brightness ≥127→CLAHE, <127→HistEq) แล้ว bicubic upscale ให้ด้านสั้น ≥800px (เหมือนข้อ 5.10) → OCR → ถ้าเพียงพอ หยุดและคืนผลลัพธ์
- **Tier 2**: ถ้า Tier 1 ไม่พอ → สลับไปใช้วิธี contrast อีกแบบ (ถ้า Tier 1 ใช้ CLAHE คราวนี้ใช้ HistEq หรือกลับกัน) + bicubic upscale เหมือนเดิม → OCR อีกครั้ง
- **Tier 3**: ถ้ายังไม่พอ → ใช้วิธี contrast แบบเดิมของ Tier 1 แต่แทนที่ bicubic ด้วย **EDSR super-resolution** (`cv2.dnn_superres`, โมเดล `models/EDSR_x4.pb`, scale ×4) — แต่ใช้ EDSR เฉพาะถ้าด้านสั้นของภาพ ≤ 250px เท่านั้น (เกินกว่านี้ EDSR ช้า/กิน memory เกินไป จึงใช้ bicubic แทน) หลัง EDSR แล้วยัง bicubic เก็บรายละเอียดให้ถึง 800px อีกที
- **Tier 4** (คำตอบสุดท้ายเสมอ ไม่มีเงื่อนไข): bicubic upscale ให้ถึง 800px ก่อน แล้วเช็คความสว่างเฉลี่ย — ถ้า mean <100 (มืด) จะ invert สี (`bitwise_not`) ก่อนสมมติว่าเป็นตัวอักษรสว่างบนพื้นมืด แล้วทำ **Adaptive Gaussian Thresholding** (`blockSize=25, C=15`)

แต่ละ tier ที่ลองจะรัน OCR จริง 1 ครั้ง (สูงสุด 4 ครั้งต่อภาพถ้าไล่ผ่านทุก tier) จึงช้ากว่า baseline มาก และเวลารวมที่บันทึกคือผลรวมเวลาของทุกครั้งที่เรียก OCR จริง

### 5.13 Deskew + Fuzzy matching (เทคนิคที่ดีที่สุด, รวม 2 แนวทาง) — `preprocess_best_plus_fuzzy.py`
เลือกจากผลการทดลอง 10 เทคนิค preprocessing เดี่ยว (ข้อ 5.3-5.12) ที่ deskew (5.8) ให้ mean CER ต่ำสุด (0.6086) จึงนำมาต่อกับ fuzzy matching (5.2) ซึ่งเป็นวิธีที่ดีที่สุดในกลุ่ม post-processing: **deskew ภาพก่อน → OCR → fuzzy-match ผลลัพธ์กับฐานข้อมูล INCI** เพื่อดูว่า 2 แนวทางที่ดีที่สุดเมื่อรวมกันจะให้ผลดีขึ้นแบบสะสม (stack) หรือไม่

---

## 6. ผลลัพธ์แบบละเอียด (mean CER ต่อ scenario ต่อเทคนิค, n=136 ภาพ/scenario)

| เทคนิค | original | blur | indoor | low_light | night | **mean CER (5 scenario)** | mean WER | avg เวลา/รูป (วิ) |
|---|---|---|---|---|---|---|---|---|
| Deskew + Fuzzy | 0.5592 | 0.6163 | 0.5669 | 0.5704 | 0.7260 | **0.6078** | 0.7435 | 0.146 |
| Deskew เดี่ยว | 0.5602 | 0.6176 | 0.5675 | 0.5702 | 0.7273 | 0.6086 | 0.7323 | 0.151 |
| Fuzzy เดี่ยว | 0.5729 | 0.6310 | 0.5700 | 0.5758 | 0.7390 | 0.6177 | 0.7549 | 0.145 |
| Baseline | 0.5743 | 0.6331 | 0.5713 | 0.5768 | 0.7402 | 0.6191 | 0.7423 | 0.163 |
| Tiled OCR | 0.5704 | 0.6297 | 0.5689 | 0.5762 | 0.7515 | 0.6193 | 0.7519 | 0.205 |
| Grayscale | 0.5862 | 0.6266 | 0.5794 | 0.5791 | 0.7378 | 0.6218 | 0.7377 | 0.149 |
| Denoise | 0.5825 | 0.7333 | 0.5857 | 0.5756 | 0.7491 | 0.6452 | 0.7880 | 0.161 |
| Contrast Adaptive | 0.5943 | 0.6524 | 0.5968 | 0.6305 | 0.7678 | 0.6484 | 0.7673 | 0.133 |
| Sharpen | 0.6433 | 0.6263 | 0.6123 | 0.5873 | 0.7726 | 0.6484 | 0.7636 | 0.164 |
| Upscale (480px) | 0.6389 | 0.6869 | 0.6366 | 0.6402 | 0.7792 | 0.6764 | 0.7800 | 0.221 |
| Multi-Tier Fallback | 0.6523 | 0.6991 | 0.6570 | 0.6579 | 0.7880 | 0.6909 | 0.8003 | 0.985 |
| Upscale800 | 0.6718 | 0.7016 | 0.6767 | 0.6727 | 0.8123 | 0.7070 | 0.8021 | 0.395 |
| Threshold (Otsu) | 0.8883 | 0.9038 | 0.8843 | 0.8912 | 0.9448 | 0.9025 | 0.9414 | 0.058 |

(ตัวเลข mean CER 5 scenario คือค่าเฉลี่ยแบบไม่ถ่วงน้ำหนักของ 5 คอลัมน์ scenario เพราะแต่ละ scenario มีจำนวนภาพเท่ากัน n=136)

---

## 7. การวิเคราะห์และข้อค้นพบสำคัญ

1. **ผลต่างระหว่างเทคนิคที่ดีที่สุดกับ baseline มีน้อยมาก** (0.6078 vs 0.6191 ≈ 1.8% CER) — ก่อนสรุปว่าเทคนิคใดเทคนิคหนึ่ง "ดีกว่า" จริงในเชิงสถิติ ควรทำ paired significance test (เช่น Wilcoxon signed-rank บนคู่ CER รายภาพ) เพิ่มเติม เพราะปัจจุบันมีแค่ค่าเฉลี่ย ยังไม่ได้พิสูจน์นัยสำคัญ
2. **Deskew เป็นเทคนิค preprocessing ภาพเดี่ยว ๆ เพียงหนึ่งเดียวที่ดีกว่า baseline ในทุก scenario** โดยไม่มีข้อยกเว้น
3. **Bicubic upscaling ทำให้แย่ลงอย่างสม่ำเสมอ** ทั้งแบบ 480px (CER 0.676) และ 800px ตาม spec จริง (CER 0.707) แม้ภาพในชุดข้อมูลนี้จะเล็กมาก (median สูง 163px) ซึ่งตามสมมติฐานควรได้ประโยชน์จากการขยายมากที่สุด — ข้อสันนิษฐาน: โมเดล PP-OCRv6 มีการ resize ภาพเข้า pipeline ภายในตัวเองอยู่แล้วก่อน inference การ bicubic ขยายภาพเพิ่มก่อนหน้านั้นจึงเพิ่มแค่ interpolation artifact/blur โดยไม่เพิ่มข้อมูลจริงให้โมเดล
4. **Multi-Tier Fallback (รวม EDSR Super-Resolution) แย่กว่า baseline ทั้งที่ซับซ้อนและช้าที่สุด** (เฉลี่ย 0.985 วินาที/รูป ช้ากว่า baseline ~6 เท่า เพราะเรียก OCR สูงสุด 4 ครั้ง/รูป) — การพยายามแก้ภาพมากขึ้นเรื่อย ๆ ไม่ได้แก้ปัญหาความแม่นยำ เพียงเปลี่ยนรูปแบบความผิดพลาด
5. **Threshold/Binarization (Otsu) แย่ที่สุดในทุกเทคนิค** (mean CER 0.9025) เพราะโมเดล PP-OCRv6 ถูกเทรนบนภาพสีจริง การตัดเหลือขาว-ดำล้วนทำลายข้อมูล texture/gradient ที่โมเดลใช้ตัดสินใจ
6. **Fuzzy matching ให้ผลดีขึ้นแบบสม่ำเสมอแต่เล็กน้อย** (baseline 0.6191→0.6177 เมื่อเพิ่ม fuzzy, deskew 0.6086→0.6078 เมื่อเพิ่ม fuzzy) เพราะเป็นการแก้ไขข้อความ "หลัง" OCR อ่านเสร็จ ไม่เปลี่ยนสิ่งที่โมเดลเห็น จึงช่วยได้แค่คำที่อ่านผิดเล็กน้อยระดับตัวอักษร (typo) แต่ช่วยไม่ได้ถ้า OCR อ่านพลาดไปเป็นคำอื่นไปเลย
7. **Tiled OCR ใกล้เคียง baseline มาก** (0.6193 vs 0.6191) เพราะภาพส่วนใหญ่ในชุดข้อมูล (median สูง 163px) สั้นกว่าหรือใกล้เคียง strip height ที่ตั้งไว้ (150px) ทำให้ส่วนใหญ่ไม่ถูกแบ่ง tile จริง ๆ — คาดว่าจะให้ผลต่างชัดกว่านี้ถ้าใช้กับภาพฉลากที่สูงมาก ๆ (เช่นถ่ายเต็มขวด)

### ข้อสรุปภาพรวม
สำหรับชุดข้อมูลภาพฉลากส่วนผสมที่ crop มาเป็นแถบเล็ก ความละเอียดต่ำ **การทำ image preprocessing แบบดั้งเดิม (classical) ส่วนใหญ่ไม่ช่วยหรือกลับทำให้โมเดล deep learning OCR อ่านแย่ลง** มีเพียง **deskew** และ **fuzzy matching หลัง OCR** เท่านั้นที่ให้ผลดีขึ้นอย่างสม่ำเสมอ (แม้เพียงเล็กน้อย) ผลลัพธ์นี้ขัดกับแนวทางออกแบบของระบบต้นแบบ (SkinSafe BOT) ที่เน้น preprocessing หนัก (CLAHE, bicubic upscale ถึง 800px, multi-tier fallback พร้อม super-resolution) — ชี้ให้เห็นว่า **ประสิทธิภาพของ preprocessing pipeline ขึ้นกับลักษณะภาพอินพุตและสถาปัตยกรรมโมเดล OCR ที่ใช้เป็นอย่างมาก ควรทดสอบเชิงประจักษ์ก่อนนำไปใช้จริง ไม่ควรอนุมานจากสมมติฐานทั่วไปเพียงอย่างเดียว**

---

## 8. ข้อจำกัดของการทดลอง (สำหรับระบุในวิทยานิพนธ์)

- **Reference text format**: ต่อรายชื่อสารด้วย comma ซึ่งอาจไม่ตรงกับรูปแบบตัวอักษรจริงบนฉลาก (ตัดบรรทัด/เครื่องหมายวรรคตอนต่างกัน) ทำให้ค่า CER/WER สัมบูรณ์สูงกว่าความเป็นจริงระดับหนึ่ง — แต่ใช้ reference เดียวกันทุกเทคนิคจึงเปรียบเทียบกันได้อย่างเป็นธรรม (fair relative comparison ไม่ใช่ absolute accuracy)
- **เกณฑ์ "อ่านได้น้อยเกินไป"** ใน Multi-Tier Fallback (< 20 ตัวอักษร หรือ < 3 กล่องข้อความ) เป็นค่าที่กำหนดขึ้นเอง เนื่องจากเอกสารต้นฉบับ (SkinSafe BOT) ไม่ได้ระบุไว้ชัดเจน
- **ขนาดภาพเล็กผิดปกติ** (สูง 66-1000px, median 163px) เมื่อเทียบกับภาพถ่ายฉลากผลิตภัณฑ์ทั่วไป ผลลัพธ์เรื่อง upscaling/tiled OCR อาจแตกต่างไปถ้าใช้กับภาพถ่ายความละเอียดสูงกว่านี้ (เช่นถ่ายเต็มขวดครบทุกด้าน)
- **A26.jpg** ไม่มีไฟล์ label คู่กัน จึงไม่ถูกนำมาคำนวณ CER/WER ในทุกเทคนิค (ทุกเทคนิคทดสอบบน 136 ภาพ/scenario, 680 ภาพรวมเท่ากัน)
- **Fuzzy matching ใช้ฐานข้อมูล INCI จริงจากภายนอก** (`ingredient_master_dataset_fixed.csv`) ถือเป็นการแทรกความรู้เฉพาะทาง (domain knowledge) เพิ่มเติม ไม่ใช่ความสามารถของ OCR เพียว ๆ — ผลดีขึ้นที่เห็นส่วนหนึ่งมาจากฐานข้อมูลภายนอก ไม่ใช่จากอัลกอริทึม OCR/preprocessing เอง
- **ยังไม่มีการทดสอบนัยสำคัญทางสถิติ** (statistical significance) ระหว่างเทคนิคที่คะแนนใกล้เคียงกัน (เช่น deskew+fuzzy vs baseline ต่างกันแค่ ~1.8% CER) จึงยังสรุปแบบเข้มงวดไม่ได้ว่าเทคนิคที่ "ดีที่สุด" ในตารางดีกว่าจริงเชิงสถิติหรือเป็นเพียงความบังเอิญจากการสุ่มของชุดทดสอบ

---

## 9. ไฟล์ที่เกี่ยวข้อง (สำหรับตรวจสอบ/ทำซ้ำการทดลอง)

| ประเภท | Path |
|---|---|
| สร้างภาพ augmented | `augument.py` |
| Utilities ร่วม (metrics, dataset loading, OCR wrapper) | `ocr_core.py` |
| Baseline | `ocr_paddle_only.py` |
| Fuzzy matching | `ocr_paddle_fuzzy.py` |
| Preprocessing เดี่ยว 10 เทคนิค | `preprocess_grayscale.py`, `preprocess_threshold.py`, `preprocess_denoise.py`, `preprocess_sharpen.py`, `preprocess_upscale.py`, `preprocess_deskew.py`, `preprocess_contrast_adaptive.py`, `preprocess_upscale800.py`, `preprocess_tiled_ocr.py`, `preprocess_multitier_fallback.py` |
| เทคนิครวมที่ดีที่สุด | `preprocess_best_plus_fuzzy.py` |
| รวมผลทุกเทคนิคเป็นตารางเดียว | `summarize_results.py` |
| ผลลัพธ์รายภาพ (raw) | `results/<technique>.csv` |
| ผลสรุปค่าเฉลี่ยทุกเทคนิค | `results/summary.csv` |
| ภาพที่ผ่าน preprocess จริงแต่ละเทคนิค (เก็บไว้เทียบด้วยตา) | `preprocessed_images/<technique>/<scenario>/` |
| ฐานข้อมูล INCI สำหรับ fuzzy matching | `ingredient_master_dataset_fixed.csv` |
| โมเดล EDSR super-resolution (ใช้ใน tier 3 ของ multi-tier fallback) | `models/EDSR_x4.pb` |
