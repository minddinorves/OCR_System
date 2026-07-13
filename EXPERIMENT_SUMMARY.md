# สรุปผลการทดลองระบบ OCR สำหรับฉลากส่วนผสมเครื่องสำอาง

เอกสารนี้สรุปการทดลองทั้งหมดที่ทำในโปรเจกต์นี้ ตั้งแต่การเตรียมชุดข้อมูล วิธีวัดผล ไปจนถึงผลลัพธ์ของแต่ละเทคนิค เพื่อใช้อ้างอิงในการเขียนวิทยานิพนธ์

---

## 1. ชุดข้อมูล (Dataset)

| รายการ | จำนวน |
|---|---|
| รูปภาพต้นฉบับ (`test_SkinSafe/image/`) | 137 รูป |
| Label/ground-truth (`test_SkinSafe/label/label*.txt`) | 136 ไฟล์ (ขาด 1 คู่: `A26.jpg` ไม่มี label) |
| ภาพ augmented ต่อ scenario (`augmented_dataset/`) | 137 รูป/scenario × 4 scenario = 548 รูป |
| **รวมภาพที่ใช้ทดสอบต่อเทคนิค OCR** | **685 รูป** (137×5 scenario ลบรูปที่ไม่มี label) |

**4 scenario จำลองสภาพแย่ ๆ ของภาพถ่ายจริง** (สร้างด้วย `augument.py` โดยใช้ไลบรารี albumentations):
- `blur` – ภาพเบลอ (Motion/Gaussian blur)
- `indoor` – แสงในร่ม สีเพี้ยน (RGB shift)
- `low_light` – แสงน้อย
- `night` – แสงน้อยมาก + noise

**Label/ground-truth** คือไฟล์ `.txt` ที่มีรายชื่อสารส่วนผสม (INCI name) ของแต่ละผลิตภัณฑ์ หนึ่งชื่อต่อบรรทัด ได้มาจากการถอดข้อความ (transcribe) จากรูปภาพฉลากด้วยมือ/ผู้ช่วย AI เพื่อใช้เป็น "คำตอบที่ถูกต้อง" — reference text ที่ใช้เทียบกับผล OCR คือการนำรายชื่อทั้งหมดมาต่อกันด้วย `, ` (comma)

---

## 2. เกณฑ์การวัดผล (Evaluation Metrics)

ใช้ 2 ค่ามาตรฐานที่ใช้กันทั่วไปในงานวิจัย OCR:

### CER (Character Error Rate)
```
CER = Levenshtein_distance(ข้อความที่ OCR อ่านได้, ข้อความคำตอบจริง) / จำนวนตัวอักษรของข้อความคำตอบจริง
```
นับจำนวนตัวอักษรที่ผิด (เพิ่ม/ลบ/แก้ไข) หารด้วยความยาวข้อความจริง — **ค่ายิ่งน้อยยิ่งดี** (0 = อ่านถูกทุกตัวอักษร)

### WER (Word Error Rate)
เหมือน CER แต่นับเป็น "คำ" แทน "ตัวอักษร" (ตัด string ด้วย whitespace ก่อนคำนวณ Levenshtein distance)

ทั้งสองค่าคำนวณบนข้อความที่ผ่านการ normalize แล้ว (แปลงเป็นตัวพิมพ์เล็ก + ตัด whitespace ส่วนเกิน) เพื่อไม่ให้case-sensitivity หรือช่องว่างเกินมีผลกับคะแนนเกินจำเป็น

คำนวณด้วยไลบรารี `rapidfuzz.distance.Levenshtein` แยกต่างหากต่อภาพ แล้วหาค่าเฉลี่ย (mean) ต่อ scenario และต่อเทคนิค

**ข้อควรระวังในการตีความ:** ค่า CER/WER ที่ได้ "สูง" (~0.6-0.7) เมื่อเทียบกับงาน OCR ทั่วไปที่มักรายงาน CER < 0.1 เพราะ (ก) reference text เป็นการต่อรายชื่อด้วย comma ซึ่งต่างจาก format จริงบนฉลาก (อาจมีตัดบรรทัด, เครื่องหมายวรรคตอนต่างกัน) (ข) รูปในชุดข้อมูลนี้มีความละเอียดต่ำมาก (สูงสุด-ต่ำสุด: 66-1000px, median 163px) ซึ่งต่ำกว่าภาพถ่ายฉลากผลิตภัณฑ์ทั่วไปมาก ตัวเลขจึงควรใช้เพื่อ**เปรียบเทียบระหว่างเทคนิคในชุดข้อมูลเดียวกัน** มากกว่าเทียบกับงานวิจัยอื่น

---

## 3. เครื่องมือ OCR ที่ใช้

- **PaddleOCR 3.7.0** (โมเดล PP-OCRv6_medium, det + rec) รันบน GPU (NVIDIA RTX 4050, CUDA 12.6, cuDNN 9.9)
- ปิดการใช้ doc orientation classification, doc unwarping, textline orientation (ใช้ค่า default OCR engine ล้วน ๆ เพื่อให้ผลของ "preprocessing" ที่เราทำเองเห็นชัด ไม่ถูกกลบโดยฟีเจอร์ในตัว)

---

## 4. รายการการทดลองทั้งหมด (13 เทคนิค)

| กลุ่ม | สคริปต์ | คำอธิบายเทคนิค |
|---|---|---|
| Baseline | `ocr_paddle_only.py` | OCR ตรง ๆ ไม่มีการแต่งภาพหรือแก้คำใด ๆ |
| Post-processing | `ocr_paddle_fuzzy.py` | OCR ตรง ๆ แล้วเอาผลลัพธ์ไป fuzzy-match กับฐานข้อมูล INCI (`ingredient_master_dataset_fixed.csv`, 11,091 ชื่อ) เพื่อแก้คำที่สะกดผิด/เพี้ยน |
| Preprocessing | `preprocess_grayscale.py` | แปลงภาพเป็น grayscale ก่อน OCR |
| Preprocessing | `preprocess_threshold.py` | Grayscale + Otsu thresholding (แปลงเป็นภาพขาว-ดำ) |
| Preprocessing | `preprocess_denoise.py` | Fast Non-Local-Means Denoising |
| Preprocessing | `preprocess_sharpen.py` | Unsharp mask (เพิ่มความคมชัด) |
| Preprocessing | `preprocess_upscale.py` | Bicubic upscale ให้สูง ≥480px (เฉพาะภาพที่เล็กกว่า) |
| Preprocessing (จาก PDF SkinSafe BOT) | `preprocess_deskew.py` | แก้ความเอียงด้วย Projection Profile Variance Maximization (หมุนทดลอง -5 ถึง +5 องศา) |
| Preprocessing (จาก PDF) | `preprocess_contrast_adaptive.py` | เลือก CLAHE (ภาพสว่าง) หรือ Histogram Equalization (ภาพมืด) ตามค่าความสว่างเฉลี่ย |
| Preprocessing (จาก PDF) | `preprocess_upscale800.py` | Bicubic upscale ให้ด้านสั้นที่สุด ≥800px ตาม spec ของ SkinSafe BOT |
| Preprocessing (จาก PDF) | `preprocess_tiled_ocr.py` | แบ่งภาพเป็นแถบแนวนอนซ้อนทับ (strip 150px, overlap 40px) OCR แยกแถบ แล้วลบข้อความซ้ำด้วย IoU ของกรอบข้อความ |
| Preprocessing (จาก PDF) | `preprocess_multitier_fallback.py` | ไล่ระดับ 4 tier: (1) contrast+bicubic upscale (2) สลับวิธี contrast (3) แทนที่ bicubic ด้วย EDSR super-resolution (4) adaptive Gaussian threshold + invert สี — เลื่อน tier เมื่ออ่านได้ < 20 ตัวอักษร หรือ < 3 กรอบข้อความ |
| **รวมเทคนิคที่ดีที่สุด** | `preprocess_best_plus_fuzzy.py` | Deskew (เทคนิคเดี่ยวที่ดีที่สุด) + fuzzy matching ต่อกัน |

---

## 5. ผลลัพธ์ (เรียงจากดีที่สุด → แย่ที่สุด ตาม mean CER เฉลี่ยข้าม 5 scenario)

| อันดับ | เทคนิค | mean CER | mean WER | เวลาเฉลี่ย/รูป (วินาที) |
|---|---|---|---|---|
| 1 | **Deskew + Fuzzy matching (รวม)** | **0.6078** | 0.7435 | 0.146 |
| 2 | Deskew เดี่ยว | 0.6086 | 0.7323 | 0.151 |
| 3 | Fuzzy matching เดี่ยว | 0.6177 | 0.7549 | 0.145 |
| 4 | **Baseline (ไม่ทำอะไรเลย)** | 0.6191 | 0.7423 | 0.163 |
| 5 | Tiled OCR | 0.6193 | 0.7519 | 0.205 |
| 6 | Grayscale | 0.6218 | 0.7377 | 0.149 |
| 7 | Denoise | 0.6452 | 0.7880 | 0.161 |
| 8 | Contrast adaptive (CLAHE/HistEq) | 0.6484 | 0.7673 | 0.133 |
| 9 | Sharpen | 0.6484 | 0.7636 | 0.164 |
| 10 | Upscale (bicubic →480px) | 0.6764 | 0.7800 | 0.221 |
| 11 | Multi-Tier Fallback (4 tier, รวม EDSR) | 0.6909 | 0.8003 | 0.985 |
| 12 | Upscale800 (bicubic →800px ตาม spec) | 0.7070 | 0.8021 | 0.395 |
| 13 | Threshold (Otsu binarization) | 0.9025 | 0.9414 | 0.058 |

### ผลแยกตาม scenario ของ 3 อันดับแรก

| เทคนิค | original | blur | indoor | low_light | night |
|---|---|---|---|---|---|
| Deskew + Fuzzy | 0.559 | 0.616 | 0.567 | 0.570 | 0.726 |
| Deskew เดี่ยว | 0.560 | 0.618 | 0.568 | 0.570 | 0.727 |
| Baseline | 0.574 | 0.633 | 0.571 | 0.577 | 0.740 |

(ค่าคือ mean CER ของ 137 รูปในแต่ละ scenario)

---

## 6. ข้อสรุปและข้อค้นพบสำคัญ

1. **เทคนิคที่ดีที่สุดคือ "Deskew + Fuzzy matching" รวมกัน** (CER 0.6078) ตามด้วย Deskew เดี่ยว (0.6086) — แต่ส่วนต่างจาก baseline (0.6191) ค่อนข้างน้อย (~1-2% CER) แสดงว่าการปรับปรุงที่ทำได้จริงในชุดข้อมูลนี้มีจำกัด

2. **Deskew เป็นเทคนิค preprocessing ภาพ "เพียงหนึ่งเดียว" ที่ดีกว่า baseline ในทุก scenario** ส่วนเทคนิคอื่นทั้งหมด (grayscale, denoise, sharpen, upscale, contrast adaptive) แย่กว่าหรือเท่า ๆ กับ baseline

3. **ผลลัพธ์ขัดกับสมมติฐาน/design ของระบบ SkinSafe BOT จริงอย่างชัดเจน:**
   - **Bicubic upscaling ทำให้แย่ลง** ทั้งแบบ 480px (CER 0.676) และ 800px ตาม spec จริง (CER 0.707) ทั้งที่ภาพในชุดข้อมูลนี้เล็กมาก (median สูง 163px) ซึ่งควรจะได้ประโยชน์จากการขยายมากที่สุด — เป็นไปได้ว่าโมเดล PP-OCRv6 resize ภาพเข้า pipeline ของตัวเองอยู่แล้ว การ bicubic ขยายภาพล่วงหน้าเพิ่มแค่ blur/interpolation artifact โดยไม่เพิ่มข้อมูลจริง
   - **Multi-Tier Fallback (รวม EDSR Super-Resolution) แย่กว่า baseline ทั้งที่ซับซ้อนและช้าที่สุด** (เฉลี่ย 0.985 วินาที/รูป หรือช้ากว่า baseline ~6 เท่า เพราะเรียก OCR ได้สูงสุด 4 ครั้ง/รูป) ชี้ว่าการพยายามแก้ภาพมากขึ้นไม่ได้แก้ปัญหาความถูกต้องของการอ่าน เพียงแค่เปลี่ยนรูปแบบของความผิดพลาด
   - **Threshold/Binarization (Otsu) แย่ที่สุดมาก** (CER 0.90) เพราะโมเดล PP-OCRv6 ถูกเทรนมาบนภาพสีจริง การตัดเหลือภาพขาว-ดำทำลายข้อมูล texture/gradient ที่โมเดลใช้ตัดสินใจ

4. **Fuzzy matching กับฐานข้อมูล INCI ให้ผลดีขึ้นแบบสม่ำเสมอแต่เล็กน้อย** (0.6191→0.6177 บน baseline, 0.6086→0.6078 บน deskew) เพราะเป็นการแก้ไขข้อความ "หลัง" OCR อ่านเสร็จแล้ว ไม่ได้เปลี่ยนสิ่งที่โมเดลเห็น จึงช่วยแก้ได้แค่คำที่อ่านผิดเล็กน้อย (typo ระดับตัวอักษร) แต่ช่วยไม่ได้ถ้า OCR อ่านพลาดไปเป็นคำอื่นไปเลย

5. **Tiled OCR ใกล้เคียง baseline มาก** (0.6193 vs 0.6191) เพราะภาพส่วนใหญ่ในชุดข้อมูล (median สูง 163px) สั้นกว่าหรือใกล้เคียงกับ strip height ที่ตั้งไว้ (150px) ทำให้ภาพส่วนใหญ่ไม่ได้ถูกแบ่ง tile จริง ๆ — เทคนิคนี้น่าจะให้ผลต่างชัดกว่าถ้าใช้กับภาพฉลากเต็มขวดความสูงมาก ๆ

### ข้อสรุปภาพรวม
สำหรับชุดข้อมูลนี้ (ภาพฉลากส่วนผสมที่ถูก crop มาเป็นแถบเล็ก ความละเอียดต่ำ) **การทำ image preprocessing แบบดั้งเดิม (classical) ส่วนใหญ่ไม่ช่วยหรือกลับทำให้โมเดล deep learning OCR (PaddleOCR) อ่านแย่ลง** มีเพียง **การแก้ความเอียงภาพ (deskew)** และ **การแก้คำหลัง OCR ด้วย fuzzy matching** เท่านั้นที่ให้ผลดีขึ้นอย่างสม่ำเสมอ (แม้เพียงเล็กน้อย) ผลลัพธ์นี้ขัดกับแนวทางการออกแบบของระบบ SkinSafe BOT จริงที่เน้นการทำ preprocessing หนัก (CLAHE, bicubic upscale ถึง 800px, multi-tier fallback พร้อม super-resolution) ซึ่งชี้ให้เห็นว่า **ประสิทธิภาพของ preprocessing pipeline ขึ้นอยู่กับลักษณะภาพอินพุตและโมเดล OCR ที่ใช้เป็นอย่างมาก** ควรมีการทดสอบเชิงประจักษ์ (empirical) แบบนี้ก่อนนำเทคนิคใด ๆ ไปใช้งานจริง ไม่ควรอนุมานจากสมมติฐานทั่วไปเพียงอย่างเดียว

---

## 7. ข้อจำกัดของการทดลอง (สำหรับระบุในวิทยานิพนธ์)

- Reference text (ground truth) เป็นการต่อรายชื่อสารด้วย comma ซึ่งอาจไม่ตรงกับ format ตัวอักษรจริงบนฉลาก (ตัดบรรทัด/เครื่องหมายวรรคตอน) ทำให้ค่า CER/WER สัมบูรณ์ (absolute) มีค่าสูงกว่าความเป็นจริงในระดับหนึ่ง — แต่ใช้ reference เดียวกันทุกเทคนิคจึงเปรียบเทียบกันได้อย่างเป็นธรรม (fair relative comparison)
- เกณฑ์ "อ่านได้น้อยเกินไป" ในขั้น Multi-Tier Fallback (< 20 ตัวอักษร หรือ < 3 กรอบข้อความ) เป็นค่าที่กำหนดขึ้นเอง เนื่องจากเอกสารต้นฉบับ (SkinSafe BOT) ไม่ได้ระบุไว้ชัดเจน
- ขนาดภาพในชุดข้อมูลนี้เล็กผิดปกติ (สูง 66-1000px, median 163px) เมื่อเทียบกับภาพถ่ายฉลากผลิตภัณฑ์ทั่วไป ผลลัพธ์เรื่อง upscaling อาจแตกต่างไปถ้าใช้กับภาพถ่ายความละเอียดสูงกว่านี้
- รูป `A26.jpg` ไม่มีไฟล์ label คู่กัน จึงไม่ถูกนำมาคำนวณ CER/WER ในทุกเทคนิค (ทุกเทคนิคทดสอบบน 685 รูปเท่ากัน)
- Fuzzy matching ใช้ฐานข้อมูล INCI จริงจากภายนอก (`ingredient_master_dataset_fixed.csv`) ถือเป็นการแทรกความรู้เฉพาะทาง (domain knowledge) เพิ่มเติม ไม่ใช่ความสามารถของ OCR เพียว ๆ

---

## 8. การทดลองเพิ่มเติม: เปลี่ยนโมเดล OCR เป็น GOT-OCR2.0

เนื่องจาก classical preprocessing แทบไม่ช่วยอะไร (ดูข้อ 6) จึงทดลองเปลี่ยนตัว OCR engine จาก PaddleOCR (detect+recognize แยกส่วน) เป็น **GOT-OCR2.0** (`stepfun-ai/GOT-OCR-2.0-hf`, ~580M params, end-to-end vision-to-text model) รันบน GPU เดียวกัน เพื่อดูว่าโมเดลที่ออกแบบมาสำหรับข้อความหนาแน่น/เล็กจะรับมือกับภาพความละเอียดต่ำในชุดข้อมูลนี้ได้ดีกว่าไหม (สคริปต์: `ocr_got.py`, ผลดิบ: `results/ocr_got.csv`)

**ผล (685 รูป, หลังแก้ repetition_penalty=1.3 + no_repeat_ngram_size=4 เพื่อกันโมเดล generate ข้อความซ้ำวนไม่จบ):**

| ตัวชี้วัด | GOT-OCR2.0 | Baseline (PaddleOCR) | Deskew+Fuzzy (ดีที่สุดเดิม) |
|---|---|---|---|
| mean CER | 0.639 | 0.619 | **0.608** |
| median CER | **0.593** | 0.740 | ~0.56-0.6 |

**ข้อค้นพบ:** GOT-OCR2.0 มี median CER ดีกว่าชัดเจน (อ่านรูปทั่วไป/ชัดเจนได้แม่นกว่า PaddleOCR จริง) แต่มี **catastrophic failure tail หนัก** — ประมาณ 32% ของรูป (215/680) มี CER สูงกว่า 0.90 (แย่กว่าเทคนิคที่แย่สุดในบรรดา 13 เทคนิคเดิม) โดยเฉพาะรูป scenario `blur`/`night` ที่ภาพแย่มาก โมเดลจะ **hallucinate** คือแต่งประโยคภาษาอังกฤษที่ฟังดูสมเหตุสมผลแต่ไม่เกี่ยวกับภาพเลย (เช่น "of course, he would like to think...") ซึ่งเป็นพฤติกรรมเฉพาะของโมเดล generative/VLM ที่ PaddleOCR (detect+recognize แบบดั้งเดิม) ไม่มี — ผลสุทธิคือ mean CER ใกล้เคียงหรือแย่กว่า baseline เดิมเล็กน้อย ไม่คุ้มที่จะเปลี่ยนมาใช้แทน PaddleOCR+deskew+fuzzy สำหรับ use case นี้ (และช้ากว่ามาก: ~5-6 วินาที/รูป เทียบกับ 0.15 วินาที/รูปของ PaddleOCR)

**ข้อสรุป:** ยังคงใช้ **Deskew + Fuzzy matching บน PaddleOCR** เป็นเทคนิคที่ดีที่สุดของโปรเจกต์นี้ การเปลี่ยนไปใช้โมเดล generative OCR ขนาดใหญ่กว่าไม่ได้แก้ปัญหาความแม่นยำโดยรวม เพียงแค่เปลี่ยนรูปแบบความผิดพลาด (จากอ่านผิด/อ่านไม่ออก เป็นแต่งข้อความมั่ว) แนวทางที่อาจคุ้มค่าต่อยอดในอนาคต (ยังไม่ได้ทำ) คือ hybrid fallback: ใช้ GOT-OCR2.0 เป็นหลักแล้ว fallback ไป PaddleOCR เมื่อ output ดูผิดปกติ

---

## 9. ไฟล์ที่เกี่ยวข้อง

- ผลลัพธ์รายภาพ (raw, ใช้ตรวจสอบ/อ้างอิงเพิ่มเติม): `results/<technique>.csv`
- ผลสรุปค่าเฉลี่ยทุกเทคนิค: `results/summary.csv`
- โค้ดทั้งหมด: ไฟล์ `ocr_*.py` และ `preprocess_*.py` ที่ root ของโปรเจกต์ ใช้ logic ร่วมกันจาก `ocr_core.py`
- ภาพที่ผ่านการ preprocess แต่ละเทคนิคจริง (เก็บไว้เทียบด้วยตา): `preprocessed_images/<technique>/<scenario>/`
- โมเดล EDSR super-resolution ที่ใช้ใน tier 3 ของ multitier fallback: `models/EDSR_x4.pb`
- โค้ดและผลของการทดลอง GOT-OCR2.0 (ข้อ 8): `ocr_got.py`, `results/ocr_got.csv`
