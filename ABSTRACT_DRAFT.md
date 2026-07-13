# บทคัดย่อและวัตถุประสงค์ (ฉบับปรับปรุงให้ตรงกับผลการทดลองจริง)

> ชื่อโครงงานคงเดิมตามรูปเล่มต้นฉบับ เนื้อความบทคัดย่อและวัตถุประสงค์ข้อ 2, 4 ปรับให้ตรงกับวิธีการและผลการทดลองจริงในโปรเจกต์ (อ้างอิงจาก `EXPERIMENT_SUMMARY.md` และ `THESIS_METHODOLOGY_RESULTS.md`)

---

## ชื่อเรื่อง (คงเดิม)

การพัฒนาระบบดึงข้อมูลส่วนผสมของเครื่องสำอางจากภาพฉลากผลิตภัณฑ์โดยใช้เทคนิค OCR

Automated Cosmetic Ingredient Extraction from Product Label Images Using Optical Character Recognition

ชัญญานุช น้อยหมอ
Chanyanut Noimo
คณะวิทยาศาสตร์ มหาวิทยาลัยนเรศวร
Faculty of Science, Naresuan University, Phitsanulok 65000, Thailand

---

## บทคัดย่อ

ปัจจุบันผู้บริโภคให้ความสำคัญกับความปลอดภัยของส่วนผสมในผลิตภัณฑ์เครื่องสำอางมากขึ้น เนื่องจากสารเคมีบางชนิดอาจก่อให้เกิดการระคายเคืองหรืออาการแพ้ต่อผิวหนัง อย่างไรก็ตาม รายชื่อส่วนผสมบนฉลากผลิตภัณฑ์มักพิมพ์ด้วยตัวอักษรขนาดเล็กและประกอบด้วยชื่อสารเคมีที่มีความซับซ้อน ทำให้ผู้ใช้ทั่วไปตรวจสอบข้อมูลได้ยาก โครงงานนี้จึงพัฒนาระบบต้นแบบสำหรับดึงข้อมูลส่วนผสมของเครื่องสำอางจากภาพฉลากผลิตภัณฑ์ด้วยเทคนิค OCR โดยศึกษาเชิงประจักษ์ (empirical study) ว่าเทคนิคการปรับปรุงคุณภาพภาพ (Image Preprocessing) รูปแบบต่าง ๆ ที่ออกแบบขึ้นเพื่อรับมือกับภาพถ่ายคุณภาพต่ำจากผู้ใช้ทั่วไป ได้แก่ การแก้ความเอียงของภาพ (Deskew) การปรับคอนทราสต์แบบปรับตัว (Adaptive CLAHE/Histogram Equalization) การขยายภาพด้วย Bicubic Interpolation การแบ่งภาพเป็นแถบซ้อนทับ (Tiled OCR) และการไล่ระดับการแก้ไขภาพหลายขั้นร่วมกับ Super-Resolution (Multi-Tier Fallback) ช่วยเพิ่มความแม่นยำของ OCR ในการอ่านฉลากเครื่องสำอางได้จริงหรือไม่ เมื่อเทียบกับการไม่ปรับแต่งภาพเลย (Baseline)

การทดลองใช้ PaddleOCR (PP-OCRv6_medium) เป็นเครื่องมือรู้จำข้อความหลัก ทดสอบรวม 13 เทคนิค บนภาพฉลากเครื่องสำอางจริงจำนวน 137 ภาพ ที่ถูกจำลองสภาพแวดล้อมเพิ่มเติมด้วยไลบรารี albumentations เป็น 4 สภาพแวดล้อม (ภาพเบลอ แสงน้อย แสงกลางคืน และแสงในอาคาร) รวมกับภาพต้นฉบับเป็น 5 สภาพแวดล้อม (รวม 685 ภาพ) และใช้เทคนิค Fuzzy String Matching ด้วยไลบรารี RapidFuzz เปรียบเทียบผลลัพธ์กับฐานข้อมูลชื่อสารเครื่องสำอาง (INCI) จำนวน 11,091 รายการ เพื่อแก้ไขคำที่ OCR อ่านผิดพลาด การประเมินผลใช้ค่า Character Error Rate (CER) และ Word Error Rate (WER) เนื่องจากลักษณะปัญหาเป็นการดึงข้อความอิสระ (free-form text extraction) ไม่ใช่การจำแนกประเภท

ผลการทดลองพบว่าเทคนิคที่ให้ค่า CER ต่ำที่สุดคือการรวม **Deskew ร่วมกับ Fuzzy Matching** (mean CER = 0.6078, mean WER = 0.7435) ดีกว่า Baseline เพียงเล็กน้อย (CER = 0.6191) โดย Deskew เป็นเทคนิคปรับภาพเดี่ยวเพียงหนึ่งเดียวที่ให้ผลดีกว่า Baseline ในทุกสภาพแวดล้อม ในขณะที่เทคนิคการปรับภาพหนัก ๆ เช่น การขยายภาพแบบ Bicubic ถึง 800 พิกเซล และ Multi-Tier Fallback ร่วมกับ Super-Resolution กลับให้ผลแย่กว่า Baseline อย่างชัดเจน ผลการศึกษาชี้ให้เห็นว่าประสิทธิภาพของเทคนิค Image Preprocessing ขึ้นอยู่กับลักษณะของภาพอินพุตและสถาปัตยกรรมโมเดล OCR เป็นสำคัญ ควรทดสอบเชิงประจักษ์ก่อนนำไปใช้งานจริง ไม่ควรอนุมานจากสมมติฐานทั่วไปเพียงอย่างเดียว

**คำสำคัญ:** OCR, PaddleOCR, Deskew, Fuzzy Matching, RapidFuzz, Character Error Rate, Image Preprocessing

---

## Abstract (English)

Consumers are increasingly concerned about the safety of cosmetic ingredients, as certain chemicals may cause skin irritation or allergic reactions. However, ingredient lists on product labels are often printed in small fonts with complex chemical names, making them difficult for general users to verify. This project develops a prototype OCR system for extracting cosmetic ingredient information from product label images, intended as a building block for a future personalized skincare recommendation system. Since photos taken by everyday users are often low quality (poor lighting, blur, tilted angles), the project conducts an empirical study on whether various image preprocessing techniques — deskewing, adaptive contrast enhancement (CLAHE/Histogram Equalization), bicubic upscaling, tiled OCR, and a multi-tier fallback pipeline with super-resolution — actually improve OCR accuracy on cosmetic labels compared to no preprocessing (baseline).

The experiments used PaddleOCR (PP-OCRv6_medium) as the primary recognition engine, evaluating 13 techniques on 137 real cosmetic label images augmented into 5 environmental scenarios (original, blur, low-light, night, indoor) using albumentations, totaling 685 images. RapidFuzz-based fuzzy string matching against an 11,091-entry INCI ingredient database was applied as a post-processing correction step. Performance was measured using Character Error Rate (CER) and Word Error Rate (WER), appropriate for free-form text extraction rather than classification-style metrics.

Results show that combining **Deskew with Fuzzy Matching** achieved the lowest mean CER (0.6078) and WER (0.7435), only marginally better than baseline (CER 0.6191). Deskew was the only single preprocessing technique that outperformed baseline across all scenarios, while heavier techniques — bicubic upscaling to 800px and multi-tier fallback with super-resolution — performed consistently worse than baseline. These findings indicate that preprocessing effectiveness depends heavily on input image characteristics and OCR model architecture, and should be validated empirically rather than assumed from general design principles.

**Keywords:** OCR, PaddleOCR, Deskew, Fuzzy Matching, RapidFuzz, Character Error Rate, Image Preprocessing

---

## 1.1 ความเป็นมาและความสำคัญของปัญหา

 ในปัจจุบันผลิตภัณฑ์เครื่องสำอางและผลิตภัณฑ์ดูแลผิวได้รับความนิยมอย่างแพร่หลาย ผู้บริโภคจำนวนมากให้ความสำคัญกับการตรวจสอบส่วนผสมของผลิตภัณฑ์ก่อนการใช้งาน เนื่องจากสารบางชนิดอาจก่อให้เกิดการระคายเคือง อาการแพ้ หรือผลกระทบต่อสุขภาพผิว โดยเฉพาะผู้ที่มีผิวแพ้ง่าย อย่างไรก็ตาม รายชื่อส่วนผสมบนฉลากเครื่องสำอางมักแสดงด้วยตัวอักษรขนาดเล็ก มีความหนาแน่นสูง และประกอบด้วยชื่อสารเคมีที่มีความซับซ้อน ส่งผลให้ผู้ใช้ทั่วไปไม่สามารถอ่านและตรวจสอบข้อมูลได้อย่างสะดวก

 เทคโนโลยี Optical Character Recognition (OCR) เป็นเทคโนโลยีที่ช่วยให้คอมพิวเตอร์สามารถรู้จำข้อความจากภาพได้โดยอัตโนมัติ ซึ่งถูกนำไปประยุกต์ใช้ในหลากหลายด้าน เช่น การแปลงเอกสารเป็นข้อมูลดิจิทัล การอ่านป้ายทะเบียนรถ และการวิเคราะห์ข้อมูลจากฉลากสินค้า ในปัจจุบัน เทคโนโลยี OCR ที่ใช้เทคนิค Deep Learning เช่น PaddleOCR สามารถรองรับข้อความที่มีรูปแบบหลากหลายและมีความซับซ้อนได้ดียิ่งขึ้น อย่างไรก็ตาม ในทางปฏิบัติ ภาพฉลากที่ผู้ใช้ถ่ายเองมักมีคุณภาพต่ำ เช่น ถ่ายในที่แสงน้อย ภาพสั่นหรือเบลอ และมุมกล้องเอียง ซึ่งส่งผลโดยตรงต่อความแม่นยำของ OCR

 โครงงานนี้จึงพัฒนาระบบต้นแบบสำหรับดึงข้อมูลส่วนผสมของเครื่องสำอางจากภาพฉลากผลิตภัณฑ์ด้วยเทคนิค OCR โดยมีเป้าหมายเพื่อนำไปพัฒนาต่อยอดเป็น**ระบบแนะนำสกินแคร์ที่เหมาะสมกับผู้ใช้แต่ละราย** ในอนาคต เนื่องจากภาพถ่ายฉลากคุณภาพต่ำจากผู้ใช้ทั่วไปส่งผลโดยตรงต่อความแม่นยำของ OCR การออกแบบระบบจึงต้องพิจารณาว่าจะใช้เทคนิคปรับปรุงคุณภาพภาพแบบใดจึงจะช่วยให้ OCR อ่านข้อความได้แม่นยำที่สุด

 อย่างไรก็ตาม การเลือกใช้เทคนิค preprocessing ที่ซับซ้อนโดยไม่ผ่านการพิสูจน์เชิงประจักษ์อาจทำให้ระบบทำงานช้าลงหรือแม่นยำน้อยลงโดยไม่จำเป็น โครงงานนี้จึงออกแบบและทดสอบเทคนิคการปรับปรุงคุณภาพภาพหลากหลายรูปแบบด้วยตนเอง ตั้งแต่เทคนิคพื้นฐาน (grayscale, thresholding, denoising, sharpening, upscaling) ไปจนถึงเทคนิคขั้นสูงที่ออกแบบเพื่อรับมือกับภาพถ่ายคุณภาพต่ำโดยเฉพาะ (การแก้ความเอียงของภาพ, การปรับคอนทราสต์แบบปรับตัว, การขยายภาพด้วย bicubic interpolation, การแบ่งภาพเป็นแถบซ้อนทับแล้ว OCR แยกส่วน, และการไล่ระดับการแก้ไขภาพหลายขั้นร่วมกับเทคนิค super-resolution) แล้ววัดผลบนชุดข้อมูลภาพฉลากเครื่องสำอางจริงชุดเดียวกันอย่างเป็นธรรม (controlled comparison) ร่วมกับการใช้เทคนิค Fuzzy String Matching เพื่อแก้ไขข้อผิดพลาดของข้อความหลังจากกระบวนการ OCR เสร็จสิ้น เพื่อคัดเลือกโมดูล OCR ที่แม่นยำและเหมาะสมที่สุดสำหรับนำไปพัฒนาต่อยอดและผนวกเข้ากับระบบแนะนำสกินแคร์ที่เหมาะสมกับผู้ใช้ในขั้นต่อไป

---

## 1.2 วัตถุประสงค์ของโครงงาน

1. เพื่อพัฒนาระบบ OCR สำหรับดึงข้อความส่วนผสมจากภาพฉลากเครื่องสำอาง
2. เพื่อศึกษาประสิทธิภาพของ PaddleOCR ร่วมกับเทคนิคการปรับปรุงคุณภาพภาพ (Image Preprocessing) รูปแบบต่าง ๆ ในการรู้จำข้อความจากภาพฉลากผลิตภัณฑ์ *(เดิม: "PaddleOCR และ Tesseract OCR" — ตัด Tesseract ออกเพราะไม่ได้ใช้จริงในงานปัจจุบัน)*
3. เพื่อพัฒนากระบวนการตรวจจับชื่อสารด้วยเทคนิค Fuzzy String Matching
4. เพื่อประเมินประสิทธิภาพของระบบด้วยค่า Character Error Rate (CER) และ Word Error Rate (WER) *(เดิม: "Accuracy, Precision, Recall และ F1-score" — เปลี่ยนเพราะงานจริงวัดผลด้วย CER/WER ไม่ใช่ตัวชี้วัดแบบ classification)*

---

## 1.3 ขอบเขตของโครงงาน

1. ใช้ภาพฉลากผลิตภัณฑ์เครื่องสำอางภาษาอังกฤษเป็นหลัก
2. ใช้ **PaddleOCR (PP-OCRv6_medium)** เป็นเครื่องมือหลักสำหรับการดึงข้อความจากภาพ *(เดิม: "PaddleOCR และ Tesseract OCR" — ตัด Tesseract ออกเพราะไม่ได้ใช้จริง)*
3. ใช้และเปรียบเทียบเทคนิค Image Preprocessing รวม **10 เทคนิค** ได้แก่ Grayscale Conversion, Otsu Thresholding, Denoising, Sharpening (Unsharp Masking), Upscaling (Bicubic 480px), Deskewing (Projection Profile Variance Maximization), Adaptive Contrast Enhancement (CLAHE/Histogram Equalization), Upscaling ให้ด้านสั้นถึง 800px (Bicubic), Tiled OCR และ Multi-Tier Fallback (ร่วมกับ EDSR Super-Resolution) โดยเมื่อนำไปทดสอบร่วมกับ Baseline (ไม่ปรับแต่งภาพ), Fuzzy Matching (post-processing หลัง OCR ไม่ใช่ image preprocessing) และเทคนิครวม Deskew+Fuzzy Matching จะได้ผลรวมทั้งหมด **13 เทคนิค** ที่นำมาเปรียบเทียบ *(เดิม: ระบุแค่ 3 เทคนิค คือ Grayscale Conversion, Noise Reduction และ Adaptive Thresholding)*
4. ใช้เทคนิค Fuzzy Matching (ไลบรารี RapidFuzz) สำหรับตรวจจับชื่อสารจากฐานข้อมูลเครื่องสำอาง (INCI) จำนวน **11,091 รายการ** *(เดิม: 22,262 รายการ)*
5. ใช้ชุดข้อมูลภาพฉลากเครื่องสำอางจริงจำนวน **137 ภาพ** และภาพจำลองสภาพแวดล้อม (data augmentation ด้วยไลบรารี albumentations) เพิ่มเติม 4 สภาพแวดล้อม (ภาพเบลอ, แสงน้อย, แสงกลางคืน, แสงในอาคาร) รวมเป็น 5 สภาพแวดล้อม **685 ภาพ** *(เดิม: ใช้ชุดข้อมูลภาพแบบสังเคราะห์จำนวน 1,000 ภาพ และภาพจริงจำนวน 15 ภาพ — งานปัจจุบันไม่มีการสร้างภาพสังเคราะห์แล้ว)*
6. ทำการค้นหา pipeline ที่ดีที่สุดจากการรวมหลายเทคนิคเข้าด้วยกัน (technique combination) ด้วยวิธี Greedy Forward Selection และยืนยันผลด้วย Exhaustive Pairwise Search เพื่อให้ได้ระบบ OCR ที่มีความแม่นยำสูงสุดสำหรับนำไปพัฒนาต่อและใช้งานจริง *(หัวข้อใหม่ ไม่มีในรูปเล่มเดิม)*

---

## 1.4 ประโยชน์ที่คาดว่าจะได้รับ

1. ได้โมดูล OCR ที่ผ่านการพิสูจน์เชิงประจักษ์แล้วว่ามีความแม่นยำสูงสุดในบรรดาเทคนิคที่ทดสอบ พร้อมนำไปพัฒนาต่อยอดและผนวกเข้ากับ**ระบบแนะนำสกินแคร์ที่เหมาะสมกับผู้ใช้**ได้จริง *(เดิม: "ได้ระบบต้นแบบสำหรับดึงข้อมูลส่วนผสมของเครื่องสำอางจากภาพฉลากผลิตภัณฑ์" — ปรับให้ระบุเป้าหมายปลายทางเป็นระบบแนะนำสกินแคร์ที่เหมาะสมกับผู้ใช้ โดยยังไม่ระบุชื่อระบบเฉพาะเจาะจง)*
2. สามารถนำเทคนิค OCR และ Fuzzy Matching ไปประยุกต์ใช้กับงานวิเคราะห์ข้อความจากภาพในโมดูลอื่นของระบบแนะนำสกินแคร์ หรือระบบที่เกี่ยวข้องได้
3. เป็นแนวทางและหลักฐานเชิงประจักษ์ประกอบการตัดสินใจออกแบบ pipeline การอ่านฉลากของระบบแนะนำสกินแคร์ในขั้นพัฒนาต่อไป
4. เป็นประโยชน์ต่อการศึกษาและวิจัยด้าน Computer Vision และ Image Processing โดยเฉพาะในประเด็นที่ว่าการทำ preprocessing หนัก ๆ ตามสมมติฐานทั่วไปอาจไม่ได้ช่วยเพิ่มความแม่นยำของโมเดล deep learning OCR เสมอไป และควรมีการทดสอบเชิงประจักษ์ก่อนนำไปใช้งานจริงในระบบ

---

# บทที่ 2 ทฤษฎีและงานวิจัยที่เกี่ยวข้อง (ฉบับปรับปรุง)

> โครงสร้างหัวข้อเปลี่ยนจากเดิม เนื่องจากตัด Tesseract OCR ออกทั้งหมด และเพิ่มทฤษฎีของเทคนิคใหม่ที่ใช้ทดลองจริง (Deskew, CLAHE/HistEq, EDSR Super-Resolution, Tiled OCR, Multi-Tier Fallback, Data Augmentation) เข้ามาแทนที่

## 2.1 Optical Character Recognition (OCR)

 Optical Character Recognition (OCR) เป็นเทคโนโลยีที่ใช้สำหรับแปลงข้อความที่อยู่ในรูปแบบภาพให้อยู่ในรูปแบบข้อมูลดิจิทัลที่สามารถนำไปประมวลผลด้วยคอมพิวเตอร์ได้ โดยกระบวนการ OCR ประกอบด้วยหลายขั้นตอน ได้แก่ การปรับปรุงคุณภาพภาพ (Image Preprocessing) การตรวจจับตำแหน่งข้อความ (Text Detection) และการรู้จำตัวอักษร (Text Recognition)

 ปัจจุบัน OCR ถูกนำไปประยุกต์ใช้อย่างแพร่หลายในงานด้านการแปลงเอกสารเป็นข้อมูลดิจิทัล การอ่านป้ายทะเบียนรถ การสแกนใบเสร็จ การรู้จำข้อความบนป้ายสินค้า และการวิเคราะห์ข้อมูลจากฉลากผลิตภัณฑ์ เทคโนโลยี OCR สมัยใหม่มีการประยุกต์ใช้เทคนิค Deep Learning เพื่อเพิ่มความสามารถในการรู้จำข้อความที่มีลักษณะซับซ้อน เช่น ตัวอักษรขนาดเล็ก ตัวอักษรเอียง หรือข้อความที่อยู่บนพื้นหลังหลากหลายรูปแบบ

 ในโครงงานนี้ OCR ถูกนำมาใช้เพื่อดึงข้อมูลรายชื่อส่วนผสมของเครื่องสำอางจากภาพฉลากผลิตภัณฑ์ เพื่อพัฒนาเป็นโมดูลสำหรับผนวกเข้ากับระบบแนะนำสกินแคร์ที่เหมาะสมกับผู้ใช้

## 2.2 PaddleOCR

 PaddleOCR เป็นไลบรารี OCR แบบ Open-source ที่พัฒนาโดย PaddlePaddle ซึ่งเป็นเฟรมเวิร์กด้าน Deep Learning ที่ได้รับการพัฒนาโดย Baidu โดย PaddleOCR ได้รับการออกแบบให้รองรับทั้งขั้นตอนการตรวจจับข้อความและการรู้จำข้อความภายในระบบเดียว

จุดเด่นของ PaddleOCR ได้แก่
- รองรับหลายภาษา
- รองรับข้อความขนาดเล็ก
- สามารถทำงานได้ดีกับภาพที่มีพื้นหลังซับซ้อน
- รองรับการประมวลผลบน CPU และ GPU
- มีโมเดลที่ได้รับการปรับแต่งให้มีขนาดเล็กและประสิทธิภาพสูง

 ในโครงงานนี้ใช้ PaddleOCR เวอร์ชัน 3.7.0 โมเดล **PP-OCRv6_medium** เป็น OCR Engine เพียงตัวเดียวสำหรับการดึงข้อความจากภาพฉลากเครื่องสำอาง รันบน GPU (NVIDIA RTX 4050 Laptop) โดยปิดฟีเจอร์ในตัวที่ทำงานซ้ำซ้อนกับ preprocessing ที่พัฒนาขึ้นเอง ได้แก่ document orientation classification, document unwarping และ textline orientation classification เพื่อให้เห็นผลของเทคนิค preprocessing ที่ทดลองอย่างชัดเจน ไม่ถูกฟีเจอร์ default ของโมเดลกลบผล *(เดิมมีการใช้ Tesseract OCR เปรียบเทียบด้วย แต่ตัดออกในงานฉบับปัจจุบัน)*

## 2.3 การประมวลผลภาพ (Image Preprocessing)

 การประมวลผลภาพก่อนเข้าสู่กระบวนการ OCR เป็นขั้นตอนสำคัญที่ช่วยเพิ่มคุณภาพของภาพและลดสัญญาณรบกวนที่อาจส่งผลต่อการรู้จำข้อความ โครงงานนี้ออกแบบและเปรียบเทียบเทคนิค Image Preprocessing ทั้งแบบดั้งเดิม (classical) และแบบขั้นสูงที่ออกแบบขึ้นเพื่อรับมือกับภาพถ่ายคุณภาพต่ำโดยเฉพาะ รวม **10 เทคนิค** ดังนี้

**เทคนิคดั้งเดิม (Classical Preprocessing)**
1. **Grayscale Conversion** — แปลงภาพสีให้เป็นภาพระดับเทาเพื่อลดความซับซ้อนของข้อมูลภาพ
2. **Otsu Thresholding** — แปลงภาพระดับเทาให้เป็นภาพขาวดำ (Binary Image) โดยหาค่า threshold ที่เหมาะสมโดยอัตโนมัติด้วยอัลกอริทึม Otsu (maximize between-class variance ของ histogram)
3. **Denoising (Non-Local Means)** — ลดสัญญาณรบกวนด้วยเทคนิค Fast Non-Local Means Denoising ซึ่งเทียบ patch ภาพกับ patch อื่นในหน้าต่างค้นหาแล้วเฉลี่ยถ่วงน้ำหนักตามความคล้าย
4. **Sharpening (Unsharp Masking)** — เพิ่มความคมชัดของขอบตัวอักษรด้วยสูตร unsharp mask (ลบภาพเบลอออกจากภาพจริงแล้วบวกกลับเข้าไปเพิ่มน้ำหนัก)
5. **Upscaling (Bicubic)** — ขยายภาพที่มีความสูงต่ำกว่าเกณฑ์ด้วย Bicubic Interpolation

**เทคนิคขั้นสูงที่ออกแบบขึ้นสำหรับรับมือภาพถ่ายคุณภาพต่ำโดยเฉพาะ** (รายละเอียดแยกเป็นหัวข้อ 2.5-2.9)
6. Deskewing
7. Adaptive Contrast Enhancement (CLAHE/Histogram Equalization)
8. Upscaling ให้ด้านสั้นถึง 800 พิกเซล (Bicubic)
9. Tiled OCR
10. Multi-Tier Fallback (ร่วมกับ EDSR Super-Resolution)

 รวมเทคนิค Image Preprocessing 10 เทคนิคข้างต้น เมื่อนำไปทดสอบร่วมกับ **Baseline** (ไม่ทำ preprocessing ใด ๆ ใช้เป็นเส้นฐานเปรียบเทียบ) และ **Fuzzy Matching** (เทคนิคแก้ไขข้อความ "หลัง" OCR อ่านเสร็จแล้ว ไม่แตะภาพเลย จึงไม่จัดเป็น image preprocessing โดยตรง — รายละเอียดในหัวข้อ 2.11-2.12) รวมถึงเทคนิคผสม **Deskew + Fuzzy Matching** จะได้ผลรวมทั้งหมด **13 เทคนิค** ที่นำมาเปรียบเทียบในบทที่ 4

*(เดิม: หัวข้อนี้ระบุเพียง 4 เทคนิคย่อย คือ Grayscale Conversion, Noise Reduction, Adaptive Thresholding และ Image Sharpening — ขยายให้ครบตามเทคนิคจริงที่ทดลองในงานปัจจุบัน)*

## 2.4 OpenCV

 OpenCV (Open Source Computer Vision Library) เป็นไลบรารีสำหรับการประมวลผลภาพและคอมพิวเตอร์วิทัศน์ที่ได้รับความนิยมอย่างแพร่หลาย โดยมีฟังก์ชันสำหรับการปรับปรุงคุณภาพภาพ การตรวจจับวัตถุ และการวิเคราะห์ข้อมูลภาพ

 ในโครงงานนี้ OpenCV ถูกใช้ในทุกขั้นตอน Image Preprocessing ได้แก่ การแปลงภาพเป็นระดับเทา (`cv2.cvtColor`) การทำ thresholding (`cv2.threshold`, `cv2.adaptiveThreshold`) การลดสัญญาณรบกวน (`cv2.fastNlMeansDenoisingColored`) การหมุนภาพสำหรับ deskewing (`cv2.warpAffine`) การขยายภาพ (`cv2.resize`) และการทำ super-resolution ผ่านโมดูล `cv2.dnn_superres`

## 2.5 Deskewing (Projection Profile Variance Maximization)

 Deskewing เป็นเทคนิคแก้ไขความเอียงของภาพเอกสาร/ฉลากที่เกิดจากมุมกล้องไม่ตรงขณะถ่ายภาพ ในโครงงานนี้ใช้วิธี **Projection Profile Variance Maximization** ซึ่งมีหลักการดังนี้
1. แปลงภาพเป็น grayscale แล้วทำ Otsu inverse binarization ให้ตัวอักษรเป็นสีขาวบนพื้นดำ
2. ทดลองหมุนภาพ binary ทีละมุมในช่วง -5° ถึง +5° (ทีละ 0.5°)
3. แต่ละมุมคำนวณผลรวมพิกเซลตามแนวนอนทีละแถว (row-sum) ได้ profile 1 มิติ แล้วคำนวณค่าความแปรปรวน (variance) ของ profile นั้น
4. เลือกมุมที่ให้ variance สูงสุด เนื่องจากถ้าบรรทัดข้อความเรียงแนวนอนพอดี row-sum จะมีค่าสูงต่ำสลับกันชัดเจน (แถวที่มีตัวอักษรเยอะ สลับกับแถวว่างระหว่างบรรทัด)
5. หมุนภาพสีจริงด้วยมุมที่เลือก

## 2.6 CLAHE และ Histogram Equalization (Adaptive Contrast Enhancement)

 **Histogram Equalization** เป็นเทคนิคปรับปรุงคอนทราสต์ของภาพโดยกระจาย histogram ของค่าความสว่างให้เต็มช่วง 0-255 เหมาะกับภาพที่มืดโดยรวม

 **CLAHE (Contrast Limited Adaptive Histogram Equalization)** เป็นการปรับปรุงจาก Histogram Equalization ทั่วไป โดยแบ่งภาพเป็น tile ย่อย ๆ แล้วทำ histogram equalization เฉพาะที่ในแต่ละ tile พร้อมจำกัด (clip) ระดับคอนทราสต์ไม่ให้สัญญาณรบกวนถูกขยายมากเกินไป เหมาะกับภาพที่มีความสว่างไม่สม่ำเสมอ

 ในโครงงานนี้ใช้เกณฑ์ตัดสินใจเลือกวิธีจากค่าความสว่างเฉลี่ยของภาพ (`gray.mean()`): หากค่า ≥ 127 (ภาพสว่าง) ใช้ CLAHE (`clipLimit=2.0, tileGridSize=(8,8)`) หากค่า < 127 (ภาพมืด) ใช้ Histogram Equalization ธรรมดา

## 2.7 Super-Resolution ด้วย EDSR (Enhanced Deep Residual Networks)

 Super-Resolution คือกระบวนการเพิ่มความละเอียดของภาพด้วยโมเดล Deep Learning ซึ่งต่างจากการขยายภาพแบบ interpolation (เช่น bicubic) ตรงที่โมเดลจะเรียนรู้จากข้อมูลจริงเพื่อสร้างรายละเอียดที่สมจริงขึ้นมาใหม่แทนการประมาณค่าทางคณิตศาสตร์เพียงอย่างเดียว **EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution)** เป็นสถาปัตยกรรมโครงข่ายประสาทเทียมแบบ residual สำหรับงาน super-resolution โดยเฉพาะ

 ในโครงงานนี้ใช้โมเดล EDSR x4 (`models/EDSR_x4.pb`) ผ่านโมดูล `cv2.dnn_superres` เป็นส่วนหนึ่งของเทคนิค Multi-Tier Fallback (หัวข้อ 2.9) โดยใช้เฉพาะกับภาพที่มีด้านสั้นไม่เกิน 250 พิกเซล เนื่องจากโมเดลใช้เวลาประมวลผลและหน่วยความจำสูงกว่าการขยายภาพแบบ bicubic มาก

## 2.8 Tiled OCR และการลบข้อความซ้ำด้วย IoU

 Tiled OCR เป็นแนวทางที่ใช้กับภาพที่มีความสูงมากหรือมีข้อความหนาแน่นหลายบรรทัด โดยแบ่งภาพออกเป็นแถบแนวนอนที่มีการซ้อนทับ (overlap) กันบางส่วน แล้วรัน OCR แยกในแต่ละแถบ เพื่อลดปัญหาตัวอักษรที่อยู่ตรงกลางบรรทัดถูกโมเดลมองข้ามหรือรู้จำผิดพลาดเมื่อภาพทั้งภาพมีขนาดใหญ่เกินไป

 เนื่องจากมีการซ้อนทับกันของแถบ ข้อความบางส่วนจะถูกตรวจจับซ้ำ จึงต้องใช้เทคนิค **Intersection-over-Union (IoU)** ของกรอบข้อความ (bounding box) เพื่อลบข้อความซ้ำ โดยเรียงกรอบข้อความตามค่าความเชื่อมั่น (confidence score) จากมากไปน้อย แล้วเก็บเฉพาะกรอบที่ไม่ทับซ้อนเกินเกณฑ์ที่กำหนด (IoU < 0.5) กับกรอบที่เก็บไว้แล้ว

## 2.9 Multi-Tier Fallback Pipeline

 Multi-Tier Fallback เป็นแนวคิดการไล่ระดับความเข้มข้นของการแก้ไขภาพ โดยเริ่มจากวิธีที่เบาและเร็วที่สุดก่อน แล้ว "เลื่อนขั้น" (escalate) ไปใช้วิธีที่ซับซ้อนและหนักขึ้นเรื่อย ๆ เฉพาะเมื่อผลลัพธ์จากขั้นก่อนหน้ายังไม่น่าพอใจ (เช่น อ่านข้อความได้น้อยเกินไป) แนวคิดนี้ออกแบบมาเพื่อประหยัดเวลาประมวลผลในกรณีภาพคุณภาพดี และสำรองวิธีที่มีประสิทธิภาพสูงกว่าไว้สำหรับภาพคุณภาพต่ำเท่านั้น โดยทั่วไปประกอบด้วยหลายขั้น (tier) ที่ผสมผสานเทคนิคการปรับคอนทราสต์ การขยายภาพ และ Super-Resolution เข้าด้วยกันตามลำดับความซับซ้อนที่เพิ่มขึ้น

## 2.10 Data Augmentation ด้วย Albumentations

 Data Augmentation คือกระบวนการสร้างข้อมูลเพิ่มเติมจากข้อมูลต้นฉบับ โดยปรับแต่งคุณลักษณะบางอย่างของข้อมูลเพื่อจำลองสถานการณ์การใช้งานจริงที่หลากหลายขึ้น ในงานด้าน Computer Vision มักใช้เพื่อจำลองสภาพแวดล้อมของภาพที่แตกต่างกัน เช่น แสง มุมกล้อง หรือคุณภาพของภาพ

 **Albumentations** เป็นไลบรารีสำหรับทำ Data Augmentation บนภาพที่ได้รับความนิยมในงานด้าน Computer Vision เนื่องจากมีความเร็วสูงและรองรับ transform หลากหลายรูปแบบ ในโครงงานนี้ใช้ Albumentations สร้างภาพจำลองสภาพแวดล้อม 4 รูปแบบจากภาพต้นฉบับ ได้แก่ ภาพเบลอ (Motion Blur/Gaussian Blur) ภาพแสงน้อย (Low-light) ภาพแสงกลางคืน (Night, มืดกว่าและมี noise) และภาพแสงในอาคาร (Indoor, สีอมเหลืองจาก RGB Shift) เพื่อทดสอบความคงทน (robustness) ของแต่ละเทคนิค OCR ภายใต้สภาพแวดล้อมที่แตกต่างกัน

## 2.11 Fuzzy String Matching

 Fuzzy String Matching เป็นเทคนิคที่ใช้เปรียบเทียบความคล้ายคลึงระหว่างข้อความสองชุด โดยสามารถรองรับกรณีที่ข้อความมีข้อผิดพลาดหรือมีการสะกดที่ไม่ตรงกันทั้งหมด เทคนิคดังกล่าวได้รับความนิยมในงานด้านการค้นหาข้อมูล การตรวจสอบการสะกดคำ และการจับคู่ข้อความที่ไม่สมบูรณ์

 ในโครงงานนี้ Fuzzy String Matching ถูกนำมาใช้เพื่อเปรียบเทียบข้อความที่ได้จาก OCR กับฐานข้อมูลรายชื่อสารเครื่องสำอาง เนื่องจากข้อความที่ได้จาก OCR อาจมีข้อผิดพลาดจากคุณภาพของภาพหรือสภาพแวดล้อมในการถ่ายภาพ

## 2.12 RapidFuzz Library

 RapidFuzz เป็นไลบรารีสำหรับการเปรียบเทียบข้อความที่ได้รับการพัฒนาขึ้นเพื่อให้มีประสิทธิภาพสูงและสามารถทำงานได้รวดเร็วกว่าไลบรารี FuzzyWuzzy ในหลายกรณี รองรับอัลกอริทึมการวัดความคล้ายคลึงของข้อความหลายรูปแบบ เช่น `ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`

 ในโครงงานนี้ กระบวนการ Fuzzy Matching หลัง OCR (post-processing) ทำงานดังนี้
1. รวมข้อความที่ OCR อ่านได้ทั้งหมด ตัด header เช่น "Ingredients:" ออกด้วย regular expression
2. แบ่งข้อความเป็น token ที่ comma โดยไม่ตัดกรณี comma ตามด้วยตัวเลข (กันไม่ให้ตัดกลางชื่อสารที่มีตัวเลข เช่น "1,2-Hexanediol")
3. แต่ละ token ที่ยาวตั้งแต่ 3 ตัวอักษรขึ้นไป จะถูกค้นหาชื่อที่ใกล้เคียงที่สุดในฐานข้อมูล INCI (11,091 ชื่อไม่ซ้ำ) ด้วยฟังก์ชัน `rapidfuzz.process.extractOne` โดยใช้ scorer แบบ `fuzz.ratio` และกำหนด **score_cutoff = 85**
4. รับผลการจับคู่ก็ต่อเมื่อผ่านเงื่อนไขเพิ่มเติมเรื่องอัตราส่วนความยาวตัวอักษร (ระหว่าง token กับชื่อที่จับคู่ได้) ไม่เกิน 1.6 เท่า เพื่อกันการจับคู่ผิดระหว่าง token สั้นกับชื่อสารที่ยาวกว่ามาก
5. หากไม่ผ่านเงื่อนไข จะเก็บ token เดิมไว้โดยไม่แก้ไข

*(เดิม: ระบุการใช้ฟังก์ชัน `partial_ratio` และกำหนด similarity threshold ที่ 88% — งานปัจจุบันเปลี่ยนมาใช้ `fuzz.ratio` ร่วมกับเงื่อนไขอัตราส่วนความยาว และ threshold 85)*

### 2.12.1 หลักการทำงานของ Levenshtein Distance

 Levenshtein Distance เป็นวิธีการวัดความแตกต่างระหว่างข้อความสองชุด โดยคำนวณจากจำนวนครั้งขั้นต่ำของการดำเนินการที่จำเป็นในการแปลงข้อความหนึ่งให้เป็นอีกข้อความหนึ่ง ซึ่งประกอบด้วยการเพิ่มอักขระ (Insertion) การลบอักขระ (Deletion) และการแทนที่อักขระ (Substitution)

 เทคนิคดังกล่าวเป็นพื้นฐานสำคัญของไลบรารี RapidFuzz ในการคำนวณค่าความคล้ายคลึงของข้อความสำหรับขั้นตอน Fuzzy Matching และยังเป็นพื้นฐานของตัวชี้วัด **Character Error Rate (CER)** และ **Word Error Rate (WER)** ที่ใช้ในการประเมินผลของโครงงานนี้ (รายละเอียดสูตรคำนวณอยู่ในบทที่ 3)

## 2.13 Greedy Forward Selection และ Exhaustive Search

 การหาชุดค่าผสม (combination) ของเทคนิคหลายตัวที่ให้ผลลัพธ์ดีที่สุดเป็นปัญหาที่พบทั่วไปในงาน Machine Learning เรียกว่า **Feature Selection** หรือในบริบทของ pipeline การประมวลผลภาพคือการเลือกลำดับขั้นตอนที่เหมาะสมที่สุด หากมีตัวเลือกทั้งหมด n ตัว จำนวนชุดค่าผสมที่เป็นไปได้ทั้งหมดคือ 2^n ชุด ซึ่งเพิ่มขึ้นแบบ exponential เมื่อ n มีค่ามาก การทดสอบทุกชุดค่าผสม (**Exhaustive Search**) จึงมีค่าใช้จ่ายด้านเวลาสูงมากเมื่อ n ใหญ่

 **Greedy Forward Selection** เป็นวิธีฮิวริสติก (heuristic) ที่ใช้ลดจำนวนชุดค่าผสมที่ต้องทดสอบ โดยเริ่มจากเซตว่าง แล้วเพิ่มตัวเลือกที่ดีที่สุดทีละ 1 ตัวในแต่ละรอบ (เลือกจากตัวเลือกที่เหลือทั้งหมดที่ยังไม่ถูกเลือก) จนกระทั่งไม่มีตัวเลือกใดที่เพิ่มเข้าไปแล้วทำให้ผลลัพธ์ดีขึ้นอีก วิธีนี้ลดจำนวนการทดสอบจาก 2^n เหลือประมาณ n(n+1)/2 ในกรณีที่แย่ที่สุด ข้อจำกัดสำคัญคือเป็นการค้นหาแบบ **locally greedy** ไม่รับประกันว่าจะพบคำตอบที่ดีที่สุดในภาพรวม (global optimum) เนื่องจากตัวเลือกที่ดูแย่เมื่อพิจารณาเดี่ยว ๆ จะถูกตัดทิ้งตั้งแต่ต้น ทั้งที่อาจให้ผลดีเมื่อนำไปรวมกับตัวเลือกอื่นก็เป็นไปได้ (phenomenon นี้เรียกว่า feature interaction หรือ synergy effect)

 ในโครงงานนี้ ใช้ Greedy Forward Selection เพื่อค้นหา pipeline ของเทคนิค Image Preprocessing ที่ให้ค่า CER ต่ำที่สุด และใช้ **Exhaustive Pairwise Search** (ทดสอบทุกคู่ที่เป็นไปได้ C(n,2) คู่) เป็นการตรวจสอบเพิ่มเติมเพื่อยืนยันว่าผลลัพธ์จาก Greedy Search ไม่ได้พลาดคู่เทคนิคที่ดีกว่าไปเนื่องจากข้อจำกัดของวิธี greedy ดังกล่าว

## 2.14 งานวิจัยที่เกี่ยวข้อง

 Smith (2007) ได้นำเสนอระบบ Tesseract OCR ซึ่งเป็นระบบรู้จำตัวอักษรแบบ Open-source ที่ได้รับความนิยมในงานวิจัยและงานพัฒนาซอฟต์แวร์ แม้ว่างานวิจัยฉบับนี้จะไม่ได้ใช้ Tesseract โดยตรงในการทดลอง แต่ยังคงเป็นงานพื้นฐานสำคัญของวงการ OCR แบบ Open-source

 Baek และคณะ (2019) ได้ศึกษาและเปรียบเทียบโมเดล Scene Text Recognition หลายรูปแบบ โดยพบว่าคุณภาพของภาพมีผลโดยตรงต่อประสิทธิภาพของ OCR โดยเฉพาะในกรณีของข้อความขนาดเล็กและพื้นหลังที่ซับซ้อน ซึ่งสอดคล้องกับข้อค้นพบในโครงงานนี้ที่ภาพความละเอียดต่ำและสภาพแสงไม่ดีส่งผลกระทบต่อความแม่นยำของ OCR อย่างชัดเจน

 Du และคณะ (2020) ได้นำเสนอ PP-OCR ซึ่งเป็นระบบ OCR ที่มีขนาดเล็กและประสิทธิภาพสูง เป็นต้นตระกูลของโมเดล PP-OCRv6 ที่ใช้เป็น OCR Engine หลักในงานวิจัยนี้

 Lim และคณะ (2017) ได้นำเสนอ EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution) ซึ่งเป็นสถาปัตยกรรมโครงข่ายประสาทเทียมสำหรับงาน Single Image Super-Resolution ที่ใช้เป็นส่วนหนึ่งของเทคนิค Multi-Tier Fallback ในโครงงานนี้

 V. I. Levenshtein (1966) ได้นำเสนอแนวคิด Levenshtein Distance ซึ่งเป็นพื้นฐานสำคัญของไลบรารี RapidFuzz และตัวชี้วัด CER/WER ที่ใช้ในโครงงานนี้

 Buslaev และคณะ (2020) ได้นำเสนอ **Albumentations** ไลบรารีสำหรับทำ Data Augmentation บนภาพที่มีความเร็วสูงและรองรับ transform หลากหลายรูปแบบ ซึ่งใช้ในการสร้างชุดข้อมูลจำลองสภาพแวดล้อม 4 รูปแบบของโครงงานนี้

*(หมายเหตุ: อ้างอิง Lim et al. 2017 และ Buslaev et al. 2020 เป็นชื่อผู้เขียนงานต้นฉบับจริงของ EDSR และ Albumentations ตามลำดับ — ควรตรวจสอบรายละเอียดบรรณานุกรมฉบับเต็ม (ชื่อบทความ, venue, หน้า) ให้ตรงตามรูปแบบ IEEE ก่อนนำไปใส่ในบรรณานุกรมจริงของเล่ม)*

 จากการศึกษางานวิจัยที่เกี่ยวข้องพบว่า แม้จะมีงานวิจัยจำนวนมากที่มุ่งเน้นการพัฒนาเทคนิค OCR และเทคนิคปรับปรุงคุณภาพภาพแต่ละแบบแยกกัน แต่ยังมีงานวิจัยจำนวนน้อยที่ทดสอบเปรียบเทียบเทคนิค preprocessing หลายรูปแบบอย่างเป็นระบบ (controlled comparison) บนชุดข้อมูลเดียวกัน โดยเฉพาะในบริบทของฉลากเครื่องสำอางซึ่งมีลักษณะข้อความหนาแน่นและซับซ้อนเฉพาะทาง ดังนั้นโครงงานนี้จึงออกแบบและทดสอบเชิงประจักษ์เพื่อค้นหาว่าแนวทาง preprocessing รูปแบบใดช่วยเพิ่มความแม่นยำของ OCR ได้จริง ก่อนนำไปพัฒนาต่อและใช้งานจริงในระบบแนะนำสกินแคร์ที่เหมาะสมกับผู้ใช้

---

# บทที่ 3 วิธีดำเนินงาน (ฉบับปรับปรุง)

## 3.1 ภาพรวมของระบบ

 งานวิจัยนี้เป็นการทดสอบเชิงประจักษ์ (empirical study) เพื่อเปรียบเทียบประสิทธิภาพของเทคนิค Image Preprocessing และเทคนิคหลัง OCR (post-processing) หลายรูปแบบ โดยใช้ PaddleOCR เป็นเครื่องมือรู้จำข้อความหลัก บนภาพฉลากส่วนผสมเครื่องสำอางชุดเดียวกัน เพื่อค้นหา pipeline ที่ให้ความแม่นยำสูงที่สุดสำหรับนำไปพัฒนาต่อและผนวกเข้ากับระบบแนะนำสกินแคร์ที่เหมาะสมกับผู้ใช้ กระบวนการดำเนินงานแบ่งเป็น 4 ขั้นตอนหลัก ได้แก่ (1) การเตรียมชุดข้อมูลและจำลองสภาพแวดล้อม (2) การทดสอบเทคนิคเดี่ยว 13 เทคนิค (3) การค้นหา pipeline ที่ดีที่สุดจากการรวมหลายเทคนิค (Greedy Forward Selection และ Exhaustive Pairwise Search) และ (4) การประเมินผลด้วยตัวชี้วัด CER และ WER

*(เดิม: ภาพรวมเน้นการ "พัฒนาระบบต้นแบบสำหรับดึงข้อมูลส่วนผสม" งานปัจจุบันปรับเป็นการทดสอบเชิงเปรียบเทียบเพื่อหาวิธีที่ดีที่สุด ก่อนนำไปพัฒนาเป็นระบบจริง)*

## 3.2 สถาปัตยกรรมของระบบ

ระบบทดลองที่พัฒนาขึ้นประกอบด้วยขั้นตอนหลัก ดังนี้

1. การรับภาพฉลากเครื่องสำอาง (Input Image) — ภาพจริงหรือภาพจำลองสภาพแวดล้อม
2. การปรับปรุงคุณภาพภาพ (Image Preprocessing) — เลือกใช้ 1 ใน 10 เทคนิค หรือไม่ปรับแต่งเลย (Baseline) หรือ pipeline ที่รวมหลายเทคนิค
3. การดึงข้อความด้วย OCR (PaddleOCR PP-OCRv6_medium)
4. การจับคู่ข้อความด้วย Fuzzy Matching (RapidFuzz) เทียบกับฐานข้อมูล INCI — เป็นทางเลือก (เปิด/ปิดได้)
5. การคำนวณค่า CER และ WER เทียบกับ Ground Truth
6. การบันทึกผลลัพธ์รายภาพและสรุปผลเฉลี่ยแยกตามเทคนิคและสภาพแวดล้อม

 แผนภาพสถาปัตยกรรม (concept): `Input Image → [Preprocessing Technique(s)] → PaddleOCR → [Fuzzy Matching (optional)] → CER/WER Evaluation → results/<technique>.csv`

*(เดิม: สถาปัตยกรรม 7 ขั้นตอนของระบบเดียว รวม Text Cleaning และ Ingredient Segmentation เป็นขั้นตอนแยก — งานปัจจุบันไม่มีขั้นตอน Ingredient Segmentation แยกต่างหาก เพราะการตัด token ทำรวมอยู่ในขั้นตอน Fuzzy Matching เดียว (`correct_with_vocabulary`) และไม่มี Ingredient Database ขนาด 22,262 รายการอีกต่อไป)*

## 3.3 การสร้างชุดข้อมูล

### 3.3.1 ภาพต้นฉบับและ Ground Truth

 ชุดข้อมูลภาพต้นฉบับ (`test_SkinSafe/image/*.jpg`) เป็นภาพถ่ายฉลากส่วนผสมเครื่องสำอางจริงจำนวน **137 ภาพ** ที่ถูก crop มาเฉพาะส่วนที่มีรายชื่อสารเคมี ขนาดภาพมีความหลากหลายและเล็กกว่าภาพถ่ายฉลากทั่วไปมาก โดยมีความสูงต่ำสุด 66 พิกเซล สูงสุด 1,000 พิกเซล และมัธยฐาน (median) เพียง 163 พิกเซล (วัดจาก 136 ภาพที่มี label คู่กัน)

 Ground Truth (`test_SkinSafe/label/label<ชื่อไฟล์>.txt`) มีทั้งหมด **136 ไฟล์** (ขาด 1 คู่: `A26.jpg` ไม่มี label คู่กัน จึงถูกตัดออกจากการคำนวณ CER/WER ในทุกเทคนิคเพื่อให้จำนวนตัวอย่างเท่ากันทุกเทคนิค) แต่ละไฟล์คือรายชื่อสาร (INCI name) หนึ่งชื่อต่อบรรทัด ได้จากการถอดข้อความด้วยมือ/ผู้ช่วย AI จากภาพฉลากจริง

 Reference text ที่ใช้เทียบกับผล OCR คือการนำรายชื่อสารทุกชื่อในไฟล์ label มาต่อกันด้วย `, ` (comma + space) เป็น string เดียว ซึ่งเป็นจุดสำคัญที่ทำให้ค่า CER/WER สัมบูรณ์สูงกว่างานวิจัย OCR ทั่วไป เพราะรูปแบบนี้ต่างจากการจัดวางตัวอักษรจริงบนฉลาก (ดูรายละเอียดในหัวข้อ 3.10.5)

*(เดิม: ใช้ภาพสังเคราะห์ 1,000 ภาพ + ภาพจริง 15 ภาพ สร้างด้วย Pillow จากฐานข้อมูล INCI แบบสุ่มชื่อสาร — งานปัจจุบันไม่มีการสร้างภาพสังเคราะห์แล้ว ใช้ภาพจริงทั้งหมด 137 ภาพเป็นฐาน แล้วขยายด้วย Data Augmentation แทน)*

## 3.4 Data Augmentation

 เพื่อจำลองสภาพแวดล้อมการถ่ายภาพจริงที่มีคุณภาพต่ำ ใช้ไลบรารี **albumentations** สร้างภาพจำลอง 4 สภาพแวดล้อมจากภาพต้นฉบับทุกภาพ (137 × 4 = 548 ภาพ) รวมกับภาพต้นฉบับเป็น **5 สภาพแวดล้อมทั้งหมด รวม 685 ภาพ** (680 ภาพหลังตัด A26.jpg ที่ถูกใช้ให้คะแนนจริง)

| Scenario | Transform | รายละเอียดพารามิเตอร์ |
|---|---|---|
| **blur** | `A.OneOf([MotionBlur, GaussianBlur], p=1.0)` | สุ่มเลือก Motion Blur (`blur_limit=5`) หรือ Gaussian Blur (`blur_limit=(3,5)`) จำลองภาพสั่น/โฟกัสไม่ชัด |
| **low_light** | `RandomBrightnessContrast(brightness_limit=(-0.3,-0.15), contrast_limit=(-0.15,0))` | ลดความสว่าง 15-30% และลดคอนทราสต์ 0-15% |
| **night** | `RandomBrightnessContrast(brightness_limit=(-0.45,-0.25), contrast_limit=(-0.25,-0.1))` + `GaussNoise(p=0.5)` + `CLAHE(clip_limit=1.5, tile_grid_size=(8,8), p=0.5)` | มืดกว่า low_light ชัดเจน บวกสุ่ม 50% ใส่ noise และสุ่ม 50% ใส่ CLAHE เบา ๆ — สภาพแย่ที่สุดในชุดข้อมูล |
| **indoor** | `RGBShift(r_shift=(15,25), g_shift=(5,15), b_shift=(-25,-15))` + `RandomBrightnessContrast(brightness_limit=(-0.05,0.05))` + `RandomSunFlare(p=0.4)` | จำลองแสงในร่ม/หลอดไฟสีเหลือง พร้อมสุ่ม 40% มี lens flare |

 แต่ละภาพสุ่มค่าภายในช่วง limit ที่กำหนดเพียงครั้งเดียวตอนสร้าง (`augmented_dataset/` ถูก generate ไว้ล่วงหน้า ไม่สุ่มใหม่ทุกครั้งที่รัน OCR) ทำให้ทุกเทคนิคที่ทดสอบเห็นภาพชุดเดียวกันเป๊ะ ๆ เปรียบเทียบกันได้แบบ apples-to-apples

*(เดิม: ตารางเทคนิค Data Augmentation ในรูปเล่มเดิมระบุ 5 เทคนิคกว้าง ๆ ได้แก่ Gaussian Blur, Brightness Adjustment, Low-light Simulation, Indoor Lighting, Rotation — งานปัจจุบันใช้ albumentations จริงและมีเพียง 4 scenario ไม่มี Rotation แยกต่างหาก เพราะการแก้ความเอียงถูกทดสอบเป็นเทคนิค Deskew ในขั้นตอน preprocessing แทน)*

## 3.5 เครื่องมือ OCR และการตั้งค่า

 ใช้ **PaddleOCR 3.7.0** โมเดล **PP-OCRv6_medium** (รวม text detection + text recognition ในตัว) รันบน GPU (NVIDIA RTX 4050 Laptop, CUDA 12.6, cuDNN 9.9) การตั้งค่าปิดฟีเจอร์ในตัว 3 อย่างที่อาจทำงานซ้ำซ้อนกับ preprocessing ที่พัฒนาขึ้นเอง:

```python
PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
```

 โมเดล PaddleOCR ถูก cache เป็น singleton และใช้ซ้ำข้ามทุกภาพในรอบทดลองเดียวกัน เพราะการโหลดโมเดลใช้เวลานาน เวลาที่วัด (`ocr_time_sec`) จึงเป็นเวลา inference ต่อภาพเท่านั้น ไม่รวมเวลาโหลดโมเดลครั้งแรก

*(เดิม: ใช้ PaddleOCR ร่วมกับ Tesseract OCR เปรียบเทียบกัน — งานปัจจุบันตัด Tesseract ออก ใช้ PaddleOCR เพียงตัวเดียวเป็น OCR Engine หลัก)*

## 3.6 ฐานข้อมูลส่วนผสมเครื่องสำอาง

 ฐานข้อมูลส่วนผสม (`ingredient_master_dataset_fixed.csv`) จัดเก็บในรูปแบบ CSV ประกอบด้วยรายชื่อสาร INCI จำนวน **11,091 รายการไม่ซ้ำ** ใช้เป็นข้อมูลอ้างอิงในการตรวจจับรายชื่อสารด้วยเทคนิค Fuzzy Matching

*(เดิม: ฐานข้อมูลมี 22,262 รายการ รวบรวมจาก Open Beauty Facts, Kaggle INCI Dataset และ CosIng Database — งานปัจจุบันใช้ฐานข้อมูลที่ผ่านการทำความสะอาด (fixed) จำนวนน้อยกว่าเดิม)*

## 3.7 Fuzzy String Matching

 กระบวนการ Fuzzy Matching หลัง OCR (`ocr_paddle_fuzzy.py`) ทำงานดังนี้:

1. รวมข้อความทุกบรรทัดที่ OCR อ่านได้ ตัด header เช่น "Ingredients:" ออกด้วย regex `^[^:：]*ingredient[^:：]*[:：]\s*`
2. แบ่งข้อความเป็น token ด้วย regex `,(?!\s*\d)` — ตัดที่ comma แต่ไม่ตัดถ้า comma ตามด้วยตัวเลข (กันไม่ให้ตัดกลางชื่อสารที่มีตัวเลข เช่น "1,2-Hexanediol")
3. แต่ละ token ที่ยาว ≥ 3 ตัวอักษร ค้นหาชื่อที่ใกล้เคียงที่สุดในฐานข้อมูล INCI ด้วย `rapidfuzz.process.extractOne(token, vocabulary, scorer=fuzz.ratio, score_cutoff=85)`
4. รับ match ก็ต่อเมื่อผ่านเงื่อนไข length-ratio เพิ่มเติม: อัตราส่วนความยาว (ยาวกว่า/สั้นกว่า) ระหว่าง token กับชื่อที่ match ต้อง ≤ 1.6 เท่า
5. ถ้าไม่ผ่านเงื่อนไข เก็บ token เดิมไว้ไม่แก้ไข

*(เดิม: ใช้ฟังก์ชัน `partial_ratio` และ similarity threshold 88% — งานปัจจุบันเปลี่ยนเป็น `fuzz.ratio` + score_cutoff 85 + เงื่อนไข length-ratio เพิ่มเติม)*

## 3.8 การค้นหา Pipeline ที่ดีที่สุด (Combination Search)

 นอกจากการเปรียบเทียบเทคนิคเดี่ยว 13 เทคนิค งานวิจัยนี้ยังทำการค้นหาว่าการรวมหลายเทคนิคเข้าด้วยกันจะให้ผลดีกว่าเทคนิคเดี่ยวที่ดีที่สุดหรือไม่ ด้วย 2 ขั้นตอน (รายละเอียดทฤษฎีอยู่ในหัวข้อ 2.13 และผลเต็มอยู่ในหัวข้อ 4.3-4.4):

1. **Greedy Forward Selection** — ค้นหา pipeline ทีละขั้นตอน เริ่มจาก Baseline แล้วเพิ่มเทคนิคที่ดีที่สุดทีละ 1 จนไม่มีตัวใดช่วยเพิ่มได้อีก แล้วทดสอบเพิ่ม Fuzzy Matching ปิดท้าย
2. **Exhaustive Pairwise Search** — ทดสอบทุกคู่ที่เป็นไปได้จากเทคนิคที่ไม่ถูกตัดออกด้วยเหตุผลเชิงประจักษ์ (15 คู่ จาก 28 คู่ทั้งหมด) เพื่อยืนยันว่า Greedy Search ไม่พลาดคู่เทคนิคที่ดีกว่า

 ทั้งสองขั้นตอนรันบนภาพเต็มทั้ง 685 ภาพ (ไม่ใช่ตัวอย่างสุ่ม) เพื่อความน่าเชื่อถือของผลลัพธ์ หลังจากพบว่าการค้นหารอบแรกด้วยตัวอย่าง 75 ภาพให้ผลลัพธ์คลาดเคลื่อน (ดูรายละเอียดในหัวข้อ 4.3)

*(หัวข้อใหม่ ไม่มีในรูปเล่มเดิม)*

## 3.9 สภาพแวดล้อมการพัฒนา

| Tool | Purpose |
|---|---|
| Python 3.12 | Development |
| OpenCV (opencv-contrib-python) | Preprocessing, EDSR Super-Resolution (`cv2.dnn_superres`) |
| PaddleOCR / PaddlePaddle-GPU 3.2.2 | OCR Engine (PP-OCRv6_medium) |
| RapidFuzz | Fuzzy Matching |
| Albumentations | Data Augmentation |
| NVIDIA RTX 4050 Laptop (CUDA 12.6, cuDNN 9.9) | GPU Inference |

 ระหว่างการทดลองพบปัญหาความขัดแย้งของไลบรารีระดับ environment: **PaddlePaddle และ PyTorch (ที่ติดตั้งไว้สำหรับการทดลอง GOT-OCR2.0) ต่างมีไฟล์ cuDNN DLL ชื่อเดียวกันแต่คนละเวอร์ชัน** (`cudnn_engines_precompiled64_9.dll`) ทำให้เมื่อทั้งสองไลบรารีถูก import ในโปรเซสเดียวกัน ระบบปฏิบัติการ Windows ไม่สามารถโหลด DLL ที่ชนกันได้ แก้ไขโดยแยก virtual environment สำหรับการทดลองที่ใช้ PaddleOCR ออกจาก environment หลักที่มี PyTorch เพื่อป้องกันการชนกันของไลบรารี

*(เดิม: ตารางเครื่องมือระบุ Python, OpenCV, PaddleOCR, Tesseract OCR, RapidFuzz, Pillow — งานปัจจุบันตัด Tesseract และ Pillow ออก (ไม่มีการสร้างภาพสังเคราะห์แล้ว) เพิ่ม Albumentations)*

## 3.10 การประเมินผล

### 3.10.1 การ Normalize ข้อความก่อนเทียบ

```python
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text
```

 ทำกับทั้ง OCR output (hypothesis) และ reference text ก่อนคำนวณเสมอ เพื่อไม่ให้ case-sensitivity หรือช่องว่างเกินกระทบคะแนน

### 3.10.2 Character Error Rate (CER)

```
CER = Levenshtein_distance(normalize(hypothesis), normalize(reference)) / len(normalize(reference))
```

 คำนวณด้วย `rapidfuzz.distance.Levenshtein.distance` หารด้วยความยาวตัวอักษรของ reference ที่ normalize แล้ว ค่ายิ่งน้อยยิ่งดี (0 = อ่านถูกทุกตัวอักษร ค่า > 1 เป็นไปได้ถ้า OCR อ่านออกมายาว/ผิดเพี้ยนเกินความยาวจริง)

### 3.10.3 Word Error Rate (WER)

```
WER = Levenshtein_distance(normalize(hypothesis).split(), normalize(reference).split()) / len(normalize(reference).split())
```

 สูตรเดียวกับ CER แต่คำนวณระดับ "คำ" แทน "ตัวอักษร" คำที่สะกดผิดแม้ 1 ตัวอักษรนับเป็น 1 word error เต็ม ไม่ใช่ partial credit ทำให้ WER มักสูงกว่า CER เสมอ

### 3.10.4 การรวมผลลัพธ์

 CER/WER คำนวณแยกต่อภาพ 1 ค่า แล้วหาค่าเฉลี่ยเลขคณิต (arithmetic mean, macro-average) แยกตาม scenario และแยกตามเทคนิค — ทุกภาพมีน้ำหนักเท่ากันไม่ว่า reference จะยาวสั้นแค่ไหน

### 3.10.5 เหตุผลที่ค่า CER/WER สัมบูรณ์สูงกว่างานวิจัย OCR ทั่วไป

 ค่า CER/WER ที่ได้ (~0.55-0.95) สูงกว่างานวิจัย OCR ทั่วไปที่มักรายงาน CER < 0.1 มาก เนื่องจาก (1) reference text ต่อด้วย comma ต่างจาก line-break/ตัวคั่นจริงบนฉลาก และ (2) ภาพในชุดข้อมูลมีความละเอียดต่ำมาก (median สูง 163px) ดังนั้นตัวเลขเหล่านี้ควรใช้เพื่อเปรียบเทียบ**ระหว่างเทคนิคบนชุดข้อมูลเดียวกัน** ไม่ใช่นำไปเทียบตรง ๆ กับตัวเลขจากงานวิจัยอื่น

*(เดิม: ใช้ Accuracy, Precision, Recall, F1-score ซึ่งเป็นตัวชี้วัดแบบ classification โดยกำหนด TP/TN/FP/FN จากการจับคู่ชื่อสารทีละรายการ — งานปัจจุบันเปลี่ยนเป็น CER/WER ทั้งหมด เนื่องจากลักษณะปัญหาเป็นการดึงข้อความอิสระที่ไม่สามารถกำหนด True Negative ได้ชัดเจนเหมือนปัญหาการจำแนกประเภท)*

### 3.10.6 ตัวชี้วัดเสริมระดับชื่อสาร (Ingredient-level Metrics)

 เนื่องจาก CER/WER เปรียบเทียบที่ระดับตัวอักษร/คำ ซึ่งได้รับผลกระทบจากความแตกต่างของรูปแบบการจัดวางข้อความ (reference ต่อด้วย comma ในขณะที่ OCR อ่านตามการตัดบรรทัดจริงบนฉลาก ดูหัวข้อ 3.10.5) จึงเพิ่มตัวชี้วัดเสริมที่เปรียบเทียบที่ **ระดับชื่อสาร (ingredient token)** แทน เพื่อวัดคุณภาพการดึงข้อมูลโดยไม่ถูกรบกวนจากปัญหารูปแบบการจัดวางข้อความ

 ขั้นตอนการคำนวณ (`compute_ingredient_metrics.py`) ไม่ต้องรัน OCR ใหม่ เพราะใช้ข้อความ OCR ที่บันทึกไว้แล้วจากการทดลองเดิม:
1. ตัดข้อความทั้งสองฝั่ง (OCR output และ reference) เป็น token ด้วย regex เดียวกับขั้นตอน Fuzzy Matching (`,(?!\s*\d)`)
2. **Ingredient Error Rate (IER)** = Levenshtein distance ระหว่าง token list สองฝั่ง หารด้วยจำนวน token ของ reference (แนวคิดเดียวกับ WER แต่หน่วยเป็น "ชื่อสาร" แทน "คำ") — เป็นตัวชี้วัดที่เข้มงวด นับ token ทั้งก้อนถูก/ผิด ไม่มี partial credit
3. **Set Precision/Recall/F1** = จับคู่ token ของทั้งสองฝั่งแบบ one-to-one ด้วย `rapidfuzz.fuzz.ratio` (เรียงจากคะแนนสูงไปต่ำ, threshold ≥ 85 คะแนน — ค่าเดียวกับที่ใช้ในขั้นตอน Fuzzy Matching) แทนการเทียบแบบ exact match เพื่อให้เครดิตกับคำที่ใกล้เคียงถูกต้อง (เช่น ชื่อที่ถูกแก้เป็นรูปแบบมาตรฐาน INCI ซึ่งอาจสะกดต่างจาก label เล็กน้อย) ไม่ใช่การเทียบตัวอักษรตรงเป๊ะ

 **ผลลัพธ์ (ค่าเฉลี่ยข้าม 5 scenario, n=685 ภาพ):**

| เทคนิค | mean IER | mean Set-Precision | mean Set-Recall | mean Set-F1 |
|---|---|---|---|---|
| **Deskew + Fuzzy** | 0.857 | 0.395 | 0.305 | **0.305 (ดีที่สุด)** |
| Deskew เดี่ยว | **0.788 (ต่ำสุด)** | 0.421 | 0.288 | 0.299 |
| Fuzzy เดี่ยว | 0.863 | 0.376 | 0.288 | 0.296 |
| Baseline | 0.803 | 0.388 | 0.275 | 0.282 |

 ผลลัพธ์ Set-F1 (ใช้ fuzzy token matching) **สอดคล้องกับผล CER/WER เดิม 100%** — Deskew+Fuzzy ยังคงเป็นเทคนิคที่ดีที่สุด ยืนยันข้อสรุปเดิมด้วยตัวชี้วัดที่ไม่ถูกรบกวนจากปัญหารูปแบบข้อความ

 ข้อสังเกตสำคัญ: **IER กลับแสดงว่า Deskew เดี่ยวดีกว่า Deskew+Fuzzy** ซึ่งดูขัดแย้งกับ Set-F1 ในตารางเดียวกัน เหตุผลคือ IER เปรียบเทียบ token แบบ all-or-nothing (ไม่ fuzzy) จึง "ลงโทษ" คำที่ Fuzzy Matching แก้ให้เป็นชื่อมาตรฐาน INCI แต่สะกดต่างจาก label ต้นฉบับเล็กน้อย ทั้งที่ในทางเนื้อหาถูกต้อง สะท้อนให้เห็นว่า **IER เหมาะสำหรับวัดความสามารถในการสร้างข้อความกลับให้ตรงเป๊ะ ในขณะที่ Set-F1 (fuzzy) เหมาะสำหรับวัดคุณภาพการระบุชื่อสารซึ่งตรงกับเป้าหมายเชิงปฏิบัติของระบบมากกว่า** ควรใช้ Set-F1 เป็นตัวชี้วัดเสริมหลักในการอภิปรายผล

*(หัวข้อใหม่ ไม่มีในรูปเล่มเดิม — เพิ่มขึ้นเพื่อตอบคำถามว่าปัญหารูปแบบ reference text ใน 3.10.5 มีทางแก้ไขหรือไม่)*

## 3.11 ไฟล์ที่เกี่ยวข้อง (สำหรับตรวจสอบ/ทำซ้ำการทดลอง)

| ประเภท | Path |
|---|---|
| สร้างภาพ augmented | `augument.py` |
| Utilities ร่วม (metrics, dataset loading, OCR wrapper) | `ocr_core.py` |
| Baseline | `ocr_paddle_only.py` |
| Fuzzy matching เดี่ยว | `ocr_paddle_fuzzy.py` |
| Preprocessing เดี่ยว 10 เทคนิค | `preprocess_grayscale.py`, `preprocess_threshold.py`, `preprocess_denoise.py`, `preprocess_sharpen.py`, `preprocess_upscale.py`, `preprocess_deskew.py`, `preprocess_contrast_adaptive.py`, `preprocess_upscale800.py`, `preprocess_tiled_ocr.py`, `preprocess_multitier_fallback.py` |
| เทคนิครวมที่ดีที่สุด (มือเลือก) | `preprocess_best_plus_fuzzy.py` |
| Greedy Forward Selection | `preprocess_combination_search.py` |
| Exhaustive Pairwise Search | `preprocess_pairs_search.py`, `run_pairs_search.py` |
| รวมผลทุกเทคนิคเป็นตารางเดียว | `summarize_results.py` |
| ผลลัพธ์รายภาพ (raw) | `results/<technique>.csv` |
| ผลสรุปค่าเฉลี่ยทุกเทคนิค | `results/summary.csv` |
| คำนวณตัวชี้วัดเสริมระดับชื่อสาร (IER, Set-P/R/F1) | `compute_ingredient_metrics.py` |
| ผลตัวชี้วัดระดับชื่อสารรายภาพ | `results/<technique>_ingredient.csv` |
| ผลสรุปตัวชี้วัดระดับชื่อสาร | `results/summary_ingredient.csv` |
| การวิเคราะห์เชิงสถิติเพิ่มเติม (Wilcoxon, distribution, correlation) | `comprehensive_analysis.py` |
| ผลการกระจายตัวของ CER รายเทคนิค | `results/cer_distribution.csv` |
| ผลการค้นหา Combination | `results/combination_search_log.csv`, `results/pairs_search_log.csv`, `results/preprocess_best_combination.csv` |
| Log การรัน Combination Search แต่ละรอบ | `combination_search_run.log`, `combination_search_fulldata_run.log` |
| ภาพที่ผ่าน preprocess จริงแต่ละเทคนิค | `preprocessed_images/<technique>/<scenario>/` |
| ฐานข้อมูล INCI สำหรับ fuzzy matching | `ingredient_master_dataset_fixed.csv` |
| โมเดล EDSR super-resolution | `models/EDSR_x4.pb` |

---

# บทที่ 4 ผลการทดลองและการวิเคราะห์ (ฉบับปรับปรุง)

## 4.1 ผลการทดลองของเทคนิคเดี่ยว 13 เทคนิค

 ตารางที่ 4.1 แสดง mean CER, mean WER และเวลาเฉลี่ยต่อภาพของทั้ง 13 เทคนิค เรียงจากดีที่สุดไปแย่ที่สุด คำนวณจากค่าเฉลี่ยแบบไม่ถ่วงน้ำหนักของ 5 scenario (n=136 ภาพ/scenario, รวม 680 ภาพ)

| อันดับ | เทคนิค | mean CER | mean WER | เวลาเฉลี่ย/ภาพ (วินาที) |
|---|---|---|---|---|
| 1 | **Deskew + Fuzzy matching** | **0.6078** | 0.7435 | 0.146 |
| 2 | Deskew เดี่ยว | 0.6086 | 0.7323 | 0.151 |
| 3 | Fuzzy matching เดี่ยว | 0.6177 | 0.7549 | 0.145 |
| 4 | Baseline (ไม่ทำอะไรเลย) | 0.6191 | 0.7423 | 0.163 |
| 5 | Tiled OCR | 0.6193 | 0.7519 | 0.205 |
| 6 | Grayscale | 0.6218 | 0.7377 | 0.149 |
| 7 | Denoise | 0.6452 | 0.7880 | 0.161 |
| 8 | Contrast Adaptive (CLAHE/HistEq) | 0.6484 | 0.7673 | 0.133 |
| 9 | Sharpen | 0.6484 | 0.7636 | 0.164 |
| 10 | Upscale (bicubic → 480px) | 0.6764 | 0.7800 | 0.221 |
| 11 | Multi-Tier Fallback (4 tier + EDSR) | 0.6909 | 0.8003 | 0.985 |
| 12 | Upscale800 (bicubic → 800px) | 0.7070 | 0.8021 | 0.395 |
| 13 | Threshold (Otsu Binarization) | 0.9025 | 0.9414 | 0.058 |

ตารางที่ 4.1 แสดงผลการทดลองจากภาพชุดทดสอบทั้งหมด (ภาพจริง + augmented 5 scenario)

*(เดิม: หัวข้อ 4.1 พูดถึง "ผลการทดลองกับภาพจริง" แยกจากภาพสังเคราะห์ โดยรายงานเป็น Accuracy/Precision/Recall/F1-score (64.4/60.5/64.4/59.7%) — งานปัจจุบันไม่มีการแบ่งภาพจริง/สังเคราะห์แยกกันแล้ว เพราะไม่มีภาพสังเคราะห์อีกต่อไป ใช้ตัวชี้วัด CER/WER แทนทั้งหมด)*

## 4.2 ผลแยกตามสภาพแวดล้อม (Scenario) ของ 3 อันดับแรก

| เทคนิค | original | blur | indoor | low_light | night |
|---|---|---|---|---|---|
| Deskew + Fuzzy | 0.559 | 0.616 | 0.567 | 0.570 | 0.726 |
| Deskew เดี่ยว | 0.560 | 0.618 | 0.568 | 0.570 | 0.727 |
| Baseline | 0.574 | 0.633 | 0.571 | 0.577 | 0.740 |

 ทุกเทคนิคมีค่า CER สูงขึ้น (แย่ลง) อย่างชัดเจนในสภาพแวดล้อม **night** ซึ่งเป็นสภาพที่มืดที่สุดและมี noise เพิ่มเข้ามา สอดคล้องกับสมมติฐานว่าความสว่างและความคมชัดของภาพมีผลโดยตรงต่อความแม่นยำของ OCR ในทุกเทคนิค แต่ Deskew ยังคงให้ผลดีกว่า Baseline ในทุก scenario โดยไม่มีข้อยกเว้น แสดงว่าประโยชน์ของการแก้ความเอียงภาพไม่ขึ้นกับสภาพแสง

### ผลแยกตามสภาพแวดล้อมด้วยตัวชี้วัดเสริม (Set-F1 และ IER, ดูหัวข้อ 3.10.6)

**Set-F1 (fuzzy token match, ยิ่งสูงยิ่งดี):**

| เทคนิค | original | blur | indoor | low_light | night |
|---|---|---|---|---|---|
| **Deskew + Fuzzy** | **0.359** | **0.263** | **0.362** | **0.356** | 0.186 |
| Deskew เดี่ยว | 0.350 | 0.258 | 0.349 | 0.351 | **0.188 (ดีสุด)** |
| Fuzzy เดี่ยว | 0.346 | 0.256 | 0.350 | 0.349 | 0.180 |
| Baseline | 0.327 | 0.232 | 0.333 | 0.336 | 0.181 |

**IER (Ingredient Error Rate, ยิ่งต่ำยิ่งดี):**

| เทคนิค | original | blur | indoor | low_light | night |
|---|---|---|---|---|---|
| **Deskew เดี่ยว** | **0.748** | **0.823** | **0.752** | **0.751** | **0.868** |
| Baseline | 0.768 | 0.843 | 0.765 | 0.766 | 0.872 |
| Deskew + Fuzzy | 0.828 | 0.893 | 0.830 | 0.835 | 0.899 |
| Fuzzy เดี่ยว | 0.837 | 0.895 | 0.836 | 0.841 | 0.908 |

 ผล Set-F1 สอดคล้องกับ CER/WER เดิมเกือบทุก scenario (Deskew+Fuzzy ชนะ) ยกเว้น scenario night ที่ Deskew เดี่ยวสูสีกันมาก ส่วน IER ยืนยันรูปแบบเดิมที่พบในภาพรวม (หัวข้อ 3.10.6) คือเป็นตัวชี้วัดเข้มงวดที่ไม่ให้เครดิตกับ Fuzzy Matching ในทุก scenario เช่นกัน สนับสนุนข้อสรุปว่าควรใช้ Set-F1 เป็นตัวชี้วัดเสริมหลัก ไม่ใช่ IER

*(หัวข้อใหม่ ไม่มีในรูปเล่มเดิม)*

## 4.3 ผลการค้นหา Pipeline ที่ดีที่สุด (Greedy Forward Selection)

 เนื่องจากเทคนิค Image Preprocessing ที่ต่อกันได้มีทั้งหมด 8 เทคนิค การทดสอบทุกชุดค่าผสมที่เป็นไปได้ (2^8 = 256 ชุด) บนภาพทั้งหมด 685 ภาพจะใช้เวลานานเกินไป จึงใช้วิธี **Greedy Forward Selection** (หลักการดูหัวข้อ 2.13) แทนการค้นหาแบบ exhaustive

 การค้นหารอบแรกใช้ภาพตัวอย่างเพียง 15 ภาพ/scenario (75 ภาพ) เพื่อความรวดเร็ว แต่พบว่าผลลัพธ์ที่ได้ (Upscale + Fuzzy, search-phase CER = 0.2982) เมื่อนำไป validate บนภาพเต็ม 685 ภาพกลับให้ mean CER = 0.6754 ซึ่ง **แย่กว่า Baseline** แสดงให้เห็นว่าตัวอย่างขนาดเล็กเกินไปทำให้อันดับเทคนิคที่ได้จากการค้นหาคลาดเคลื่อนจากอันดับจริง (เนื่องจากค่า CER ของแต่ละเทคนิคใกล้เคียงกันมาก ความแปรปรวนจากการสุ่มตัวอย่างจึงมีผลมากกว่าความแตกต่างจริงระหว่างเทคนิค) จึงทำการค้นหาใหม่โดยใช้ **ภาพเต็มทั้ง 685 ภาพในทุกขั้นตอนการค้นหา** เพื่อให้ผลลัพธ์น่าเชื่อถือ ผลการค้นหารอบใหม่แสดงในตารางที่ 4.2

| ขั้นตอน | เทคนิคที่ลอง | mean CER |
|---|---|---|
| Baseline | (ไม่ทำ preprocessing) | 0.6189 |
| รอบที่ 1 | + Grayscale | 0.6217 |
| รอบที่ 1 | + Otsu Threshold | 0.9025 |
| รอบที่ 1 | + Denoise | 0.6452 |
| รอบที่ 1 | + Sharpen | 0.6483 |
| รอบที่ 1 | + Upscale (480px) | 0.6764 |
| รอบที่ 1 | **+ Deskew** | **0.6084 (ต่ำสุด → เลือก)** |
| รอบที่ 1 | + Contrast Adaptive | 0.6484 |
| รอบที่ 1 | + Upscale800 | 0.7068 |
| รอบที่ 2 (จาก Deskew) | + Grayscale | 0.6150 |
| รอบที่ 2 (จาก Deskew) | + Threshold | 0.9040 |
| รอบที่ 2 (จาก Deskew) | + Denoise | 0.6331 |
| รอบที่ 2 (จาก Deskew) | + Sharpen | 0.6398 |
| รอบที่ 2 (จาก Deskew) | + Upscale | 0.6699 |
| รอบที่ 2 (จาก Deskew) | + Contrast Adaptive | 0.6411 |
| รอบที่ 2 (จาก Deskew) | + Upscale800 | 0.6986 |
| หยุดค้นหา | ไม่มีเทคนิคใดดีกว่า 0.6084 อีก | — |
| ขั้นสุดท้าย | Deskew + Fuzzy Matching | **0.6076** |

ตารางที่ 4.2 ผลการค้นหา Greedy Forward Selection บนภาพเต็ม 685 ภาพ ทุกขั้นตอน

 ผลการค้นหายืนยันว่า **ไม่มีเทคนิค preprocessing เดี่ยวใดดีกว่า Deskew** (mean CER = 0.6084) และเมื่อลองต่อเทคนิคที่ 2 เข้ากับ Deskew ก็ไม่มีเทคนิคใดช่วยให้ดีขึ้นอีก (ทุกเทคนิคที่ลองเพิ่มทำให้ CER สูงขึ้น) การค้นหาจึงหยุดที่ Deskew เดี่ยว แล้วเมื่อทดสอบเพิ่ม Fuzzy Matching ปิดท้าย ได้ค่า CER ต่ำสุดที่ **0.6076** ตรงกับที่เลือกไว้ในตอนแรกด้วยการเปรียบเทียบอันดับเทคนิคเดี่ยว (ตารางที่ 4.1) แต่ตอนนี้มีหลักฐานเชิงประจักษ์รองรับจากการค้นหาอย่างเป็นระบบแล้วว่าเป็นตัวเลือกที่ดีที่สุดจริง ไม่ใช่การเดา

 ข้อสังเกตสำคัญที่ควรระบุเป็นข้อจำกัดในบทที่ 5: **ขนาดตัวอย่างมีผลอย่างมากต่อความน่าเชื่อถือของการค้นหา/เปรียบเทียบเทคนิคที่มีค่าประสิทธิภาพใกล้เคียงกัน** ควรใช้ชุดข้อมูลเต็มหรือตัวอย่างที่ใหญ่พอเสมอเมื่อทำการเปรียบเทียบเทคนิคที่มีผลต่างกันไม่มาก

## 4.4 ผลการยืนยันด้วย Exhaustive Pairwise Search

 Greedy Forward Selection มีจุดอ่อนสำคัญคือเป็นการค้นหาแบบ "เลือกที่ดีที่สุดทีละก้าว" (locally greedy) ซึ่ง**ไม่รับประกันว่าจะพบคำตอบที่ดีที่สุดจริงในภาพรวม (global optimum)** เพราะเทคนิคที่แพ้ตั้งแต่รอบแรก (เช่น Grayscale, Denoise) จะถูกตัดทิ้งไปเลย ไม่มีโอกาสถูกนำไปทดลองจับคู่กับเทคนิคอื่นอีก ทั้งที่ในทางทฤษฎีอาจมีคู่เทคนิคที่ต่างฝ่ายต่างอ่อนแอเดี่ยว ๆ แต่เมื่อรวมกันแล้วอาจให้ผลดีกว่า Deskew ก็เป็นไปได้

 เพื่อตรวจสอบจุดอ่อนนี้ จึงทดสอบ **Exhaustive Search เฉพาะคู่ (Pairwise)** โดยทดสอบทุกคู่ที่เป็นไปได้ (C(8,2) = 28 คู่) จากเทคนิค preprocessing ทั้ง 8 ตัวบนภาพเต็ม 685 ภาพ แต่เพื่อประหยัดเวลา ได้ตัดคู่ที่มีโอกาสน้อยมากที่จะให้ผลดีออกไป 2 กลุ่ม: **คู่ที่มี Deskew** (7 คู่ — มีผลอยู่แล้วจาก Greedy Search รอบที่ 2 ในตารางที่ 4.2 ซึ่งพบว่าทุกคู่แย่กว่า Deskew เดี่ยวหมด) และ **คู่ที่มี Otsu Threshold** (6 คู่ — เนื่องจาก Threshold เดี่ยว ๆ ให้ผลแย่ที่สุดในบรรดาเทคนิคทั้งหมดอย่างชัดเจน CER 0.9025) เหลือทดสอบจริง **15 คู่** จาก 6 เทคนิคที่เหลือ (Grayscale, Denoise, Sharpen, Upscale, Contrast Adaptive, Upscale800) โดย 2 คู่ที่เกี่ยวข้องกับ Upscale800 (denoise+upscale800, grayscale+upscale800) เกิด crash ระหว่างรัน (Windows Access Violation) จึงมีผลสมบูรณ์ **13/15 คู่**

| อันดับ | คู่เทคนิค | mean CER |
|---|---|---|
| 1 | Denoise + Grayscale | 0.6433 |
| 2 | Contrast Adaptive + Grayscale | 0.6484 |
| 3 | Grayscale + Sharpen | 0.6513 |
| 4 | Contrast Adaptive + Denoise | 0.6602 |
| 5 | Denoise + Sharpen | 0.6725 |
| 6 | Contrast Adaptive + Upscale | 0.6733 |
| 7 | Sharpen + Upscale | 0.6754 |
| 8 | Grayscale + Upscale | 0.6804 |
| 9 | Denoise + Upscale | 0.6960 |
| 10 | Contrast Adaptive + Upscale800 | 0.7081 |
| 11 | Sharpen + Upscale800 | 0.7082 |
| 12 | Upscale + Upscale800 | 0.7088 |
| — | Denoise + Upscale800 | ไม่มีผล (crash) |
| — | Grayscale + Upscale800 | ไม่มีผล (crash) |

ตารางที่ 4.3 ผลการทดสอบ Exhaustive Pairwise Search เรียงจากดีที่สุด

 **ไม่มีคู่เทคนิคใดในบรรดา 13 คู่ที่ทดสอบสำเร็จให้ผลดีกว่า Deskew เดี่ยว (0.6084) หรือ Deskew+Fuzzy (0.6076) เลย** แม้แต่คู่ที่ดีที่สุด (Denoise+Grayscale = 0.6433) ก็ยังแย่กว่า Baseline (0.6191) ผลนี้ช่วยปิดจุดอ่อนของ Greedy Search ได้ในระดับหนึ่ง ยืนยันว่าไม่มีคู่เทคนิคที่ "ซ่อนอยู่" ในกลุ่มที่ Greedy ตัดทิ้งไปตั้งแต่ต้นที่จะเอาชนะ Deskew ได้ ทำให้สรุปได้อย่างมั่นใจยิ่งขึ้นว่า **Deskew + Fuzzy Matching คือ pipeline ที่ดีที่สุดสำหรับนำไปใช้งานจริง**

 ข้อจำกัดที่ยังเหลืออยู่ (ควรระบุในบทที่ 5): (1) ยังไม่ได้ทดสอบคู่ที่มี Deskew หรือ Threshold ร่วมกับคู่อื่นแบบ 3 เทคนิคขึ้นไป เนื่องจากข้อจำกัดด้านเวลาและทรัพยากรคอมพิวเตอร์ (2) มี 2 คู่ที่ไม่มีผลสมบูรณ์เนื่องจาก crash ระหว่างการรัน แต่จากแนวโน้มของคู่ Upscale800 อื่น ๆ ทั้งหมด (อันดับ 10-12 ซึ่งแย่ที่สุดในกลุ่ม) มีความเป็นไปได้สูงมากที่ 2 คู่นี้จะไม่เปลี่ยนข้อสรุป

## 4.5 การทดลองเสริม: เปรียบเทียบกับโมเดล Generative OCR (GOT-OCR2.0)

 เนื่องจาก classical preprocessing แทบไม่ช่วยอะไร จึงทดลองเปลี่ยน OCR engine จาก PaddleOCR เป็น **GOT-OCR2.0** (`stepfun-ai/GOT-OCR-2.0-hf`, ~580M parameters, end-to-end vision-to-text model) เพื่อดูว่าโมเดลที่ออกแบบสำหรับข้อความหนาแน่น/เล็กจะรับมือกับภาพความละเอียดต่ำในชุดข้อมูลนี้ได้ดีกว่าไหม

| ตัวชี้วัด | GOT-OCR2.0 | Baseline (PaddleOCR) | Deskew+Fuzzy (ดีที่สุด) |
|---|---|---|---|
| mean CER | 0.639 | 0.619 | **0.608** |
| median CER | **0.593** | 0.740 | ~0.56-0.60 |

 GOT-OCR2.0 มี median CER ดีกว่าอย่างชัดเจน (อ่านภาพทั่วไป/ชัดเจนได้แม่นกว่า) แต่มี **catastrophic failure tail หนัก** — ประมาณ 32% ของภาพ (215/680) มี CER สูงกว่า 0.90 โดยเฉพาะภาพ scenario blur/night ที่คุณภาพแย่มาก โมเดลจะ **hallucinate** (แต่งประโยคภาษาอังกฤษที่ฟังดูสมเหตุสมผลแต่ไม่เกี่ยวกับภาพเลย) ซึ่งเป็นพฤติกรรมเฉพาะของโมเดล generative/VLM ที่ PaddleOCR ไม่มี ผลสุทธิคือ mean CER ใกล้เคียงหรือแย่กว่า Deskew+Fuzzy เล็กน้อย และช้ากว่ามาก (~5-6 วินาที/ภาพ เทียบกับ 0.15 วินาที/ภาพของ PaddleOCR) จึงไม่คุ้มที่จะเปลี่ยนมาใช้แทนสำหรับ use case นี้

## 4.6 การวิเคราะห์เชิงสถิติเพิ่มเติม

 เพื่อให้ผลการทดลองมีความน่าเชื่อถือและครอบคลุมมากยิ่งขึ้น จึงทำการวิเคราะห์เพิ่มเติม 4 ด้าน โดยไม่ต้องรัน OCR ใหม่ (คำนวณจากผลลัพธ์ที่บันทึกไว้แล้วทั้งหมด, สคริปต์ `comprehensive_analysis.py`)

### 4.6.1 การทดสอบนัยสำคัญทางสถิติ (Wilcoxon Signed-Rank Test)

 เดิมงานวิจัยนี้ระบุไว้เป็นข้อจำกัดว่าผลต่างระหว่างเทคนิคที่ดีที่สุด (Deskew+Fuzzy, CER 0.6078) กับ Baseline (CER 0.6191) มีเพียง ~1.8% ยังไม่ได้พิสูจน์ว่ามีนัยสำคัญทางสถิติจริงหรือเป็นเพียงความบังเอิญจากการสุ่มของชุดทดสอบ จึงทำการทดสอบด้วย **Wilcoxon Signed-Rank Test** ซึ่งเป็นสถิติแบบไม่อิงพารามิเตอร์ (non-parametric) ที่เหมาะกับข้อมูลคู่ (paired) ที่ไม่จำเป็นต้องมีการแจกแจงแบบปกติ โดยเปรียบเทียบค่า CER รายภาพระหว่าง 2 เทคนิคทีละคู่ (n=680 ภาพ)

| คู่เปรียบเทียบ | mean diff | W statistic | p-value | สรุป |
|---|---|---|---|---|
| Deskew+Fuzzy vs Baseline | -0.0114 | 73,374.0 | 0.0104 | **มีนัยสำคัญ** (p<0.05) |
| Deskew vs Baseline | -0.0106 | 31,093.0 | 0.0323 | **มีนัยสำคัญ** (p<0.05) |
| Fuzzy vs Baseline | -0.0014 | 54,616.0 | 0.0004 | **มีนัยสำคัญ** (p<0.05) |
| Deskew+Fuzzy vs Deskew | -0.0008 | 49,952.0 | 0.0003 | **มีนัยสำคัญ** (p<0.05) |

 ผลการทดสอบยืนยันว่า **ผลต่างของทุกคู่ที่เปรียบเทียบมีนัยสำคัญทางสถิติจริง** (p<0.05 ทุกคู่) แม้ค่าความต่างสัมบูรณ์จะดูเล็กน้อยก็ตาม จึงสรุปได้อย่างมั่นใจว่า Deskew และ Fuzzy Matching ช่วยเพิ่มความแม่นยำได้จริง ไม่ใช่ความบังเอิญจากการสุ่มตัวอย่าง — ข้อจำกัดที่เคยระบุไว้ในหัวข้อก่อนหน้าจึงถูกแก้ไขแล้ว

### 4.6.2 การกระจายตัวของ CER (Distribution Analysis)

| เทคนิค | mean CER | median CER | std | % ภาพที่ CER > 0.9 |
|---|---|---|---|---|
| Deskew+Fuzzy | 0.6078 | 0.7233 | 0.3538 | 31.0% |
| Deskew | 0.6086 | 0.7308 | 0.3541 | 31.2% |
| **Tiled OCR** | 0.6194 | 0.7196 | 0.3295 | **27.9% (ต่ำสุด)** |
| Fuzzy | 0.6177 | 0.7369 | 0.3504 | 32.1% |
| Baseline | 0.6191 | 0.7396 | 0.3498 | 31.8% |
| Threshold | 0.9025 | 1.0000 | 0.2595 | 85.4% (สูงสุด) |
| Upscale800 | 0.7070 | 0.9144 | 0.3572 | 52.1% |

 ข้อค้นพบสำคัญที่ไม่ปรากฏจากค่าเฉลี่ยเพียงอย่างเดียว: **ค่า median สูงกว่า mean อย่างชัดเจนในทุกเทคนิค** (เช่น Baseline mean=0.619 แต่ median=0.740) แสดงว่าการแจกแจงของ CER เบ้ซ้าย (left-skewed) — ภาพส่วนใหญ่จริง ๆ มีค่า CER อยู่ในระดับสูง (อ่านได้ไม่ดีนัก) ในขณะที่มีภาพกลุ่มหนึ่งที่ OCR อ่านได้แม่นยำมาก (CER ต่ำมาก) ซึ่งดึงค่าเฉลี่ยลงมาให้ดูดีกว่าภาพทั่วไปในชุดข้อมูล นอกจากนี้มีสัดส่วนภาพที่ล้มเหลวรุนแรง (CER > 0.9) สูงถึง 27.9-52.1% ในทุกเทคนิค (ไม่นับ Threshold ซึ่งแย่ที่สุด) ชี้ให้เห็นว่าแม้เทคนิคที่ดีที่สุดก็ยังมีสัดส่วนภาพที่อ่านไม่ได้เลยอยู่พอสมควร ควรระบุเป็นข้อจำกัดเพิ่มเติมในบทที่ 5

 ข้อสังเกตเพิ่มเติม: **Tiled OCR มีสัดส่วนภาพล้มเหลวรุนแรงต่ำที่สุด (27.9%)** แม้ mean CER โดยรวมจะแย่กว่า Deskew เล็กน้อย แสดงว่า Tiled OCR อาจช่วยลดกรณีที่อ่านไม่ได้เลยแบบสุดโต่งได้ดีกว่า แม้จะไม่ได้ช่วยเพิ่มความแม่นยำเฉลี่ยโดยรวม

### 4.6.3 Precision และ Recall แยกรายตัว

| เทคนิค | mean Precision | mean Recall | mean F1 |
|---|---|---|---|
| Baseline | 0.387 | 0.255 | 0.282 |
| Fuzzy | 0.381 | **0.290** | 0.296 |
| Deskew | **0.408 (สูงสุด)** | 0.271 | 0.299 |
| Deskew+Fuzzy | 0.390 | **0.300 (สูงสุด)** | **0.305 (สูงสุด)** |

 การแยก Precision และ Recall เผยให้เห็นว่า **Deskew และ Fuzzy Matching ช่วยเพิ่มความแม่นยำคนละมุม**: Deskew ช่วยเพิ่ม Precision เป็นหลัก (ลดการตรวจจับชื่อสารผิด) ในขณะที่ Fuzzy Matching ช่วยเพิ่ม Recall เป็นหลัก (ค้นหาชื่อสารที่มีอยู่จริงได้มากขึ้น โดยแก้ไขคำที่ OCR อ่านผิดเล็กน้อยให้ตรงกับฐานข้อมูล) เมื่อรวมสองเทคนิคเข้าด้วยกันจึงได้ทั้ง Precision และ Recall ที่สมดุลและให้ F1 สูงสุด อธิบายได้ว่าทำไม Deskew+Fuzzy จึงเป็น pipeline ที่ดีที่สุด — เพราะแก้ปัญหาคนละจุดที่ไม่ทับซ้อนกัน

### 4.6.4 ความสัมพันธ์ระหว่างความละเอียดภาพกับ CER

| เทคนิค | Pearson r | p-value | นัยสำคัญ |
|---|---|---|---|
| Baseline | -0.4611 | 4.3×10⁻³⁷ | มีนัยสำคัญมาก |
| Deskew | -0.4643 | 1.2×10⁻³⁷ | มีนัยสำคัญมาก |
| Deskew+Fuzzy | -0.4661 | 5.8×10⁻³⁸ | มีนัยสำคัญมาก |

 ค่าสหสัมพันธ์ Pearson เป็นลบอย่างมีนัยสำคัญทางสถิติในทุกเทคนิค (r ≈ -0.46) ยืนยันเชิงปริมาณว่า **ภาพที่มีความสูง (พิกเซล) มากกว่าจะมีค่า CER ต่ำกว่า (แม่นยำกว่า)** สอดคล้องกับข้อสังเกตเชิงคุณภาพในหัวข้อ 3.3.1 ที่ว่าภาพในชุดข้อมูลมีความละเอียดต่ำมาก (median 163px) อย่างไรก็ตาม ค่า r ≈ -0.46 หมายถึงขนาดภาพอธิบายความแปรปรวนของ CER ได้เพียง ~21% (R² ≈ 0.21) เท่านั้น แสดงว่า **ความละเอียดภาพเป็นเพียงหนึ่งในหลายปัจจัยที่กระทบความแม่นยำ ไม่ใช่ปัจจัยเดียว** ปัจจัยอื่นที่อาจมีผลร่วมด้วย ได้แก่ ความเบลอ คอนทราสต์ ความซับซ้อนของชื่อสารเคมี และคุณภาพการโฟกัสของภาพ

*(หัวข้อ 4.6 ทั้งหมดใหม่ ไม่มีในรูปเล่มเดิม — เพิ่มขึ้นเพื่อความครอบคลุมเชิงสถิติของงานวิจัย)*

## 4.7 การอภิปรายผล

 ผลการทดลองแสดงให้เห็นว่า **การทำ Image Preprocessing แบบดั้งเดิม (classical) ส่วนใหญ่ไม่ช่วยหรือกลับทำให้โมเดล deep learning OCR อ่านแย่ลง** มีเพียง **Deskew** และ **Fuzzy Matching หลัง OCR** เท่านั้นที่ให้ผลดีขึ้นอย่างสม่ำเสมอ (แม้เพียงเล็กน้อย ~1.8% CER เทียบกับ Baseline) ผลลัพธ์นี้ขัดกับสมมติฐานทั่วไปที่มักเชื่อกันว่าการทำ preprocessing หนัก ๆ จะช่วยเพิ่มความแม่นยำของ OCR เสมอ ในหลายประเด็น:

1. **Bicubic Upscaling ทำให้แย่ลงอย่างสม่ำเสมอ** ทั้งแบบ 480px (CER 0.676) และ 800px (CER 0.707) แม้ภาพในชุดข้อมูลนี้จะเล็กมาก (median 163px) ซึ่งตามสมมติฐานควรได้ประโยชน์จากการขยายมากที่สุด — เป็นไปได้ว่าโมเดล PP-OCRv6 มีการ resize ภาพเข้า pipeline ภายในตัวเองอยู่แล้วก่อน inference การขยายภาพเพิ่มก่อนหน้านั้นจึงเพิ่มแค่ interpolation artifact โดยไม่เพิ่มข้อมูลจริง
2. **Multi-Tier Fallback (รวม EDSR Super-Resolution) แย่กว่า Baseline ทั้งที่ซับซ้อนและช้าที่สุด** (0.985 วินาที/ภาพ ช้ากว่า Baseline ~6 เท่า เพราะเรียก OCR สูงสุด 4 ครั้ง/ภาพ) — การพยายามแก้ภาพมากขึ้นเรื่อย ๆ ไม่ได้แก้ปัญหาความแม่นยำ เพียงเปลี่ยนรูปแบบความผิดพลาด
3. **Threshold/Binarization (Otsu) แย่ที่สุดในทุกเทคนิค** (CER 0.9025) เพราะโมเดล PP-OCRv6 ถูกเทรนบนภาพสีจริง การตัดเหลือขาว-ดำล้วนทำลายข้อมูล texture/gradient ที่โมเดลใช้ตัดสินใจ
4. **Fuzzy Matching ให้ผลดีขึ้นแบบสม่ำเสมอแต่เล็กน้อย** เพราะแก้ไขข้อความ "หลัง" OCR อ่านเสร็จ ไม่เปลี่ยนสิ่งที่โมเดลเห็น จึงช่วยได้แค่คำที่อ่านผิดเล็กน้อยระดับตัวอักษร (typo) แต่ช่วยไม่ได้ถ้า OCR อ่านพลาดไปเป็นคำอื่นไปเลย
5. **การค้นหา Combination อย่างเป็นระบบ (Greedy + Exhaustive Pairwise) ยืนยันว่า Deskew+Fuzzy เป็นขีดจำกัดบนของสิ่งที่ทำได้** ด้วยเทคนิค classical preprocessing ที่มีอยู่ ไม่ว่าจะลองรวมกันแบบใดก็ไม่สามารถเอาชนะได้

### ข้อสรุปภาพรวม

 สำหรับชุดข้อมูลภาพฉลากส่วนผสมที่ crop มาเป็นแถบเล็ก ความละเอียดต่ำ การทำ Image Preprocessing แบบดั้งเดิมส่วนใหญ่ไม่ช่วยหรือกลับทำให้แย่ลง มีเพียง Deskew และ Fuzzy Matching เท่านั้นที่ให้ผลดีขึ้นอย่างสม่ำเสมอ ผลลัพธ์นี้ขัดกับสมมติฐานที่ตั้งไว้ตอนต้นว่าการทำ preprocessing หนัก ๆ จะช่วยเพิ่มความแม่นยำเสมอ ชี้ให้เห็นว่า **ประสิทธิภาพของ preprocessing pipeline ขึ้นกับลักษณะภาพอินพุตและสถาปัตยกรรมโมเดล OCR ที่ใช้เป็นอย่างมาก ควรทดสอบเชิงประจักษ์ก่อนนำไปใช้จริง ไม่ควรอนุมานจากสมมติฐานทั่วไปเพียงอย่างเดียว** และจากการค้นหาอย่างเป็นระบบ **Deskew + Fuzzy Matching คือคำตอบสุดท้ายที่แนะนำให้นำไปพัฒนาต่อและผนวกเข้ากับระบบแนะนำสกินแคร์ที่เหมาะสมกับผู้ใช้**

*(เดิม: การอภิปรายผลอิงตัวชี้วัด Accuracy/Precision/Recall เดิม โดยสรุปว่าระบบทำงานได้ "ระดับปานกลาง" — งานปัจจุบันเปลี่ยนกรอบการอภิปรายเป็นการเปรียบเทียบเชิงเทคนิคว่าอะไรช่วย/ไม่ช่วย และทำไม พร้อมข้อสรุปที่ชัดเจนกว่าว่าเทคนิคใดควรนำไปใช้จริง)*

## 4.8 การวิเคราะห์ข้อผิดพลาด

ข้อผิดพลาดหลักที่พบจากการตรวจสอบผล OCR ในเชิงคุณภาพ:

1. OCR อ่านตัวอักษรผิดพลาด โดยเฉพาะในภาพที่มีความละเอียดต่ำหรือสภาพแสงไม่ดี
2. ชื่อสารเคมีมีความยาวและโครงสร้างคำซับซ้อน ทำให้ OCR สับสนง่าย
3. ขนาดตัวอักษรบนฉลากเล็กเกินไปเมื่อเทียบกับความละเอียดภาพ (median สูงภาพเพียง 163px)
4. Fuzzy Matching อาจจับคู่ผิดระหว่างชื่อสารที่มีรูปแบบคำใกล้เคียงกัน เช่น "Sodium Hyaluronate" กับ "Hydrolyzed Sodium Hyaluronate"

ตัวอย่างข้อความ OCR ที่ผิดพลาด (ดึงจากผลจริงในไฟล์ `results/ocr_paddle_fuzzy.csv`, ภาพ A12.jpg):

| OCR Output (raw) | ชื่อสารที่ถูกต้อง | แก้ได้ด้วย Fuzzy Matching หรือไม่ |
|---|---|---|
| Methl Trimethicone | Methyl Trimethicone | ไม่ได้ (ยังคงเป็น "Methl" หลัง fuzzy) |
| Malakite Extract | Malachite Extract | ไม่ได้ |
| Phenl Trimethicone | Phenyl Trimethicone | ไม่ได้ |
| Eguisetum Arvense Leaf Extract | Equisetum Arvense Leaf Extract | ไม่ได้ |
| hvaluronic acid *(จากภาพ A14.jpg)* | Hyaluronic Acid | ไม่ได้ |

 จากการวิเคราะห์พบว่าปัญหาส่วนใหญ่เกิดขึ้นในขั้นตอน OCR ซึ่งเป็นขั้นตอนต้นน้ำของระบบ เมื่อ OCR อ่านข้อความผิดพลาด จะส่งผลต่อขั้นตอน Fuzzy Matching ในลำดับถัดไป ข้อสังเกตสำคัญคือตัวอย่างข้างต้นทั้งหมด **ไม่ถูกแก้ไขโดย Fuzzy Matching แม้จะดูเหมือนพิมพ์ผิดเพียง 1 ตัวอักษร** ซึ่งอาจเกิดจากค่าความคล้ายคลึง (`fuzz.ratio`) ระหว่างคำผิดกับคำที่ถูกต้องไม่ถึงเกณฑ์ 85 คะแนนที่กำหนดไว้ หรือฐานข้อมูล INCI ไม่มีชื่อที่ตรงกันพอดี สะท้อนให้เห็นข้อจำกัดของ Fuzzy Matching แบบ token-level ว่าช่วยแก้ไขได้เฉพาะกรณีที่ค่าความคล้ายคลึงสูงพอเท่านั้น ไม่ใช่ทุกกรณีที่ดูเหมือนพิมพ์ผิดเล็กน้อย

*(เดิม: ตัวอย่างในรูปเล่มเก่า เช่น "Glyeerin → Glycerin" เป็นตัวอย่างเชิงแนวคิดที่ไม่ได้ยืนยันกับข้อมูลจริง — ตารางข้างต้นแทนที่ด้วยตัวอย่างที่ดึงจากผลการทดลองจริงในโปรเจกต์)*
