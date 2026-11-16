# วิธีประเมิน YOLO บน CCTV600

**Protocol 1.0.0 · FROZEN_FOR_PREFLIGHT · 16 พฤศจิกายน 2026**

## 1. ขอบเขต

ทดสอบ pretrained segmentation checkpoints 17 ตัว แบ่งเป็น X/E, L/C, M, S และ N ตามลำดับ รายชื่ออยู่ใน `configs/study.json` การวัดหลักใช้กรอบบุคคลที่มองเห็น แต่ละตระกูลมีจำนวน parameters และ pretraining ต่างกัน จึงตีความผลเป็นการเปรียบเทียบ checkpoint พร้อม configuration

ตัวควบคุมรัน quality ตามกลุ่มขนาดจนครบ 17 ตัว แล้วทำ dedicated timing และสร้างรายงาน ระหว่างรันตรวจสุขภาพงาน ความถูกต้อง และความครบถ้วน โดยคง settings ตาม protocol

## 2. Dataset และ GT

ใช้ CCTV Human Parsing 600 รุ่น 1.0.0 จำนวน 600 unique image IDs ตรวจ image/annotation SHA256 กับ manifest ที่รับรอง และเก็บ derived bbox GT ใน `PREPARED.json` ข้อมูลภาพและ annotation ต้นฉบับคงเดิม

ฐานของ person inventory คือคำยืนยันผู้ใช้ว่าตรวจรับ 600 ภาพตามคู่มือที่กำหนดให้ตรวจทุกคนด้านยาวตั้งแต่ 50 px จึงใช้กรอบและภาพไม่มี bbox เป็นคำตอบอ้างอิงภายใต้ขอบเขตนี้ งานนี้ยังไม่มี independent review ครบทุกคนเพิ่มเติม

มี bbox 774 รายการ ผ่านเกณฑ์ขนาด 769 รายการ และคงคนต่ำกว่า 50 px อีก 5 รายการ Eligibility วัดจาก source visible bbox ก่อน clipping ส่วนกรอบที่เกินขอบเล็กน้อยจะ clip ที่ขอบภาพแบบ continuous เฉพาะ derived GT สำหรับ scoring พร้อมเก็บพิกัดต้นฉบับไว้

Ignore union ประกอบด้วย shapes ที่มี label `ignore` และ bbox ของคนต่ำกว่า 50 px การอ่าน ignore ให้ยึด label แม้มี legacy owner ID ค้าง ข้อสังเกตของ part shapes และ part classes ยังคงอยู่ใน dataset โดยไม่กระทบการแก้ GT สี่ส่วนจากงาน detector นี้

## 3. Inference

ใช้ batch 1, FP32, input 640×640, centered aspect-preserving letterbox, padding 114, RGB/255, rect=False และ retina_masks=True คง no AMP, no TF32 และ no extra TTA

YOLO26 ใช้ one-to-many/NMS path ก่อน fusion เช่นเดียวกับ runtime งาน MOTS20 ที่นำมาใช้ บันทึก unfused checkpoint parameters และ fused runtime parameters แยกกัน ที่มาและขอบเขตการแก้ source อยู่ใน `manifests/RUNTIME_ORIGIN.json`

เลือก Person model class 0 ใช้ AP confidence floor 0.001 ตาม framework NMS, NMS box IoU 0.70 และ model max_det 1000 หาก cap อิ่มหรือมี NMS truncation warning ให้หยุดพร้อมหลักฐาน

เก็บ boxes, scores, image-local prediction IDs และ original-grid native masks เป็น lossless COCO RLE พร้อม bindings ของ input/checkpoint/source บันทึกจำนวน post-NMS และ retained outputs แยกกัน เพราะ native empty-mask filtering เป็นส่วนหนึ่งของ segmentation route ที่ใช้

## 4. Bbox matching และ AP

Fixed operating point ใช้ confidence ≥0.25 และ bbox IoU ≥0.50 เรียง prediction ตาม score แล้ว geometry จับคู่ valid GT หนึ่งต่อหนึ่งด้วย best IoU และ geometry tie-break ก่อนพิจารณา ignore

Unmatched prediction ที่มีพื้นที่ตัดกับ ignore union หารด้วยพื้นที่ prediction box ตั้งแต่ 0.50 จะถูกยกเว้น ที่เหลือเป็น FP ส่วน eligible GT ที่ไม่ match เป็น FN การนับพื้นที่ ignore ใช้ union ของ continuous rectangles เพื่อหลีกเลี่ยงการบวกพื้นที่ทับกันซ้ำ

AP ใช้ COCO-style bbox evaluator ที่ IoU 0.50:0.05:0.95, 101 recall points และ maxDets 1/10/200 โดยปรับ coarse-ignore union ตามกติกาข้างต้น Extract summary จาก precision/recall tensors ที่ cap จริงโดยตรง

AP maxDet 200 เป็น common baseline ที่ประกาศไว้ก่อนรัน ตรวจเทียบ reference 1000 บน 100 ภาพคงที่ของแต่ละโมเดล ก่อนขยาย full quality ในกลุ่มนั้น ผลต่าง AP50, AP75 และ AP50–95 ทุกค่าต้องน้อยกว่า 0.0001 หากไม่ผ่านให้หยุดและเก็บ `CAP_CHECK.json` เพื่อทบทวน common policy

รายงาน AP, AP50, AP75, AR, TP/FP/FN, Precision/Recall/F1, TP-only box IoU และ support แยกกัน PR curve ใช้ IoU 0.50 ส่วน F1-confidence curve คำนวณจาก saved predictions เพื่ออธิบาย trade-off โดย main operating point ยังคง 0.25

## 5. เวลางานจริงและ clean timing

Quality pass เก็บเวลาต่อภาพของ input read/hash/decode, preprocessing/H2D, forward, postprocessing/native-mask transfer, RLE preparation และ packet compression/write พร้อม wall time จน prediction packet เขียนเสร็จ เวลาหลัง write เก็บใน receipt ที่ผูก packet SHA แยกจาก immutable packet

เวลาจาก quality 600 ภาพเป็น active-processing diagnostics ซึ่งรวม cold/session effects ได้ Dedicated timing ใช้ 100 ภาพที่เลือกด้วย seed 20261116 และ metadata round-robin ตามกล้อง สถานที่ และกลางวัน/กลางคืนก่อนเห็นผล ทุกโมเดลใช้สมาชิกและลำดับภาพเดียวกัน

โหลดทีละโมเดล ทำ warmup 10 calls ต่อรอบ แล้ววัด 100 ภาพสาม clean rounds รวม 300 measured observations และ 30 warmup calls ต่อโมเดล ใช้ seeded/cyclic model order ระหว่างรอบ

Synchronize CUDA ที่ stage boundaries Adapter pipeline รวม preprocessing, forward และ postprocessing ส่วน input I/O, RLE และ output writing วัดแยก FPS คำนวณจาก 1000 หาร mean adapter ms ค่า mean, P50, P95 และ population std คำนวณจาก observations ทั้ง 300 ที่ตรวจรับ

รอบเวลาต้องไม่มี external GPU compute และ preload idle utilization ต้องไม่เกิน 5% หลังงานของตัวเองมี settle wait ได้ไม่เกิน 10 วินาที หาก GPU ยังไม่พร้อมให้เลื่อน timing โดยเก็บ quality results ส่วนรอบที่ปนเปื้อนระหว่างวัดเก็บเป็น diagnostics

Reset allocated/reserved peaks หลัง warmup โดยรวม resident model เก็บ model loading เป็นวินาทีแยก GFLOPs เป็น static estimate จากงานเดิมที่ตรวจ exact checkpoint SHA256 ตรงกันและใช้ input 640/NMS path เดียวกัน

## 6. ทรัพยากรและ failure policy

หนึ่ง study GPU worker ทำงานตาม device visibility ที่อนุญาต Admission ก่อนโหลดต้องมี free VRAM 4 GiB และ free disk 20 GiB จำกัด study outputs 20 GiB และเวลา 8 ชั่วโมงต่อ explicit controller attempt ไม่มี automatic model retry ค่าเหล่านี้เป็น guards ของ study นี้

Preflight 100 ภาพต่อโมเดลเป็น canonical observations ได้เมื่อ contract ผ่าน จากนั้น quality รันอีก 500 ภาพ รวม 600 quality calls ต่อโมเดล Dedicated timing อีก 330 calls แยกจาก accuracy และนับใน operational resources

OOM, CUDA error, invalid packet หรือ resource guard ล้มทำให้หยุด เก็บ tracebacks, attempt ledger และ packets ที่สำเร็จ Setup failures, failed image execution และ artifact/scorer failures แยกตาม phase และ ledger โดยไม่ปรับ resolution, precision หรือ device เพื่อข้ามปัญหา

เมื่อ interruption ให้ตรวจ process/status และ reconcile ก่อน explicit resume คง hashes, bindings, old attempts และผลที่ใช้ได้ Packet ที่ขาด receipt ยังต้องตรวจ contract ก่อน reuse ส่วนเวลาที่ขาดเป็น null รอบเวลาที่ล้มหรือปนเปื้อนต้องมี fresh controlled segment

Primary ของแต่ละ checkpoint ใช้ STRICT_COMPLETE บน 600 image IDs ถ้ามี missing, unexpected หรือ invalid ให้ primary เป็น null เวลาหลักแสดงเมื่อครบสาม clean rounds ความล้มเหลวในการ push Git เก็บแยกจาก scientific completion

## 7. Evidence และขอบเขตคำอ้าง

CPU acceptance ใช้คำตอบคาดหวังอิสระผ่าน actual packet writer, readback, scorer และ reporter ก่อน GPU ผูก source/config/checkpoint/environment ใน PREPARED และ LAUNCH พร้อม CUDA visibility, process identity, commit และคำสั่งจริง

Controller ทำงาน detached ผ่าน tmux พร้อม process lock และ PID start token การหยุดแบบปกติทำที่ขอบภาพ Progress viewer อ่านสถานะและ artifacts โดยไม่เปิด inference

Public repositories เก็บ code, configs, docs และ aggregates ที่ตรวจรับ Raw packets, per-image CSV, metadata และ gallery 16 ภาพชุดคงที่อยู่ใน outputs ของแต่ละ tier ภาพเปรียบเทียบแสดง GT boxes กับ native predicted masks เพื่ออธิบายผล

ข้อมูลยังขาด human-accepted full-person masks สำหรับ Person Mask AP, track IDs สำหรับ tracking และ GT สีครบสำหรับ color accuracy ประวัติ training/pretraining ที่ไม่มีหลักฐานให้ระบุ unknown คะแนนชุดนี้ยังไม่มี uncertainty analysis หรือข้อรับรอง deployment
