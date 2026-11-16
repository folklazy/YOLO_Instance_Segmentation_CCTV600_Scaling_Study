# วิธีอ่านคะแนนและเวลาของ YOLO CCTV600

คู่มือนี้ใช้กับ protocol 1.0.0 ของ CCTV600 และผลสรุป 17 checkpoints รายงานค่าคุณภาพของ **กรอบบุคคล** หน่วยคะแนนใน CSV เป็นสัดส่วน 0–1 ส่วนตารางภาษาไทยแสดงเป็นเปอร์เซ็นต์

## 1. หน่วยที่นับและ GT

หนึ่ง observation ของการทดสอบคุณภาพคือหนึ่งภาพ แต่ละภาพมีบุคคลได้หลายคน ชุดนี้มี 600 ภาพและกรอบคน 774 รายการ ก่อนประเมินใช้ด้านยาวของ visible bbox ต้นฉบับคัดคนตั้งแต่ 50 px ได้ 769 คน ส่วนคนต่ำกว่าเกณฑ์ 5 คนเก็บเป็น ignore พร้อม explicit ignore rectangles อีก 172 รายการ กรอบที่เกินขอบภาพเล็กน้อยถูก clip เฉพาะ derived scoring GT

GT bbox ล้อมส่วนของคนที่มองเห็นในภาพ คนที่ถูกบังหรือถูกตัดด้วยขอบเฟรมจึงมีกรอบตามส่วนที่เห็น การไม่มี full-person silhouette ยังทำให้วัดความแม่นของขอบ mask ทั้งตัวไม่ได้ พื้นที่สี่ส่วน Hair/Face/Upper/Lower ใน dataset เป็น annotation อีกระดับ งาน detector รอบนี้ใช้ bbox เท่านั้น

ตรวจคนเข้าเกณฑ์ครบทั้ง 600 ภาพตามการรับรองของผู้จัดทำ dataset ยังไม่มีผู้ตรวจอิสระทวนบัญชีบุคคลทุกคนอีกครั้ง ภาพไม่มี eligible GT ยังอยู่ในประชากร 600 ภาพ prediction ที่อยู่นอก ignore และจับคู่ใครไม่ได้ยังเป็น FP

## 2. การจับคู่และความหมายของ TP, FP, FN

ใช้ confidence ≥0.25 และ bbox IoU ≥0.50 สำหรับคะแนนที่ operating point คงที่ เรียง predictions ตามคะแนนจากมากไปน้อย แล้วใช้พิกัดกรอบตัดสินเมื่อคะแนนเท่ากัน จับคู่กับ valid GT ที่มี IoU สูงสุดที่ยังว่างแบบหนึ่งต่อหนึ่ง

- **TP** คือกรอบทำนายที่จับคู่ GT ได้ตามเกณฑ์
- **FN** คือ GT ที่ยังไม่มี prediction จับคู่ คนที่ตรวจไม่พบหรือกรอบคลาดจน IoU ต่ำกว่าเกณฑ์อาจเป็น FN ได้
- **FP** คือ prediction ที่จับคู่ valid GT ไม่ได้และไม่ผ่านกติกายกเว้น ครอบคลุมกรอบซ้ำ กรอบผิดตำแหน่ง และการตรวจสิ่งอื่นเป็นคน
- **Ignored prediction** คือ prediction ที่จับคู่ valid GT ไม่ได้ แต่พื้นที่กรอบทับ ignore union ตั้งแต่ครึ่งหนึ่งของพื้นที่ prediction กรอบนี้แยกออกจาก TP/FP

ตัวอย่างมี GT 3 คน โมเดลตรวจถูก 2 คน พลาด 1 คน และส่งกรอบเกินอีก 1 กรอบ จะได้ TP=2, FP=1, FN=1 จึงได้ Precision=Recall=F1=2/3 หรือ 66.67% กรอบที่คลาดมากอาจทำให้เกิดทั้ง FP หนึ่งรายการและ FN หนึ่งรายการ การตัดสินนี้เป็นผลของ bbox matching

งาน detection ไม่มีจำนวน True Negative ที่นิยามตามจำนวนกรอบได้ตายตัว จึงใช้ TP/FP/FN แทนการสร้าง confusion matrix ที่เติม TN จากพื้นที่ว่าง

## 3. สูตรและคำถามที่แต่ละตัวชี้วัดตอบ

| ตัวชี้วัด | คำนวณอย่างไร | ใช้ตอบคำถาม |
| --- | --- | --- |
| Box IoU | พื้นที่ intersection ของกรอบ / พื้นที่ union ของสองกรอบ | กรอบทำนายครอบตำแหน่งคนตรงกับ GT เพียงใด |
| Precision | TP / (TP + FP) | กรอบที่ระบบส่งออกและไม่ได้ถูก ignore ถูกต้องกี่ส่วน |
| Recall | TP / (TP + FN) | คนใน GT ที่เข้าเกณฑ์ถูกตรวจพบกี่ส่วน |
| F1 | 2TP / (2TP + FP + FN) | สมดุลระหว่างการพลาดคนกับกรอบเกินที่ threshold เดิม |
| Matched box IoU mean | ผลรวม IoU ของคู่ TP / จำนวน TP | กรอบของคนที่จับคู่สำเร็จแนบ GT เพียงใด |
| AP50 | AP ที่ IoU 0.50 | คุณภาพการเรียง confidence เมื่อใช้เกณฑ์กรอบค่อนข้างผ่อนปรน |
| AP75 | AP ที่ IoU 0.75 | ผลเมื่อกำหนดให้กรอบตรงมากขึ้น |
| Box AP@[.50:.95] | เฉลี่ย AP ของ IoU 10 ระดับ ตั้งแต่ 0.50 ถึง 0.95 ทีละ 0.05 | คุณภาพรวมทั้งการตรวจพบและความตรงของกรอบหลายความเข้มงวด |
| Box AR@200 | เฉลี่ย maximum recall ของ IoU 10 ระดับ เมื่อเก็บไม่เกิน 200 detections/ภาพ | เพดาน recall ภายใต้รายการทำนายที่ใช้ประเมิน AP |

Precision/Recall/F1 รวมทั้ง dataset ใช้ผลรวม TP/FP/FN ก่อนหาร จึงเป็น micro aggregation Matched IoU เฉลี่ยเฉพาะ TP คนที่พลาดไม่ได้ถูกใส่ IoU=0 ต้องอ่าน Recall ประกอบ โมเดลต่างกันอาจมีชุด TP ต่างกัน

AP ใช้ score-ranked precision–recall curve แล้วหา interpolated precision ที่ 101 recall points จาก 0 ถึง 1 จากนั้นเฉลี่ย ค่า AP เป็นการประเมินหลาย confidence thresholds ส่วน P/R/F1 ในตารางหลักใช้ confidence เดียวที่ตรึงไว้ การมี Person เพียงคลาสเดียวทำให้ค่า mean over classes มีคลาสเดียวในงานนี้

ตัวอย่างกรอบ A และ B พื้นที่ 100 หน่วยเท่ากัน ทับกัน 60 หน่วย จะได้ union =100+100−60=140 และ IoU=60/140≈0.429 จับคู่ไม่ได้เมื่อเกณฑ์ 0.50 แม้ภาพดูว่ากรอบทับกันอยู่

เมื่อ denominator เป็นศูนย์ implementation คืน JSON `null` และ CSV ช่องว่าง รายงานแสดง `—` การแสดงผลไม่แทนค่าดังกล่าวด้วยศูนย์ ชุดรวมนี้มี GT 769 คนจึงคำนวณ Recall รวมได้

## 4. ส่วนที่ยืมจาก COCO และส่วนที่กำหนดเฉพาะ CCTV600

ใช้ pycocotools 2.0.11 ใน environment ที่บันทึกกับ run ยึด IoU 10 ระดับและ 101 recall points ตาม COCO evaluator ที่มาและค่า defaults ตรวจได้จาก source ต้นทาง [COCO detection evaluation](https://cocodataset.org/#detection-eval) และ [COCOeval source](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)

Protocol CCTV600 ใช้ maxDets [1,10,200] แทนค่า default [1,10,100] และเติมกติกา coarse-ignore union ให้ unmatched detection ที่ทับ ignore ตาม IoA ได้รับการยกเว้น ใช้ valid GT ก่อน ignore เสมอ การสรุป AP/AR อ่าน precision/recall tensors ที่ cap 200 จริงโดยตรง ตรวจเทียบ cap 1000 บน 100 ภาพเดิมผ่านทุกโมเดลแล้ว ส่วน source ของกติกานี้อยู่ใน [scoring.py](src/cctv600/scoring.py)

IoA สำหรับ ignore ใช้พื้นที่กรอบ prediction ที่ทับ union ของ ignore rectangles หารพื้นที่ prediction คำนวณ union เพื่อหลีกเลี่ยงพื้นที่ยกเว้นที่ทับกันถูกบวกซ้ำ การนำ bbox คนเล็กมาเป็น ignore เป็นข้อตกลงเฉพาะข้อมูลชุดนี้ จึงต้องกำกับว่าเป็น COCO-style bbox evaluation with dataset-specific ignore policy

AP confidence floor 0.001, NMS IoU 0.70, model max_det 1000 และ retina masks เป็นค่าตาม frozen segmentation runtime ผลหลัง native empty-mask filtering เป็นรายการที่นำมาประเมิน ต้องเก็บความหมายนี้ไว้เมื่อเทียบกับ raw detection-only head

## 5. กราฟและผลตามกลุ่ม

**PR curve** แสดงความสัมพันธ์ precision กับ recall ที่ bbox IoU 0.50 จาก evaluator AP แต่ละจุดอยู่บน interpolated curve ส่วน **F1-confidence curve** ใช้ fixed matcher จาก saved predictions เพื่อแสดงการเปลี่ยน threshold ค่า threshold หลักยังคง 0.25 โดยไม่เลือกใหม่จากกราฟ

Day/night AP คำนวณแยกจากรายการภาพของแต่ละกลุ่ม เป็นผลเชิงพรรณนาตาม metadata มี day 243 ภาพและ night 357 ภาพ ข้อมูลแต่ละกลุ่มอาจต่างกันทั้งกล้อง จำนวนคนและความยาก AP รวม 600 ภาพเป็น pooled AP ต้องคำนวณจาก prediction ranking รวม การเฉลี่ย AP รายกลุ่มให้คำตอบคนละแบบ

Recall ตามด้านยาว GT ใช้ matching ของภาพเดิม แล้วนับ TP ของ GT ในช่วง 50–99, 100–199 และ ≥200 px แยกกัน GT ในสามช่วงมี 178, 342 และ 249 คน รายงานนี้ยังไม่ได้กำหนด Precision/AP รายช่วงขนาด เพราะการแจก FP ตาม GT size ต้องมีกติกาเพิ่ม

**Scaling deltas** ใช้ smaller − larger ในตระกูลเดียวกัน มี 13 คู่ติดกัน ตัวอย่าง AP ลดจาก 0.65 เป็น 0.60 มีส่วนต่าง −0.05 หรือ −5 จุดเปอร์เซ็นต์ ส่วนต้นทุนสัมพัทธ์คำนวณ (smaller−larger)/larger×100

**Near-tie** ใช้ส่วนต่าง AP ไม่เกิน 0.001 หรือ 0.1 จุดเปอร์เซ็นต์เพื่อคัดคู่มาอธิบาย เป็นเกณฑ์เชิงพรรณนา การยืนยัน equivalence หรือ statistical significance ต้องมีการวิเคราะห์ uncertainty เพิ่ม

**Pareto frontier** แยก AP–forward, AP–adapter และ AP–allocated VRAM โมเดลถูก dominate เมื่อมีอีกตัว AP ไม่น้อยกว่าและต้นทุนไม่มากกว่า โดยดีกว่าอย่างน้อยหนึ่งแกน คำนวณจากสถิติเต็มก่อนปัดเศษ ไม่มี weighted overall score ใหม่

## 6. เวลาแต่ละส่วนวัดอะไร

ทุกค่าต่อภาพในงานนี้หมายถึงหนึ่ง full frame ที่ส่งเข้า detector หนึ่งครั้ง ระบบยังไม่ได้ทำ human parsing จึงไม่มีหน่วย ms/ROI ของ parser ในตารางนี้

| Timer | จุดเริ่มถึงจุดจบ | หน่วยและการใช้ |
| --- | --- | --- |
| Input read/decode | อ่าน input ตรวจ hash และ decode เป็นภาพที่ adapter รับ | ms/ภาพ ใน quality pass |
| Preprocess/H2D | ภาพ decoded → resize/letterbox/channel/tensor → tensor บน GPU พร้อม infer | ms/ภาพ synchronize ที่ boundary |
| Forward | tensor พร้อม infer → model forward จบและ synchronize | ms/ภาพ วัดทุก native output ที่เส้นทางนี้ต้องคำนวณ |
| Postprocess | หลัง forward → NMS/native masks, original-grid geometry, compact transfer และ prediction objects พร้อมใช้ | ms/ภาพ ภายใน adapter |
| Adapter total | preprocess + forward + postprocess | ms/ภาพ ขอบเขตเดียวกับ FPS ในรายงาน |
| RLE preparation | เตรียม lossless compact/RLE สำหรับบันทึกจาก prediction objects | ms/ภาพ แยกจาก adapter |
| Output write | serialize/compress และเขียน prediction packet แบบ atomic สำเร็จ | ms/ภาพ อยู่ใน receipt |
| Application | ก่อนอ่านภาพ → prediction packet เขียนสำเร็จ | ms/ภาพ ใน quality pass รวม cold/session effects |
| Model loading | ก่อนสร้าง adapter → model โหลดพร้อมและ synchronize | วินาที/รอบ clean timing |

Application timer สิ้นสุดก่อนคำนวณ packet SHA สำหรับ receipt, เขียน receipt, readback/audit, ledger, progress update, scorer และ gallery จึงต้องระบุขอบเขตนี้เมื่อใช้คำว่า pipeline เวลา clean adapter ก็ไม่รวม input I/O, RLE, packet writing, load/warmup, queue หรือ downtime

การสกัด source timers ทำใน [yolo_runtime.py](src/cctv600/yolo_runtime.py) ส่วนขอบเขต read/write/application อยู่ใน [engine.py](src/cctv600/engine.py) จึงตรวจคำอธิบายกลับไปยังจุดจับเวลาจริงได้

### วิธีสรุปเวลา

Dedicated timing ใช้ชุด 100 ภาพและลำดับเดียวกัน 3 clean rounds มี warmup 10 calls แยกทุกรอบ รวม measured observations 300 รายการต่อโมเดล ค่า mean คือผลรวม/300 ค่า median และ P95 คำนวณจาก observations ทั้ง 300 โดย NumPy percentile แบบ linear ส่วน std ใช้ population standard deviation ไม่เฉลี่ย percentile ของแต่ละรอบ

FPS = 1000 / mean adapter ms ตัวอย่าง mean 50 ms/ภาพได้ 20 FPS ค่านี้เป็น throughput ที่แปลงจากขอบเขต adapter โดยไม่มี input/output I/O มิได้วัดการเปิดเว็บจนบันทึกฐานข้อมูลเสร็จ

Timing หลักต้องผ่านเกณฑ์ไม่มี external GPU compute PID ตาม GPU samples และ utilization ก่อนโหลดไม่เกิน 5% หลัง settle wait ถ้ารอบไม่ผ่านเก็บเป็น diagnostics แล้วใช้ fresh controlled segment ที่ตรวจ compatibility ตาม protocol เก็บประวัติเดิมทุกครั้ง การสุ่มตรวจ GPU ทุก 10 ภาพยังมีความละเอียดจำกัด จึงมี timestamps/resource observations กำกับผล

`ACTIVE_600_TIMING.csv` ให้ผลรวมเวลาของ 600 quality frames และค่าเฉลี่ยต่อภาพของแต่ละ stage รวม overhead/cold effects ตามขอบเขต timer ผลรวมช่วงทำงานไม่รวมช่วงพักระหว่างภาพหรือระหว่าง attempts จึงใช้แทน elapsed wall time ตั้งแต่ launch จนจบทั้งรอบไม่ได้ Application เป็น inclusive timer ห้ามบวก application กับ stage ภายในซ้ำ

### รายละเอียด postprocessing

Timer ย่อยที่ runtime เก็บมี Ultralytics postprocess, binary mask validation, mask bit packing, packed mask GPU→CPU, mask NumPy view, bbox GPU→CPU, bbox NumPy view และ prediction objects/bbox extraction รายละเอียดอยู่ใน TIMING_MASTER.csv ทุกช่องลงท้าย `_ms` ใช้หน่วย ms แม้บาง timer เป็น CPU work

Timer ย่อยเหล่านี้อยู่ภายใน postprocess ของ adapter การบวก substage เหล่านี้เข้ากับ postprocess อีกครั้งจะนับงานซ้ำ บางช่วงมี overhead ที่ไม่ได้แยก จึงใช้ adapter total ที่วัดตรงเป็นเวลารวม

## 7. VRAM และความซับซ้อน

| ค่า | ความหมาย |
| --- | --- |
| Allocated peak MiB | ค่าสูงสุดของ PyTorch allocated memory หลัง reset peak เมื่อจบ warmup ใช้ max ของสาม clean rounds รวม resident model |
| Reserved peak MiB | หน่วยความจำที่ PyTorch allocator จองไว้สูงสุด ขอบเขตเดียวกับ allocated |
| nvidia-smi memory | หน่วยความจำระดับ GPU process รวม CUDA context และ overhead ที่อาจอยู่นอก PyTorch counters |
| Parameters | จำนวนพารามิเตอร์ checkpoint ก่อน fusion เก็บเป็น integer |
| GFLOPs estimate | Static estimate จาก checkpoint hash เดียวกัน input 640 และ NMS route เดียวกับงาน MOTS20 |
| Checkpoint MB | ขนาดไฟล์ weights หาร 1,000,000 |

MiB ใช้ 1,048,576 bytes VRAM จาก benchmark นี้ยังไม่รวมบริการเว็บหลายโมเดล/หลายงานที่รันพร้อมกัน GFLOPs เป็นค่าประมาณตามวิธี profiling เดิม จำนวนพารามิเตอร์หรือ GFLOPs อย่างเดียวใช้คาดเวลา postprocessing ทั้ง pipeline ไม่ได้

## 8. ตารางเครื่องอ่านและแหล่งตัวเลข

| ไฟล์ | แถว/หน่วย/แหล่ง |
| --- | --- |
| MASTER_17_MODELS.csv | หนึ่งแถวต่อ checkpoint ตาม roster คะแนน fraction, เวลาหลัก ms/ภาพ, VRAM MiB จาก canonical metrics และ clean timing |
| RESULT_INDEX.json | แหล่ง canonical quality, hashes, membership, timing rounds ที่ใช้รวม และประวัติ round เดิม |
| TIMING_MASTER.csv | หนึ่งแถวต่อ model/stage มี observations, mean, median, P95, std หน่วย `_ms` หรือ seconds สำหรับ model loading |
| ACTIVE_600_TIMING.csv | ผลรวมวินาที ค่าเฉลี่ย ms/ภาพ และ coverage ของ quality timers |
| MODEL_COMPLEXITY_MASTER.csv | params integer, GFLOPs estimate, checkpoint MB |
| PER_GROUP_RESULTS.csv | Day/night AP และ counts หรือ size-band recall ช่องที่นิยามไม่ได้เว้นว่าง |
| TIER_WINNERS.csv | เจ็ดเป้าหมายต่อ tier แยกทิศทาง max/min สำหรับคะแนน เวลา FPS และ allocated VRAM |
| SCALING_DELTAS.csv | 13 คู่ × 11 metrics ความต่าง smaller−larger คำนวณ Decimal จากสถิติเต็ม |
| NEAR_TIES.csv | คู่ที่ AP ต่างไม่เกิน 0.1 จุดเปอร์เซ็นต์ พร้อมทิศทาง B−A |
| PARETO_FRONTIER / PIPELINE / VRAM.csv | แกนต้นทุนตามชื่อ พร้อม comparison_models และ nondominated |
| PREFLIGHT_MAXDET.csv | ผลต่าง AP ของ cap 200 เทียบ 1000 บน fixed sample 100 ภาพ |
| CLOSEOUT.json | สถานะรวมหลัง reconciliation จำนวน calls และหลักฐานการปิดรอบ |

METRICS.json และ PACKET_AUDIT.json ใน outputs ของแต่ละโมเดลเป็นหลักฐานคุณภาพเดิม ส่วนตาราง public เป็น derived summaries เก็บ snapshots ก่อนปรับรูปแบบไว้ใน private closeout directory ช่อง null/ว่างยังคงความหมายว่าขาดหลักฐานหรือคำนวณไม่ได้

## 9. ที่มาของรูปแบบงาน

นำวิธีจัดกลุ่ม X/E, L/C, M, S, N, การแสดง AP/P/R/F1, scaling, complexity, timing composition และ Pareto มาต่อยอดจากงาน MOTS20 เดิม พร้อมเปลี่ยน metric/GT policy ให้ตรงกับ CCTV600 แหล่งเดิมอยู่ใน [MOTS20 Scaling Study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study)

ผลนี้รองรับการเปรียบเทียบ checkpoint/configuration บนข้อมูลที่รับรองอยู่ ยังไม่มี full-person GT mask สำหรับ Person Mask AP, track IDs สำหรับ tracking หรือ uncertainty analysis สำหรับคำอ้างนัยสำคัญ

## 10. ตารางประกอบการนำเสนอ

TIMING_COMPOSITION.csv คง stage means เดิมและเพิ่มสัดส่วน preprocess/forward/postprocess ต่อ adapter total ส่วน PLOT_READY_RESULTS.csv เพิ่ม family, display order และ Pareto flags โดยคะแนน/เวลาคงเดิม NEAR_TIE_RESOURCE_DELTAS.csv ใช้ B−A และหาร A สำหรับร้อยละสัมพัทธ์ ทิศทางนี้ต่างจาก SCALING_DELTAS.csv ที่ใช้ smaller−larger

TIER_WINNERS.csv มีเจ็ดเป้าหมายในแต่ละ tier ได้แก่ Box AP, AP75, Recall, forward time, adapter time, FPS และ allocated VRAM คะแนน/FPS ใช้ค่าสูงสุด ส่วนเวลา/VRAM ใช้ค่าต่ำสุด เก็บ exact ties ทุกตัวและเก็บค่าคะแนนต้นฉบับเป็น fraction
