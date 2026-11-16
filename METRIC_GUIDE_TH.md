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

<a id="protocol-comparison"></a>

## 11. Protocol ของ CCTV600 ใช้ร่วมกับ MOTS20 ส่วนใดบ้าง

ตรวจเอกสาร config, frozen inference source, evaluator และ accepted records เมื่อ 16 พฤศจิกายน 2026 การตรวจนี้ใช้หลักฐานเดิม และแก้คำอธิบายผล maxDet ที่เคยระบุว่าผลต่างเป็นศูนย์ทุกโมเดล

### 11.1 สิ่งที่ตรวจว่าใช้ร่วมกันจริง

เทียบ config ร่วม 18 fields ได้ค่าเท่ากัน รวม device, batch, input size, FP32, augmentation, NMS path, confidence, IoU thresholds, candidate cap, AP cap, warmup และจำนวน timing repeats ตรวจ checkpoint manifest SHA256 ตรงกันครบ 17 ตัว และ framework-source hashes ของ Ultralytics ที่ MOTS20 ตรึงไว้ตรงกันครบ 7 ไฟล์

ส่วน detector runtime ใช้อัลกอริทึมจาก frozen MOTS20 โดยต่างที่ package-relative imports และการบันทึกจำนวน parameters ก่อน fusion การเก็บ packet/RLE เป็นโค้ดจัดเก็บของ CCTV600 ส่วน evaluator ของ CCTV600 เขียนสำหรับ bbox โดยเฉพาะ แหล่งเปรียบเทียบอยู่ใน [runtime provenance](manifests/RUNTIME_ORIGIN.json), [MOTS20 baseline](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study/blob/main/provenance/BASELINE_REFERENCE.json), [MOTS20 mask evaluator](https://github.com/folklazy/YOLO_Large_Seg_MOTS20_Benchmark/blob/main/src/frozen_pilot/metrics.py) และ [CCTV600 bbox evaluator](src/cctv600/scoring.py)

| ส่วน | MOTS20 ที่รันจริง | CCTV600 ที่รันจริง | ผลต่อการอ่านคะแนน |
| --- | --- | --- | --- |
| Model roster | 17 segmentation checkpoints, X/E → L/C → M → S → N | Checkpoint manifest hashes ตรงกันทั้ง 17 ตัว | เปรียบเทียบ checkpoint เดิมข้ามข้อมูลได้เมื่อระบุ metric และ domain กำกับ |
| Input | BGR → aspect-preserving centered letterbox 640×640, padding 114 → RGB BCHW /255 | อัลกอริทึม runtime เดียวกัน | รับภาพเต็มเฟรมหนึ่งภาพต่อ call |
| Model execution | FP32, batch 1, ไม่มี extra TTA, Person class 0, YOLO26 one-to-many/NMS | ค่าเดียวกันใน config ที่ตรวจ | ควบคุมเส้นทาง inference ร่วมได้ |
| Thresholds | AP floor 0.001, fixed confidence 0.25, NMS box IoU 0.70 | ค่าเดียวกัน | Threshold confidence/NMS มีความหมายร่วมกัน |
| GT และประชากร | 2,862 frames, 26,894 Person mask instances | 600 images, 769 eligible visible person boxes | จำนวนเหล่านี้เป็น annotation รายภาพ บุคคลเดิมอาจปรากฏหลายภาพ |
| Eligibility | ใช้ valid Person masks ใน GT ที่ให้มา ไม่มีเงื่อนไขด้านยาว bbox 50 px ใน loader | max(width,height) ≥50 px จาก source visible bbox ก่อน clipping | คะแนน CCTV600 มีขอบเขตขนาดคนตามที่กำหนด |
| Fixed matching | Greedy 1–1 ที่ mask IoU ≥0.50 | Greedy 1–1 ที่ bbox IoU ≥0.50 | ค่า IoU 0.50 วัดรูปทรงคนละแบบ จึงตีความ P/R/F1 ตามชนิด matching |
| Ties | Prediction order เมื่อ score เท่ากัน, GT object ID เมื่อ IoU เท่ากัน | ใช้ bbox geometry ในการเรียง ties | กติกาตัดสินเสมอมีรายละเอียดต่างกัน |
| Ignore | Class-10 pixel-mask union, prediction-mask IoA ≥0.50 | Explicit ignore rectangles + bbox คนต่ำกว่า 50 px, prediction-box IoA ≥0.50 | ใช้ GT-first เหมือนกัน แต่พื้นที่ที่ได้รับยกเว้นต่างรูปแบบ |
| AP | COCO-style segmentation AP จาก mask IoUs | COCO-style bbox AP จาก box IoUs | ใช้ 10 IoU thresholds และ 101 recall points ร่วมกัน แต่ชนิด AP ต่างกัน |
| AP cap | เลือก 200 จาก sensitivity caps 100/200/300/1000 ก่อน full run | ตรึง 200 แล้วตรวจเทียบ 1000 บน 100 ภาพเดิมของแต่ละโมเดล | ผ่านเกณฑ์ผลต่าง <0.0001 ทุกโมเดล |
| Timing sample | 25 เฟรมต่อ sequence รวม 100 ภาพ, seed 20260929 | 100 ภาพจาก metadata round-robin, seed 20261116 | ทุกโมเดลในแต่ละ study ใช้ sample เดียวกัน แต่ sample ข้าม dataset ต่างกัน |
| Clean timing | Warmup 10 ต่อรอบ, 3×100 measured observations, synchronize CUDA | หลักการและขอบเขต adapter เดียวกัน | เปรียบเทียบภายใน CCTV600 ได้ตาม workload นี้ |
| FPS และ VRAM | FPS=1000/mean adapter ms, peak หลัง warmup รวม resident model | นิยามเดียวกัน | การเปรียบเทียบตัวเลขข้าม dataset ต้องคำนึงถึง image/mask workload |
| กลุ่มผลย่อย | MOTS20 sequences 02/05/09/11 | Day/night metadata และ GT size-band recall | กลุ่มเหล่านี้ใช้บรรยายข้อมูลตามที่มี |
| Outputs และ recovery | Run-scoped lossless masks, แยก contaminated timing และ clean repetitions | Atomic packets/receipts/ledger, exact-ID reconciliation และ linked timing segment | รูปแบบ orchestration และ accounting ต่างกัน |

### 11.2 ส่วนที่ GT ปัจจุบันยังรองรับไม่ถึง MOTS20

GT ของ CCTV600 ให้กรอบคนและ mask สี่ส่วนที่เลือก ส่วน Person Mask AP, person-mask IoU/Dice และการวัดขอบตัวคนตามแบบ MOTS20 ต้องมี silhouette GT ที่ตรวจรับครบรายคน การ union เฉพาะ Hair/Face/Upper/Lower ยังขาดส่วนอื่นของคน จึงใช้แทน GT คนทั้งตัวไม่ได้

Box AP รองรับการวัดตำแหน่งและความตรงของกรอบคนได้โดยตรง ส่วนการยืนยันว่าขอบ mask ตามแขน ขา หรือบริเวณคนซ้อนกันถูกต้องเพียงใด ต้องใช้ GT mask ที่ตรงกับคำถามนั้น คะแนนจาก union สี่ส่วนจะต้องประกาศชื่อ target และ valid domain เพิ่มเป็นอีกการประเมินหนึ่ง

งาน MOTS20 เดิมใน repo นี้เองใช้ frame-level segmentation metrics ตาม COCO-style evaluator การประเมิน official MOTS tracking ต้องมีวิธีเชื่อม track IDs และ tracking evaluator เพิ่ม ส่วน CCTV600 รอบนี้ประเมิน detector ตาม GT bbox ที่มีอยู่

### 11.3 Timing ใช้นิยามเดียวกัน แต่ workload ต่างกัน

Adapter total ในสองงานรวม preprocess/H2D + forward + native postprocessing/mask transfer และวัด RLE แยก Input disk/decode, packet writing, model loading, warmup, scoring และ queue time อยู่ภายนอก adapter FPS ส่วน CCTV600 เพิ่ม timers ของ input hash/read/decode, RLE, output write และ application span ระหว่าง quality pass

CCTV600 มีภาพ 1920×1080 จำนวน 594 ภาพ และอีกหกภาพที่ความละเอียดต่างออกไป รวมถึง 2688×1520 ส่วน MOTS20 มีทั้ง 1920×1080 และ 640×480 จำนวนคนและจำนวน masks ที่แต่ละโมเดลส่งออกก็ต่างกัน แม้ network input จะ 640 เท่ากัน การสร้าง masks กลับ original grid จึงมีต้นทุนต่างกัน

เกณฑ์ clean timing ของทั้งสองงานยกเว้นรอบที่พบ external GPU compute PID และต้องผ่าน preload idle utilization ≤5% มีการสุ่มตรวจระหว่างรันทุก 10 ภาพ MOTS20 เก็บ telemetry ด้าน driver/clocks/temperature/power มากกว่า CCTV600 ซึ่งบันทึก free VRAM/utilization/PIDs เป็นหลัก จึงยังขาดหลักฐานสำหรับรับรองสภาวะนาฬิกาและอุณหภูมิที่เหมือนกันทุกช่วง

CCTV600 ระบุ matmul/cudnn TF32 flags เป็น false อย่างชัดเจน ส่วน frozen MOTS20 runner ที่ตรวจไม่ได้กำหนดสอง flags นี้โดยตรง ทั้งสองงานบันทึก FP32 บน Tesla T4 ตาราง hardware ของ NVIDIA ระบุว่า T4 ไม่รองรับ TF32 การเทียบ flags ยังคงต้องระบุระดับหลักฐานตามที่บันทึกไว้ และการย้ายการทดลองไป GPU รุ่นอื่นต้องตรวจ effective settings ใหม่ [NVIDIA hardware/precision support](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/pdf/TensorRT-Support-Matrix-Guide.pdf)

### 11.4 Preflight, completeness และข้อกำหนดทรัพยากร

MOTS20 Largest เลือก common AP cap จากหลายค่าแล้วทำ full inference ครบประชากร ส่วน CCTV600 ใช้ predictions จาก preflight 100 ภาพเป็น accepted canonical observations และทำอีก 500 ภาพเมื่อ bindings ผ่าน รวม 600 quality calls ต่อโมเดล การนำกลับมาใช้ยึด input/checkpoint/config/source bindings โดยไม่มีการเลือกตามคะแนน

CCTV600 ใช้ STRICT_COMPLETE แยกรายโมเดล ต้องครบ exact 600 IDs พร้อม valid packets และคะแนนหลักเป็น null เมื่อไม่ครบ เวลาหลักต้องครบสาม clean rounds ตาม result index การมีไฟล์ครบยังต้องอ่านควบคู่กับข้อจำกัด GT inventory ซึ่งอ้างการตรวจรับของผู้จัดทำ dataset และยังไม่มี independent full-inventory adjudication

Free VRAM 4 GiB, free disk 20 GiB, output budget 20 GiB และเวลา 8 ชั่วโมงต่อ attempt เป็น operational guards ที่กำหนดใน CCTV600 ไม่ใช่ข้อกำหนดของนิยาม AP หรือคำรับรองว่าโมเดลต้องใช้ VRAM อย่างน้อย 4 GiB

### 11.5 Correction ของผลเทียบ AP cap

รายงานปิดรอบรุ่นก่อนเขียนว่าผลต่าง AP ทั้งสามค่าเป็นศูนย์ทุกโมเดล ตรวจ CAP_CHECK.json ที่ผูก hashes กับ accepted result index แล้วพบผลต่างไม่เป็นศูนย์ใน YOLOv8s, YOLO26n และ YOLOv8n

ผลต่าง absolute สูงสุดคือ **AP50 ของ YOLO26n = 0.000022855779105435126** หรือประมาณ **0.002286 จุดเปอร์เซ็นต์** ขณะที่เกณฑ์รับรองคือ <0.0001 หรือ <0.01 จุดเปอร์เซ็นต์ ทุกโมเดลจึงผ่านเกณฑ์ที่ตรึงไว้ แหล่งตัวเลขอยู่ใน [PREFLIGHT_MAXDET.csv](metrics/PREFLIGHT_MAXDET.csv) การแก้ครั้งนี้เป็น correction ของคำอธิบาย โดยใช้ผลคุณภาพและ timing ที่ตรวจรับแล้วเป็นหลักฐานเดิม

### 11.6 ขอบเขตการเปรียบเทียบที่รองรับ

ภายใน CCTV600 ทั้ง 17 โมเดลใช้ population, eligibility, bbox scorer, ignore policy, thresholds และ timing protocol ร่วมกัน ผลจึงใช้เปรียบเทียบ checkpoint/configuration ภายใต้ขอบเขตนี้ได้ ส่วน capacity/pretraining, ความถูกต้องของ GT inventory และ uncertainty ยังเป็นข้อจำกัดที่ต้องระบุ

การเทียบ MOTS20 กับ CCTV600 ใช้อธิบายแนวโน้มบนคนละ dataset ได้ โดยเขียนกำกับให้ครบว่า Mask AP หรือ Box AP, GT/eligibility แบบใด และ timing workload ใด คำอธิบายวิธีที่ตรงงานนี้คือ “นำ inference runtime และหลักการควบคุมการทดลองจาก MOTS20 มาปรับเป็น CCTV600 visible-bbox evaluation”
