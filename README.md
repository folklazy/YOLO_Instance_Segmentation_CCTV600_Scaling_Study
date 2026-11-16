# YOLO CCTV600 Scaling Study

ประเมิน pretrained segmentation checkpoints 17 ตัวบน CCTV 600 ภาพ แบ่งเป็น 5 กลุ่มขนาดและ 6 repositories ตามรูปแบบงาน MOTS20 คะแนนหลักใช้ Person Box AP บน visible bbox ของคนด้านยาวตั้งแต่ 50 px

## กลุ่มการทดสอบ

| กลุ่ม | Repository | Checkpoints |
| --- | --- | --- |
| largest | [YOLO_Large_Seg_CCTV600_Benchmark](https://github.com/folklazy/YOLO_Large_Seg_CCTV600_Benchmark) | yolo26x-seg.pt, yolo11x-seg.pt, yolov9e-seg.pt, yolov8x-seg.pt |
| second_largest | [YOLO_Second_Largest_Seg_CCTV600_Benchmark](https://github.com/folklazy/YOLO_Second_Largest_Seg_CCTV600_Benchmark) | yolo26l-seg.pt, yolo11l-seg.pt, yolov9c-seg.pt, yolov8l-seg.pt |
| medium | [YOLO_Medium_Seg_CCTV600_Benchmark](https://github.com/folklazy/YOLO_Medium_Seg_CCTV600_Benchmark) | yolo26m-seg.pt, yolo11m-seg.pt, yolov8m-seg.pt |
| small | [YOLO_Small_Seg_CCTV600_Benchmark](https://github.com/folklazy/YOLO_Small_Seg_CCTV600_Benchmark) | yolo26s-seg.pt, yolo11s-seg.pt, yolov8s-seg.pt |
| nano | [YOLO_Nano_Seg_CCTV600_Benchmark](https://github.com/folklazy/YOLO_Nano_Seg_CCTV600_Benchmark) | yolo26n-seg.pt, yolo11n-seg.pt, yolov8n-seg.pt |

## ผลตรวจปิดรอบ

คุณภาพครบ **17/17 checkpoints ตัวละ 600 ภาพ** และ dedicated timing ครบ **17/17 checkpoints ตัวละ 3 clean rounds** เติมเฉพาะ Nano 4 รอบใน fresh timing segment ตามเงื่อนไขเดิม แล้วเปิด worker เว็บคืน

- [รายงานหลักพร้อมตารางและกราฟ](MASTER_RESULTS.md)
- [บทสรุปสำหรับเลือกโมเดลและนำเสนอ](EXECUTIVE_SUMMARY_TH.md)
- [ตัวชี้วัด สูตร ขอบเขตเวลา และความหมายของตาราง](METRIC_GUIDE_TH.md)
- [สถานะรวมหลัง reconciliation](metrics/CLOSEOUT.json) และ [แหล่งผลพร้อม hashes](metrics/RESULT_INDEX.json)

คะแนนหลักเป็น **Person Box AP** บน CCTV600 ส่วนงาน MOTS20 เดิมมี GT และชนิด AP ต่างกัน รายงานนี้จัดรูปแบบการนำเสนอให้สอดคล้องกับงานเดิม โดยระบุขอบเขต metric ตาม GT ที่มี

## วิธีและผล

- [Protocol และข้อกำหนดคะแนน](EXPERIMENT_PROTOCOL.md)
- [สถานะ](STATUS.md) และ [ผลรวม](MASTER_RESULTS.md)
- [ตารางรวมเครื่องอ่าน](metrics/MASTER_17_MODELS.csv)
- [ข้อมูล600ภาพ](https://github.com/folklazy/CCTV-Human-Parsing-600)

## เปิดและดูสถานะจากworkspace

ใช้ environment เดิมที่ตรวจความเข้ากันได้แล้ว

```bash
.venv/bin/python YOLO_Instance_Segmentation_CCTV600_Scaling_Study/analysis/closeout.py --status
```

ดูอัปเดตต่อเนื่องได้ด้วย

```bash
watch -n 5 .venv/bin/python YOLO_Instance_Segmentation_CCTV600_Scaling_Study/analysis/closeout.py --status
```

คำสั่งข้างต้นอ่านสถานะปิดรอบที่รวม parent และ timing continuation แล้ว ส่วน `scripts/study.py status` อ่านสถานะของ original execution ซึ่งบันทึกก่อนการเติมเวลาไว้เป็นประวัติ

ภาพเปรียบเทียบจริง 16 ภาพอยู่ใน private `outputs/<run_id>/gallery/index.html` ของแต่ละ tier ดัชนีรวมอยู่ที่ `outputs/cctv600-20261116T073208561699Z/closeout/index.html` ของ Master

## คำสั่งสำหรับสร้างการทดลองรอบใหม่

เตรียมจากprivate600manifestที่รับรองแล้วและใช้freshoutput

```bash
.venv/bin/python YOLO_Instance_Segmentation_CCTV600_Scaling_Study/scripts/study.py prepare --manifest /private/path/manifest.csv
.venv/bin/python YOLO_Instance_Segmentation_CCTV600_Scaling_Study/scripts/study.py launch
```

`launch` ตรวจacceptance/sourcebindingsและprocessเดิมก่อนเปิดtmux หนึ่งcontrollerรันqualityตามX/E→L/C→M→S→Nแล้วทำdedicatedtiming มีpacketreadbackและsaved-packetrescore เมื่อจบสร้างรายงานและpushaggregatesอัตโนมัติ

เมื่อต้องหยุดหรือเกิดinterruption

```bash
.venv/bin/python YOLO_Instance_Segmentation_CCTV600_Scaling_Study/scripts/study.py stop
.venv/bin/python YOLO_Instance_Segmentation_CCTV600_Scaling_Study/scripts/study.py reconcile
```

`stop` ให้หยุดที่ขอบภาพถัดไป `reconcile` ตรวจแบบread-only หลังworkerหยุด ตรวจevidenceก่อนexplicit`launch --resume`และเก็บoldattemptsไว้ STOP_REQUESTที่ผู้ใช้สร้างต้องได้รับการacknowledgeก่อนcontinuation รอบเวลาที่ล้ม/ปนเปื้อนต้องมีfreshcontrolledsegment

## สร้าง derived reports จากผลที่ตรวจรับแล้ว

```bash
.venv/bin/python YOLO_Instance_Segmentation_CCTV600_Scaling_Study/analysis/closeout.py
```

คำสั่งนี้ตรวจ bindings และผลเดิม จัดตาราง/กราฟใหม่ เก็บ snapshot เอกสารเดิมใน private outputs และไม่มี model calls ส่วน `analysis/timing_continuation.py` เป็นตัวควบคุมเฉพาะการเติมเวลา 4 รอบครั้งนี้ ต้องมี manifest ที่ตรวจแล้วและหยุด worker เว็บตามการอนุมัติ ใช้ original frozen timer และคืน worker ใน finally รอบนี้เสร็จแล้ว

## ตรวจscorerบนCPU

```bash
.venv/bin/python -B -m unittest discover -s YOLO_Instance_Segmentation_CCTV600_Scaling_Study/tests -v
```

AssertionsมีindependentexpectedAP/counts,ignoreunion,50px,geometry,RLE,immutablepacketและnullhandling รายงานCPUacceptanceผูกกับsourcehashesที่ใช้งานจริง

รายละเอียดภาพจริงและmanifestรายภาพอยู่private ส่วนrepositoryเก็บcode/configs/docs,checkpoint/sourcebindingsและผลรวมที่ตรวจรับ
