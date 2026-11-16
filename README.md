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

## วิธีและผล

- [Protocol และข้อกำหนดคะแนน](EXPERIMENT_PROTOCOL.md)
- [สถานะ](STATUS.md) และ [ผลรวม](MASTER_RESULTS.md)
- [ตารางรวมเครื่องอ่าน](metrics/MASTER_17_MODELS.csv)
- [ข้อมูล600ภาพ](https://github.com/folklazy/CCTV-Human-Parsing-600)

## เปิดและดูสถานะจากworkspace

ใช้ environment เดิมที่ตรวจความเข้ากันได้แล้ว

```bash
.venv/bin/python YOLO_Instance_Segmentation_CCTV600_Scaling_Study/scripts/study.py status
```

ดูอัปเดตต่อเนื่องได้ด้วย

```bash
watch -n 5 .venv/bin/python YOLO_Instance_Segmentation_CCTV600_Scaling_Study/scripts/study.py status
```

## เริ่มหรือดำเนินงาน

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

## ตรวจscorerบนCPU

```bash
.venv/bin/python -B -m unittest discover -s YOLO_Instance_Segmentation_CCTV600_Scaling_Study/tests -v
```

AssertionsมีindependentexpectedAP/counts,ignoreunion,50px,geometry,RLE,immutablepacketและnullhandling รายงานCPUacceptanceผูกกับsourcehashesที่ใช้งานจริง

รายละเอียดภาพจริงและmanifestรายภาพอยู่private ส่วนrepositoryเก็บcode/configs/docs,checkpoint/sourcebindingsและผลรวมที่ตรวจรับ
