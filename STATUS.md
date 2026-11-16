# สถานะ YOLO CCTV600

**RUNNING · snapshot 16/11/2026 14:42:22 +07**

Run `cctv600-20261116T073208561699Z`

Controller PID `2989946` · tmux `yolo_cctv600_073208`

คิว quality ทำงานตาม X/E → L/C → M → S → N แล้วต่อ dedicated clean timing ตาม protocol

CPU acceptance ผ่าน14กรณี และตรวจreadback/geometry/sourcebindingsของ3packetsแรกพร้อมเปิดดูภาพเทียบแล้ว

| Tier | Model | Accepted packets |
| --- | --- | --- |
| largest | yolo26x-seg.pt | 600/600 |
| largest | yolo11x-seg.pt | 270/600 |
| largest | yolov9e-seg.pt | 100/600 |
| largest | yolov8x-seg.pt | 100/600 |
| second_largest | yolo26l-seg.pt | 0/600 |
| second_largest | yolo11l-seg.pt | 0/600 |
| second_largest | yolov9c-seg.pt | 0/600 |
| second_largest | yolov8l-seg.pt | 0/600 |
| medium | yolo26m-seg.pt | 0/600 |
| medium | yolo11m-seg.pt | 0/600 |
| medium | yolov8m-seg.pt | 0/600 |
| small | yolo26s-seg.pt | 0/600 |
| small | yolo11s-seg.pt | 0/600 |
| small | yolov8s-seg.pt | 0/600 |
| nano | yolo26n-seg.pt | 0/600 |
| nano | yolo11n-seg.pt | 0/600 |
| nano | yolov8n-seg.pt | 0/600 |

จำนวนข้างต้นเป็น snapshot สถานะสดให้ใช้คำสั่งต่อไปนี้จาก workspace

```bash
.venv/bin/python YOLO_Instance_Segmentation_CCTV600_Scaling_Study/scripts/study.py status
```

ภาพจริง native packets และ logs เก็บ private ใน outputs ของแต่ละ experiment คะแนน/รายงานรวมสร้างเมื่อผ่านขั้นตรวจตาม protocol
