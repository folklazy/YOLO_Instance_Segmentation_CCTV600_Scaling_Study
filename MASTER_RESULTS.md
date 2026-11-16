# ผลรวม YOLO CCTV600

Accuracy complete 17/17 checkpoints.

| Model | Tier | Images | Box AP % | AP50 % | AP75 % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yolo26x-seg.pt | largest | 600/600 | 67.201 | 86.080 | 76.046 | 81.144 | 82.869 |
| yolo11x-seg.pt | largest | 600/600 | 64.917 | 84.329 | 73.520 | 76.333 | 81.189 |
| yolov9e-seg.pt | largest | 600/600 | 67.808 | 87.168 | 77.973 | 79.584 | 85.356 |
| yolov8x-seg.pt | largest | 600/600 | 63.481 | 83.016 | 72.864 | 77.763 | 79.840 |
| yolo26l-seg.pt | second_largest | 600/600 | 65.099 | 83.971 | 74.179 | 78.544 | 80.965 |
| yolo11l-seg.pt | second_largest | 600/600 | 63.696 | 84.760 | 72.222 | 78.674 | 83.391 |
| yolov9c-seg.pt | second_largest | 600/600 | 62.788 | 84.033 | 71.876 | 75.553 | 80.415 |
| yolov8l-seg.pt | second_largest | 600/600 | 62.978 | 82.235 | 72.434 | 76.723 | 78.824 |
| yolo26m-seg.pt | medium | 600/600 | 61.437 | 81.107 | 69.662 | 74.252 | 79.361 |
| yolo11m-seg.pt | medium | 600/600 | 60.436 | 80.994 | 67.273 | 73.212 | 79.858 |
| yolov8m-seg.pt | medium | 600/600 | 61.489 | 82.569 | 69.790 | 75.423 | 79.890 |
| yolo26s-seg.pt | small | 600/600 | 56.099 | 76.079 | 64.778 | 69.181 | 75.088 |
| yolo11s-seg.pt | small | 600/600 | 52.740 | 74.202 | 59.316 | 66.710 | 74.133 |
| yolov8s-seg.pt | small | 600/600 | 52.991 | 74.809 | 59.256 | 67.880 | 72.803 |
| yolo26n-seg.pt | nano | 600/600 | 40.817 | 61.914 | 45.176 | 51.105 | 61.695 |
| yolo11n-seg.pt | nano | 600/600 | 39.493 | 59.325 | 44.019 | 48.375 | 59.520 |
| yolov8n-seg.pt | nano | 600/600 | 39.903 | 61.634 | 44.693 | 52.796 | 63.240 |

คะแนน Box AP สูงสุดในรอบนี้คือ **yolov9e-seg.pt** 67.808%

ตารางเรียง roster order ค่าทั้งหมดมาจาก saved-packet scoring ความต่างคะแนนยังไม่มี uncertainty/significance analysis

อ่านวิธีและข้อจำกัดใน [study protocol](EXPERIMENT_PROTOCOL.md) เวลาจาก clean subsets และเวลางานจริง600ภาพมีขอบเขตแยกกัน
