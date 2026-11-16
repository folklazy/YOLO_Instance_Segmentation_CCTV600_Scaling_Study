# ข้อเสนอการเลือก YOLO จากผล CCTV600

ครบ 17 checkpoints ตัวละ 600 ภาพ GT เข้าเกณฑ์ 769 คน พร้อม 3 clean timing rounds ต่อโมเดล ใช้ pretrained, input 640, batch 1, FP32 บน Tesla T4

| เป้าหมาย | ตัวเลือก | หลักฐานและข้อแลกเปลี่ยน |
| --- | --- | --- |
| คุณภาพกรอบและ F1 | yolov9e-seg.pt | AP 67.808%, F1 85.356%, FP 53, FN 157 |
| เก็บคนให้ได้มากที่ confidence เดิม | yolo26x-seg.pt | Recall 81.144%, FP 113, FN 145 |
| Forward latency | yolov8n-seg.pt | 9.923 ms/ภาพ แต่ adapter 79.317 ms/ภาพ |
| Adapter latency | yolo26m-seg.pt | 56.126 ms/ภาพ, AP 61.437% |
| Allocated VRAM | yolov9c-seg.pt | 506.215 MiB, AP 62.788% |
| AP กับ adapter budget | yolo26l-seg.pt | AP 65.099%, adapter 59.770 ms/ภาพ อยู่บน AP–adapter Pareto |

การให้โมเดลตรวจคนส่งเข้า parsing ต้องพิจารณาว่ายอมรับ FN/FP เท่าไร การชนะ AP, Recall และเวลาตอบคำถามคนละด้าน ยังไม่มี weighted score ที่กำหนดผู้ชนะทุกข้อจำกัด

คะแนนนี้วัด visible bbox ≥50 px ของ GT ที่ผู้จัดทำ dataset ตรวจรับ Person Mask AP ยังต้องมี GT silhouette เพิ่ม ยังไม่มี uncertainty analysis หรือข้อรับรอง deployment ค่า timing ครอบคลุม adapter และมี I/O/RLE/load/warmup แยกออกไป

[ผลเต็ม](MASTER_RESULTS.md) · [วิเคราะห์ข้ามขนาด](RESEARCH_INSIGHTS_TH.md) · [เนื้อหานำเสนอ](MEETING_SUMMARY_TH.md) · [สูตร metric](METRIC_GUIDE_TH.md)
