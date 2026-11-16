"""Canonical bbox results, completeness, private diagnostics and public reports."""
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from .common import STUDY, WORKSPACE, load, now, sha, write
from .packets import read_packet
from .scoring import aggregate, fixed, ratios


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def model_directory(prepared, tier, model):
    return WORKSPACE / tier["repository"] / "outputs" / prepared["run_id"] / model.removesuffix(".pt")


def model_binding(prepared, model):
    return {"scientific_binding": prepared["scientific_binding"],
            "checkpoint_sha256": prepared["checkpoints"][model]["sha256"], "model": model}


def score_model(prepared, tier, model):
    directory = model_directory(prepared, tier, model)
    frames = prepared["frames"]
    expected = {frame["image_id"] for frame in frames}
    files = {path.name.removesuffix(".json.gz"): path for path in (directory / "predictions").glob("*.json.gz")}
    membership = {"expected": len(expected), "completed": len(expected & files.keys()),
                  "missing": len(expected - files.keys()), "unexpected": len(files.keys() - expected),
                  "duplicate": 0, "invalid": 0}
    result = {"model": model, "run_id": prepared["run_id"], "at_utc": now(),
              "binding": model_binding(prepared, model), "membership": membership, "primary": None}
    if membership["missing"] or membership["unexpected"]:
        result["state"] = "INCOMPLETE"
        write(directory / "METRICS.json", result)
        return result
    records = []
    packet_hashes = {}
    try:
        for frame in frames:
            path = files[frame["image_id"]]
            records.append(read_packet(path, frame, model_binding(prepared, model)))
            packet_hashes[frame["image_id"]] = sha(path)
    except Exception as error:
        membership["invalid"] += 1
        result.update(state="INVALID_ARTIFACT", error=str(error))
        write(directory / "METRICS.json", result)
        raise
    primary, details = aggregate(frames, [r["predictions"] for r in records], prepared["config"])
    if primary["tp"] + primary["fn"] != primary["gt_people"]:
        raise ValueError("Fixed-metric denominator mismatch")
    result.update(state="COMPLETE", primary=primary)
    floor_events = [event for frame, record in zip(frames, records)
        for event in fixed(frame, record["predictions"], prepared["config"]["ap_confidence_floor"],
            prepared["config"]["matching_iou"], prepared["config"]["ignore_prediction_ioa"])["events"]]
    confidence_curve = []
    thresholds = {round(float(value), 3) for value in np.linspace(.05, 1, 20)}
    thresholds.update({prepared["config"]["ap_confidence_floor"], .25})
    for threshold in sorted(thresholds):
        accepted = [item for item in floor_events if item["confidence"] >= threshold]
        tp = sum(item["disposition"] == "TP" for item in accepted)
        fp = sum(item["disposition"] == "FP" for item in accepted)
        confidence_curve.append({"confidence": threshold, "tp": tp, "fp": fp,
            "fn": primary["gt_people"] - tp, **ratios(tp, fp, primary["gt_people"] - tp)})
    result["confidence_curve"] = confidence_curve
    timing_keys = ["input_read_decode_ms", "preprocess_ms", "inference_ms", "postprocess_ms",
                   "total_ms", "rle_preparation_ms", "output_write_ms", "application_ms"]
    result["active_timing"] = {}
    for key in timing_keys:
        values = [r["timing"].get(key) for r in records]
        complete_values = [value for value in values if value is not None]
        result["active_timing"][key + "_seconds"] = sum(complete_values) / 1000 if len(complete_values) == len(records) else None
        result["active_timing"][key + "_coverage"] = len(complete_values)
    result["active_timing"]["scope"] = "600 accepted packet work spans across attempts; includes cold/session effects"
    result["active_timing"]["frames"] = len(records)
    public_groups, per_frame = [], []
    for frame, record, detail in zip(frames, records, details):
        per_frame.append({"image_id": frame["image_id"], "gt_people": len(frame["people"]),
            "tp": detail["tp"], "fp": detail["fp"], "fn": detail["fn"],
            "ignored": detail["ignored_predictions"], "post_nms": record["post_nms_candidates"],
            "retained": len(record["predictions"]), **record["timing"]})
    for group in sorted({frame["diurnal"] for frame in frames}):
        indexes = [i for i, frame in enumerate(frames) if frame["diurnal"] == group]
        metrics, _ = aggregate([frames[i] for i in indexes], [records[i]["predictions"] for i in indexes], prepared["config"])
        public_groups.append({"group_type": "metadata_day_night", "group": group, **metrics})
    for band in ["50to99", "100to199", "200plus"]:
        gt = tp = 0
        for frame, detail in zip(frames, details):
            gt += sum(person["size_band"] == band for person in frame["people"])
            tp += sum(frame["people"][m["gt_index"]]["size_band"] == band for m in detail["matches"])
        public_groups.append({"group_type": "gt_long_side_recall", "group": band,
            "images": None, "gt_people": gt, "tp": tp, "fp": None, "fn": gt - tp,
            "ignored_predictions": None, "precision": None, "recall": tp / gt if gt else None,
            "f1": None, "matched_box_iou_mean": None, "box_ap50_95": None,
            "box_ap50": None, "box_ap75": None, "box_ar_maxdet": None})
    result["groups"] = public_groups
    csv_write(directory / "PER_IMAGE.csv", per_frame)
    write(directory / "PACKET_AUDIT.json", {"state": "PASS", "binding": result["binding"],
        "packet_sha256": packet_hashes, "membership": membership})
    write(directory / "METRICS.json", result)
    return result


def statistics(values):
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {key: None for key in ["mean", "median", "p95", "std"]}
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "std": float(array.std())}


def timing_summary(directory):
    rounds = [load(path) for path in sorted((directory / "timing").glob("round*.json"))]
    accepted = [r for r in rounds if r["state"] == "CLEAN_COMPLETE"]
    complete = len(accepted) == 3 and all(len(r["observations"]) == 100 for r in accepted)
    summary = {"clean_rounds": len(accepted), "expected_rounds": 3,
               "measured_images": sum(len(r["observations"]) for r in accepted),
               "state": "COMPLETE" if complete else "INCOMPLETE", "primary": None}
    if complete:
        rows = [row for item in accepted for row in item["observations"]]
        primary = {key: statistics([row[key] for row in rows]) for key in
                   ["preprocess_ms", "inference_ms", "postprocess_ms", "total_ms", "rle_preparation_ms"]}
        primary["fps"] = 1000 / primary["total_ms"]["mean"]
        for key in ["peak_gpu_allocated_mib", "peak_gpu_reserved_mib"]:
            primary[key] = max(r["memory"][key] for r in accepted)
        primary["model_loading_seconds"] = statistics([r["load_seconds"] for r in accepted])
        primary["parameters"] = accepted[0]["parameters"]
        summary["primary"] = primary
    return summary


def display(value, multiplier=1., digits=3):
    return "—" if value is None else f"{value * multiplier:.{digits}f}"


def write_reports(prepared):
    overall = []
    for tier in prepared["config"]["tiers"]:
        repository = WORKSPACE / tier["repository"]
        rows, public, times = [], [], []
        for model in tier["models"]:
            directory = model_directory(prepared, tier, model)
            result = load(directory / "METRICS.json") if (directory / "METRICS.json").exists() else None
            primary = result["primary"] if result else None
            timing = timing_summary(directory)
            packet_count = len(list((directory / "predictions").glob("*.json.gz")))
            model_metadata = sorted(directory.glob("MODEL_*.json"))
            parameters = load(model_metadata[0])["checkpoint_parameters"] if model_metadata else None
            row = {"model": model, "state": result["state"] if result else "PARTIAL" if packet_count else "NOT_RUN",
                   "images": result["membership"]["completed"] if result else packet_count,
                   **{key: primary[key] if primary else None for key in
                      ["box_ap50_95", "box_ap50", "box_ap75", "precision", "recall", "f1", "tp", "fp", "fn"]},
                   "timing_state": timing["state"], "clean_rounds": timing["clean_rounds"],
                   "inference_mean_ms": timing["primary"]["inference_ms"]["mean"] if timing["primary"] else None,
                   "adapter_mean_ms": timing["primary"]["total_ms"]["mean"] if timing["primary"] else None,
                   "fps": timing["primary"]["fps"] if timing["primary"] else None,
                   "vram_allocated_mib": timing["primary"]["peak_gpu_allocated_mib"] if timing["primary"] else None,
                   "checkpoint_mb": prepared["checkpoints"][model]["bytes"] / 1e6,
                   "parameters": parameters,
                   "gflops_estimate": prepared["checkpoints"][model].get("reference_gflops")}
            rows.append(row)
            overall.append({"tier": tier["id"], **row})
            public.append({"model": model, "membership": result["membership"] if result else None,
                "primary": primary, "active_timing": result.get("active_timing") if result else None,
                "groups": result.get("groups") if result else None,
                "confidence_curve": result.get("confidence_curve") if result else None,
                "checkpoint_sha256": prepared["checkpoints"][model]["sha256"],
                "scientific_binding": prepared["scientific_binding"], "timing": timing})
        write(repository / "metrics/RESULTS.json", {"run_id": prepared["run_id"], "models": public})
        csv_write(repository / "metrics/TIER_RESULTS.csv", rows)
        lines = ["# ผลประเมิน YOLO บน CCTV600", "", "Run `" + prepared["run_id"] + "`", "",
            "คะแนนหลักเป็น Person Box AP ของ segmentation checkpoints บน visible GT bbox เกณฑ์คนด้านยาว ≥50 px", "",
            "| Model | Complete | Box AP % | AP50 % | AP75 % | P % | R % | F1 % |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            lines.append("| " + row["model"] + " | " + str(row["images"]) + "/600 | " + " | ".join(
                display(row[key], 100) for key in ["box_ap50_95", "box_ap50", "box_ap75", "precision", "recall", "f1"]) + " |")
        lines += ["", "| Model | Clean rounds | Forward ms/image | Adapter ms/image | FPS | Allocated MiB |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            lines.append("| " + row["model"] + " | " + str(row["clean_rounds"]) + "/3 | " + " | ".join(
                display(row[key]) for key in ["inference_mean_ms", "adapter_mean_ms", "fps", "vram_allocated_mib"]) + " |")
        lines += ["", "Adapter time รวม preprocessing/H2D, forward, NMS/native masks และ compact transfer ส่วน input I/O, RLE และ output writing เก็บแยก", "",
            "P/R/F1 ใช้confidence≥0.25 และbboxIoU≥0.50 ส่วนAPใช้IoU0.50:0.05:0.95 และmaxDet200 โดยเก็บpredictionsเหนือconfidencefloor0.001", "",
            "Timing หลักแสดงเมื่อมีสาม clean rounds ครบ งาน GPU อื่นทำให้รอบเวลาเป็น diagnostics", "",
            "GT person inventory อ้างคำยืนยันตรวจ600ภาพตามกติกาทุกคน≥50px ข้อมูลบางส่วนเคยใช้พัฒนาannotation helper ส่วนผู้เข้าสอบใช้pretrained checkpointที่ตรึง", "",
            "ภาพจริงและ per-image diagnostics เก็บ private ใน outputs ของ tier ส่วน results JSON/CSV เก็บสถิติรวม", "",
            "ผลนี้เป็น checkpoint+application configuration comparison ของกรอบคน และยังไม่มีผล Person Mask AP หรือ tracking", ""]
        (repository / "RESULTS_SUMMARY_TH.md").write_text("\n".join(lines))
        completed = sum(row["state"] == "COMPLETE" for row in rows)
        (repository / "STATUS.md").write_text(f"# สถานะ {tier['repository']}\n\nRun `{prepared['run_id']}`\n\nAccuracy complete {completed}/{len(rows)} models.\n\nอ่าน [ผลรวม](RESULTS_SUMMARY_TH.md) และ [วิธีดำเนินงาน](EXPERIMENT_PROTOCOL.md)\n")
    csv_write(STUDY / "metrics/MASTER_17_MODELS.csv", overall)
    write(STUDY / "metrics/MASTER_RESULTS.json", {"run_id": prepared["run_id"],
        "scientific_binding": prepared["scientific_binding"], "models": overall})
    complete = sum(row["state"] == "COMPLETE" for row in overall)
    text = ["# ผลรวม YOLO CCTV600", "", f"Accuracy complete {complete}/17 checkpoints.", "",
        "| Model | Tier | Images | Box AP % | AP50 % | AP75 % | Recall % | F1 % |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in overall:
        text.append("| " + row["model"] + " | " + row["tier"] + " | " + str(row["images"]) + "/600 | " +
            " | ".join(display(row[key], 100) for key in ["box_ap50_95", "box_ap50", "box_ap75", "recall", "f1"]) + " |")
    if complete == 17:
        winner = max(overall, key=lambda row: row["box_ap50_95"])
        text += ["", "คะแนน Box AP สูงสุดในรอบนี้คือ **" + winner["model"] + "** " + display(winner["box_ap50_95"], 100) + "%", "",
                 "ตารางเรียง roster order ค่าทั้งหมดมาจาก saved-packet scoring ความต่างคะแนนยังไม่มี uncertainty/significance analysis"]
    text += ["", "อ่านวิธีและข้อจำกัดใน [study protocol](EXPERIMENT_PROTOCOL.md) เวลาจาก clean subsets และเวลางานจริง600ภาพมีขอบเขตแยกกัน", ""]
    (STUDY / "MASTER_RESULTS.md").write_text("\n".join(text))
    (STUDY / "STATUS.md").write_text(f"# สถานะ YOLO CCTV600\n\nRun `{prepared['run_id']}`\n\nAccuracy complete {complete}/17 checkpoints.\n\nสถานะcontrollerจริงดูด้วยคำสั่งในREADME ซึ่งตรวจprocess identityและartifactsปัจจุบัน\n\nอ่าน [ผลรวม](MASTER_RESULTS.md)\n")
    return overall
