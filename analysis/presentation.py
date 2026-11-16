"""MOTS20-style documentation from completed CCTV600 evidence, CPU only.

Scientific source/configs, accepted measurements and original MOTS20 files are
read-only. Raw images, per-frame matching and rendered cases remain private.
"""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

STUDY = Path(__file__).resolve().parents[1]
WORKSPACE = STUDY.parent
sys.path.insert(0, str(STUDY / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pycocotools import mask as coco_mask

from cctv600.common import load, now, sha, verify_source, write
from cctv600.packets import read_packet
from cctv600.reporting import model_binding, model_directory
from cctv600.scoring import fixed, box_iou

MASTER_URL = "https://github.com/folklazy/" + STUDY.name
MASTER_DOC = MASTER_URL + "/blob/main/"
COLORS = {"YOLO26": "#0072B2", "YOLO11": "#D55E00", "YOLOv9": "#CC79A7", "YOLOv8": "#009E73"}
TITLES = {"largest": "Largest (X/E)", "second_largest": "Second-largest (L/C)",
    "medium": "Medium (M)", "small": "Small (S)", "nano": "Nano (N)"}
OBJECTIVES = [("box_ap50_95", "Box AP", "max", "%"), ("box_ap75", "AP75", "max", "%"),
    ("recall", "Recall", "max", "%"), ("inference_mean_ms", "Forward time", "min", "ms/ภาพ"),
    ("adapter_mean_ms", "Adapter time", "min", "ms/ภาพ"), ("fps", "Adapter FPS", "max", "ภาพ/วินาที"),
    ("vram_allocated_mib", "Allocated VRAM", "min", "MiB")]
CASE_SPECS = [
    (2, "A", "คนถูกตัดด้วยขอบภาพ", "ตรวจการคืนพิกัดและ visible bbox เมื่อเห็นคนเพียงบางส่วน"),
    (12, "B", "คนอยู่หลังโครงสร้างบัง", "ตรวจ GT หลายคนและกรอบส่วนเกินในฉากที่มีโครงสร้างพาดผ่าน"),
    (14, "C", "คนหลายขนาดร่วมกับ ignore", "ตรวจคนขนาดเล็กและความต่างระหว่าง FN, FP กับ ignored prediction"),
    (15, "D", "ภาพไม่มี eligible GT", "ตรวจ negative control ภายใต้ inventory ที่รับรอง")]


def family(model):
    return next(name for name in COLORS if model.lower().startswith(name.lower()))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
        ["| " + " | ".join(str(item) for item in row) + " |" for row in rows])


def document(path, content):
    path.write_text(content.strip() + "\n", encoding="utf-8")


def fmt(row, key):
    if row[key] is None:
        return "—"
    if key in ["box_ap50_95", "box_ap50", "box_ap75", "precision", "recall", "f1"]:
        return f"{row[key] * 100:.3f}"
    return f"{row[key]:.3f}"


def winner_rows(rows):
    result = []
    for key, label, direction, unit in OBJECTIVES:
        value = (max if direction == "max" else min)(row[key] for row in rows)
        result.extend({"tier": row["tier"], "metric": key, "direction": direction,
            "model": row["model"], "value": value, "unit": "fraction" if unit == "%" else unit}
            for row in rows if row[key] == value)
    return result


def winner_table(rows):
    winners = winner_rows(rows)
    return markdown_table(["ด้าน", "โมเดล", "ค่าที่วัด"], [
        [label, item["model"], fmt(next(row for row in rows if row["model"] == item["model"]), key) + " " + unit]
        for key, label, _, unit in OBJECTIVES for item in winners if item["metric"] == key])


def results_table(rows):
    keys = ["box_ap50_95", "box_ap75", "recall", "inference_mean_ms", "adapter_mean_ms", "fps", "vram_allocated_mib"]
    return markdown_table(["Model", "Box AP %", "AP75 %", "Recall %", "Forward ms/ภาพ", "Adapter ms/ภาพ", "FPS", "Allocated MiB"],
        [[row["model"]] + [fmt(row, key) for key in keys] for row in rows])


def navigation(tiers):
    return " · ".join([f"[{TITLES[t['id']]}](https://github.com/folklazy/{t['repository']})" for t in tiers] + [f"[Master]({MASTER_URL})"])


def protected_files(prepared, run):
    paths = [STUDY / "metrics" / name for name in ["MASTER_17_MODELS.csv", "MASTER_RESULTS.json", "RESULT_INDEX.json",
        "CLOSEOUT.json", "TIMING_MASTER.csv", "ACTIVE_600_TIMING.csv", "MODEL_COMPLEXITY_MASTER.csv",
        "SCALING_DELTAS.csv", "PER_GROUP_RESULTS.csv", "NEAR_TIES.csv", "PARETO_FRONTIER.csv", "PARETO_PIPELINE.csv", "PARETO_VRAM.csv"]]
    paths.extend(run / name for name in ["PREPARED.json", "COMPLETION.json", "STATUS.json"])
    for tier in prepared["config"]["tiers"]:
        repo = WORKSPACE / tier["repository"]
        paths.extend(repo / "metrics" / name for name in ["RESULTS.json", "TIER_RESULTS.csv", "TIMING_SUMMARY.csv"])
        for model in tier["models"]:
            directory = model_directory(prepared, tier, model)
            paths.extend(directory / name for name in ["METRICS.json", "PACKET_AUDIT.json", "CAP_CHECK.json"])
    return {str(path.relative_to(WORKSPACE)): sha(path) for path in paths}


def reference_audit():
    references = []
    roots = [WORKSPACE / "YOLO_Instance_Segmentation_MOTS20_Scaling_Study"]
    roots.extend(WORKSPACE / name for name in ["YOLO_Large_Seg_MOTS20_Benchmark", "YOLO_Second_Largest_Seg_MOTS20_Benchmark",
        "YOLO_Medium_Seg_MOTS20_Benchmark", "YOLO_Small_Seg_MOTS20_Benchmark", "YOLO_Nano_Seg_MOTS20_Benchmark"])
    for root in roots:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        for name in ["README.md", "MASTER_RESULTS.md", "STUDY_STANDARD.md", "DOCUMENTATION_LANGUAGE_POLICY.md",
            "RESEARCH_INSIGHTS_TH.md", "MEETING_SUMMARY_TH.md", "METRIC_GUIDE_TH.md", "DATA_SCHEMA.md", "METHODOLOGY_REFERENCE.md",
            "templates/TIER_REPORT_TEMPLATE.md", "templates/TIER_RESULTS_SUMMARY_TH_TEMPLATE.md", "templates/TIER_PRESENTATION_SUMMARY_TH_TEMPLATE.md", "scripts/plot_master.py",
            "RESULTS_SUMMARY_TH.md", "REPORT.md", "PRESENTATION_SUMMARY_TH.md"]:
            path = root / name
            if path.exists():
                text = path.read_text()
                references.append({"repository": root.name, "commit": commit, "file": name, "sha256": sha(path),
                    "sections": [line for line in text.splitlines() if line.startswith("#")]})
    return references


def remote_reference_checks():
    import hashlib
    import urllib.request
    checked = []
    samples = [("YOLO_Instance_Segmentation_MOTS20_Scaling_Study", "README.md"),
        ("YOLO_Large_Seg_MOTS20_Benchmark", "README.md"),
        ("YOLO_Large_Seg_MOTS20_Benchmark", "RESULTS_SUMMARY_TH.md"),
        ("YOLO_Large_Seg_MOTS20_Benchmark", "PRESENTATION_SUMMARY_TH.md")]
    for repo, name in samples:
        url = f"https://raw.githubusercontent.com/folklazy/{repo}/main/{name}"
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read()
        digest = hashlib.sha256(body).hexdigest()
        assert digest == sha(WORKSPACE / repo / name)
        checked.append({"url": url, "sha256": digest, "matches_local": True})
    return checked


def plots(repo, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = repo / "metrics/plots"
    output.mkdir(parents=True, exist_ok=True)
    names = [row["model"].removesuffix("-seg.pt") for row in rows]
    colors = [COLORS[family(row["model"])] for row in rows]
    for filename, key, ylabel in [
        ("01_box_ap50_95.png", "box_ap50_95", "Person Box AP@[.50:.95] (%)"),
        ("03_inference_latency.png", "inference_mean_ms", "Forward mean (ms/image)"),
        ("04_pipeline_fps.png", "fps", "Adapter FPS (excludes I/O and RLE)"),
        ("05_peak_vram.png", "vram_allocated_mib", "Peak allocated VRAM (MiB)")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        values = [row[key] * (100 if key == "box_ap50_95" else 1) for row in rows]
        bars = ax.bar(names, values, color=colors)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.15)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(rows))
    ax.bar(x - .18, [row["box_ap50"] * 100 for row in rows], .36, label="AP50", color="#0072B2")
    ax.bar(x + .18, [row["box_ap75"] * 100 for row in rows], .36, label="AP75", color="#D55E00")
    ax.set_xticks(x, names)
    ax.set_ylabel("Person Box AP (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "02_ap50_ap75.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for row, color in zip(rows, colors):
        ax.scatter(row["adapter_mean_ms"], row["box_ap50_95"] * 100, color=color)
        ax.annotate(row["model"].removesuffix("-seg.pt"), (row["adapter_mean_ms"], row["box_ap50_95"] * 100),
            fontsize=9, xytext=(5, 5), textcoords="offset points")
    ax.margins(.2)
    ax.set_xlabel("Adapter mean (ms/image)")
    ax.set_ylabel("Person Box AP (%)")
    fig.tight_layout()
    fig.savefig(output / "06_accuracy_vs_latency.png", dpi=160)
    plt.close(fig)


def master_figures(rows, tiers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = STUDY / "metrics/plots"
    names = [row["model"].removesuffix("-seg.pt") for row in rows]
    colors = [COLORS[family(row["model"])] for row in rows]
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(names, [row["box_ap50_95"]*100 for row in rows], color=colors)
    ax.set_ylabel("Person Box AP@[.50:.95] (%)")
    ax.tick_params(axis="x", rotation=65)
    save(fig, "box_ap_by_model.png")
    for metric, filename, ylabel in [
        ("box_ap50_95","family_accuracy_scaling.png","Person Box AP (%)"),
        ("inference_mean_ms","family_forward_scaling.png","Forward mean (ms/image)"),
        ("fps","family_fps_scaling.png","Adapter FPS"),
        ("vram_allocated_mib","family_vram_scaling.png","Peak allocated VRAM (MiB)")]:
        fig, ax = plt.subplots(figsize=(9,5))
        for name, color in COLORS.items():
            selected = [row for row in rows if family(row["model"])==name]
            ax.plot([[t["id"] for t in tiers].index(row["tier"]) for row in selected],
                [row[metric]*(100 if metric=="box_ap50_95" else 1) for row in selected], "o-", color=color, label=name)
        ax.set_xticks(range(5), ["X/E","L/C","M","S","N"])
        ax.set_ylabel(ylabel)
        ax.legend()
        save(fig, filename)
    for cost, filename, xlabel, frontier_file in [
        ("parameters","accuracy_parameters.png","Unfused checkpoint parameters",None),
        ("inference_mean_ms","accuracy_forward_tradeoff.png","Forward mean (ms/image)","PARETO_FRONTIER.csv"),
        ("adapter_mean_ms","accuracy_pipeline_tradeoff.png","Adapter mean (ms/image)","PARETO_PIPELINE.csv"),
        ("vram_allocated_mib","accuracy_vram_tradeoff.png","Peak allocated VRAM (MiB)","PARETO_VRAM.csv")]:
        fig, ax = plt.subplots(figsize=(12,6))
        frontier = {item["model"] for item in read_csv(STUDY/"metrics"/frontier_file) if item["nondominated"]=="True"} if frontier_file else set()
        for i,row in enumerate(rows):
            ax.scatter(row[cost], row["box_ap50_95"]*100, color=COLORS[family(row["model"])], s=50)
            if row["model"] in frontier:
                ax.scatter(row[cost], row["box_ap50_95"]*100, s=130, facecolors="none", edgecolors="black", linewidths=.8)
            ax.annotate(row["model"].removesuffix("-seg.pt"), (row[cost],row["box_ap50_95"]*100), fontsize=8,
                xytext=(4,5 if i%2==0 else -11),textcoords="offset points")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Person Box AP (%)")
        ax.set_title("17 completed checkpoints · ring indicates nondominated model" if frontier_file else "17 completed checkpoints")
        ax.margins(.1)
        save(fig, filename)
    composition = read_csv(STUDY/"metrics/TIMING_COMPOSITION.csv")
    fig, ax = plt.subplots(figsize=(12,5))
    bottom = np.zeros(17)
    for key,label,color in [("preprocess_ms","Preprocess/H2D","#88CCEE"),("inference_ms","Forward","#0072B2"),("postprocess_ms","Postprocess/native masks","#D55E00")]:
        values = np.array([float(item[key]) for item in composition])
        ax.bar(names,values,bottom=bottom,label=label,color=color)
        bottom += values
    ax.set_ylabel("Adapter mean (ms/image)")
    ax.tick_params(axis="x",rotation=65)
    ax.legend()
    save(fig,"timing_composition.png")
    for tier in tiers:
        models = load(WORKSPACE/tier["repository"]/"metrics/RESULTS.json")["models"]
        for kind in ["pr","f1"]:
            fig,ax=plt.subplots(figsize=(8,5))
            for model in models:
                if kind=="pr":
                    curve=model["primary"]["pr_curve_iou50"]
                    x,y=curve["recall"],curve["precision"]
                else:
                    curve=model["confidence_curve"]
                    x,y=[r["confidence"] for r in curve],[r["f1"] for r in curve]
                ax.plot(x,y,label=model["model"],color=COLORS[family(model["model"])])
            ax.set_xlabel("Recall" if kind=="pr" else "Confidence threshold")
            ax.set_ylabel("Interpolated precision at box IoU .50" if kind=="pr" else "Fixed-matcher F1")
            ax.set_ylim(0,1.01)
            ax.legend(fontsize=8)
            save(fig,tier["id"]+"_"+kind+"_curve.png")
    deltas=read_csv(STUDY/"metrics/SCALING_DELTAS.csv")
    pairs=list(dict.fromkeys((row["larger"],row["smaller"]) for row in deltas))
    fig,axes=plt.subplots(1,2,figsize=(12,8),sharey=True)
    labels=[a.removesuffix("-seg.pt")+" → "+b.removesuffix("-seg.pt") for a,b in pairs]
    for ax,key,value_key,label in [(axes[0],"box_ap50_95","delta_percentage_points","AP change (percentage points)"),
        (axes[1],"inference_mean_ms","relative_percent","Forward change (%)")]:
        values=[float(next(r[value_key] for r in deltas if r["larger"]==a and r["smaller"]==b and r["metric"]==key)) for a,b in pairs]
        ax.barh(range(13),values,color=[COLORS[family(a)] for a,b in pairs])
        ax.set_xlabel(label)
        ax.axvline(0,color="black",lw=.8)
        ax.set_yticks(range(13),labels)
    axes[0].invert_yaxis()
    save(fig,"adjacent_scaling_deltas.png")


def render_cases(prepared, tier, output):
    """68 saved-packet readbacks, no dataset-wide scoring or model inference."""
    frames = {frame["image_id"]: frame for frame in prepared["frames"]}
    cfg = prepared["config"]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    cases = []
    output.mkdir(parents=True, exist_ok=True)
    specifications = CASE_SPECS
    if tier["id"] == "small":
        specifications = [CASE_SPECS[0],
            (8, "B", "คนหลายคนบนทางเดิน", "ตรวจการเก็บคนและกรอบเกินที่แยกพฤติกรรมของโมเดลในกลุ่ม Small ได้"),
            CASE_SPECS[2], CASE_SPECS[3]]
    for position, letter, title, reason in specifications:
        frame = frames[prepared["visual_sample"][position]]
        assert sha(Path(frame["image_path"])) == frame["image_sha256"]
        source = Image.open(frame["image_path"]).convert("RGB")
        ground = source.copy()
        draw = ImageDraw.Draw(ground)
        for i, person in enumerate(frame["people"]):
            draw.rectangle(person["bbox"], outline="#29ad71", width=3)
            draw.text(tuple(person["bbox"][:2]), f"G{i+1}", font=font, fill="white", stroke_width=2, stroke_fill="black")
        for box in frame["ignore"]:
            draw.rectangle(box, outline="#888888", width=3)
        roi = [0, 0, source.width, source.height]
        if frame["people"]:
            boxes = [person["bbox"] for person in frame["people"]]
            roi = [max(0, int(min(b[0] for b in boxes)) - 40), max(0, int(min(b[1] for b in boxes)) - 40),
                min(source.width, int(max(b[2] for b in boxes)) + 40), min(source.height, int(max(b[3] for b in boxes)) + 40)]
        panels = [("Original + GT (green) / ignore (gray)", ground)]
        records = []
        for model in tier["models"]:
            directory = model_directory(prepared, tier, model)
            path = directory / "predictions" / (frame["image_id"] + ".json.gz")
            record = read_packet(path, frame, model_binding(prepared, model))
            detail = fixed(frame, record["predictions"], cfg["fixed_confidence"], cfg["matching_iou"], cfg["ignore_prediction_ioa"])
            with (directory / "PER_IMAGE.csv").open(newline="") as stream:
                canonical = next(row for row in csv.DictReader(stream) if row["image_id"] == frame["image_id"])
            for key in ["tp", "fp", "fn"]:
                assert detail[key] == int(canonical[key])
            ordered = sorted((p for p in record["predictions"] if p["confidence"] >= cfg["fixed_confidence"]),
                key=lambda p: (-p["confidence"], *p["bbox_xyxy"]))
            image = np.asarray(source).copy()
            for prediction, event in zip(ordered, detail["events"]):
                rle = prediction["rle"]
                mask = coco_mask.decode({"size": rle["size"], "counts": rle["counts"].encode()}).astype(bool)
                color = {"TP": [40, 170, 110], "FP": [220, 55, 55], "IGNORE": [145, 145, 145]}[event["disposition"]]
                image[mask] = (image[mask] * .65 + np.array(color) * .35).astype(np.uint8)
            panel = Image.fromarray(image)
            draw = ImageDraw.Draw(panel)
            for prediction, event in zip(ordered, detail["events"]):
                color = {"TP": "#29ad71", "FP": "#dc3737", "IGNORE": "#888888"}[event["disposition"]]
                draw.rectangle(prediction["bbox_xyxy"], outline=color, width=3)
            for index in detail["fn_indices"]:
                draw.rectangle(frame["people"][index]["bbox"], outline="#ff9600", width=4)
            per_gt = []
            for i, person in enumerate(frame["people"]):
                match = next((item for item in detail["matches"] if item["gt_index"] == i), None)
                best = max((box_iou(person["bbox"], prediction["bbox_xyxy"]) for prediction in ordered), default=0.)
                per_gt.append({"gt_index": i+1, "state": "TP" if match else "FN", "matched_iou": match["box_iou"] if match else None,
                    "best_candidate_iou_at_fixed_confidence": best})
            matched_indices = {match["gt_index"] for match in detail["matches"]}
            extra_on_matched_gt = sum(event["disposition"] == "FP" and any(
                box_iou(prediction["bbox_xyxy"], frame["people"][j]["bbox"]) >= cfg["matching_iou"] for j in matched_indices)
                for prediction, event in zip(ordered, detail["events"]))
            records.append({"model": model, "packet_sha256": sha(path), "counts": {key: detail[key] for key in
                ["tp", "fp", "fn", "ignored_predictions"]}, "per_gt": per_gt,
                "extra_boxes_overlapping_already_matched_gt_at_iou50": extra_on_matched_gt})
            panels.append((model + f"  TP {detail['tp']}  FP {detail['fp']}  FN {detail['fn']}", panel))
        tile_w, tile_h, header_h = 880, 495, 58
        canvas = Image.new("RGB", (tile_w * 2, (tile_h + header_h) * len(panels)), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (label, panel) in enumerate(panels):
            for j, view in enumerate([panel, panel.crop(roi)]):
                view = ImageOps.contain(view, (tile_w, tile_h), method=Image.Resampling.BICUBIC)
                origin = (j * tile_w, i * (tile_h + header_h))
                draw.text((origin[0]+8, origin[1]+4), label, font=font, fill="black")
                draw.text((origin[0]+8, origin[1]+29), "Full frame" if j == 0 else "Same ROI " + str(roi), font=font, fill="black")
                canvas.paste(view, (origin[0] + (tile_w-view.width)//2, origin[1]+header_h))
        target = output / ("case_" + letter + ".png")
        canvas.save(target)
        cases.append({"case": letter, "position_1based": position+1, "title": title, "reason": reason,
            "image_id": frame["image_id"], "image_sha256": frame["image_sha256"] , "annotation_sha256": frame["annotation_sha256"],
            "roi": roi, "gt_people": len(frame["people"]), "models": records, "file": target.name, "sha256": sha(target)})
    write(output / "MANIFEST.json", {"source_binding": prepared["scientific_binding"],
        "selection": "Post-study GT/scene/error-pattern diagnostics within original fixed16; three shared anchors and a Small-specific comparison; no metric/settings changes",
        "cases": cases})
    private_lines = ["# หลักฐานภาพรายกรณี", "", "สีเขียว TP, สีแดง FP, สีส้ม FN และสีเทา ignored predictions ใช้ bbox matching เดิม Mask overlay เป็น native output diagnostics GT แสดง visible bbox", ""]
    for case in cases:
        private_lines += ["## Case " + case["case"] + " — " + case["title"], "", case["reason"], "",
            "![full frame and same ROI](" + case["file"] + ")", "",
            markdown_table(["Model", "TP", "FP", "FN", "IGN"], [[item["model"]] + [item["counts"][key] for key in ["tp", "fp", "fn", "ignored_predictions"]] for item in case["models"]]), ""]
        for item in case["models"]:
            private_lines += ["### " + item["model"], "", markdown_table(["GT", "ผล", "Matched IoU", "Best candidate IoU"],
                [["G"+str(p["gt_index"]), p["state"], "—" if p["matched_iou"] is None else f"{p['matched_iou']:.3f}",
                    f"{p['best_candidate_iou_at_fixed_confidence']:.3f}"] for p in item["per_gt"]]), ""]
    document(output / "PRESENTATION_CASES_TH.md", "\n".join(private_lines))
    body = "<meta charset='utf-8'><title>CCTV600 diagnostic cases</title><style>body{font-family:sans-serif;margin:24px}img{max-width:100%}</style>"
    for case in cases:
        body += "<h2>Case " + case["case"] + " · " + html.escape(case["title"]) + "</h2><p>" + html.escape(case["reason"]) + "</p><img src='" + case["file"] + "'>"
    document(output / "index.html", body)
    return cases


def role_table(rows):
    ordered = sorted(rows, key=lambda row: row["box_ap50_95"], reverse=True)
    leader = ordered[0]
    forward = min(rows, key=lambda row: row["inference_mean_ms"])
    adapter = min(rows, key=lambda row: row["adapter_mean_ms"])
    memory = min(rows, key=lambda row: row["vram_allocated_mib"])
    output = []
    for row in rows:
        strengths = []
        if row == leader: strengths.append("Box AP สูงสุดในกลุ่ม")
        if row["precision"] == max(r["precision"] for r in rows): strengths.append("Precision สูงสุด")
        if row["recall"] == max(r["recall"] for r in rows): strengths.append("Recall สูงสุด")
        if row == forward: strengths.append("forward เร็วสุดในกลุ่ม")
        if row == adapter: strengths.append("adapter เร็วสุดในกลุ่ม")
        if row == memory: strengths.append("allocated VRAM ต่ำสุดในกลุ่ม")
        gap = (leader["box_ap50_95"] - row["box_ap50_95"]) * 100
        tradeoffs = []
        if gap:
            tradeoffs.append(f"AP ห่างตัวนำ {gap:.3f} pp")
        recall_gap = (max(r["recall"] for r in rows) - row["recall"]) * 100
        if recall_gap:
            tradeoffs.append(f"Recall ห่างตัวนำ {recall_gap:.3f} pp")
        if row != adapter:
            tradeoffs.append(f"adapter เพิ่ม {row['adapter_mean_ms']-adapter['adapter_mean_ms']:.3f} ms")
        if row != memory:
            tradeoffs.append(f"VRAM เพิ่ม {row['vram_allocated_mib']-memory['vram_allocated_mib']:.3f} MiB")
        priorities = []
        if row == leader: priorities.append("เน้น Box AP")
        if row["precision"] == max(r["precision"] for r in rows): priorities.append("ลดสัดส่วนกรอบเกิน")
        if row["recall"] == max(r["recall"] for r in rows): priorities.append("เก็บคนให้ครบ")
        if row == adapter: priorities.append("จำกัดเวลา adapter")
        if row == memory: priorities.append("จำกัด allocated VRAM")
        output.append([row["model"], ", ".join(strengths) if strengths else f"AP {fmt(row, 'box_ap50_95')}% และ Recall {fmt(row, 'recall')}%",
            ", ".join(tradeoffs[:3]) if tradeoffs else f"FP {row['fp']}, FN {row['fn']} ที่ threshold เดิม",
            ", ".join(priorities) if priorities else "ใช้เป็นจุดอ้างอิงของ checkpoint ในกลุ่มนี้"])
    return markdown_table(["โมเดล", "จุดเด่นที่วัดได้", "ข้อแลกเปลี่ยน", "เงื่อนไขการเลือก"], output)


def tier_documents(tier, tiers, rows, groups, cases, revision):
    repo = WORKSPACE / tier["repository"]
    title = TITLES[tier["id"]]
    leader, second = sorted(rows, key=lambda row: row["box_ap50_95"] , reverse=True)[:2]
    fast = min(rows, key=lambda row: row["adapter_mean_ms"])
    memory = min(rows, key=lambda row: row["vram_allocated_mib"])
    speed_finding = (f"ตัวนำ AP มี adapter ต่ำสุดในกลุ่มด้วยที่ {fmt(leader, 'adapter_mean_ms')} ms/ภาพ"
        if leader == fast else f"Adapter ของ {fast['model']} เร็วกว่าตัวนำ AP {leader['adapter_mean_ms']-fast['adapter_mean_ms']:.3f} ms/ภาพ")
    memory_finding = (f"ตัวนำ AP มี allocated VRAM ต่ำสุดในกลุ่มด้วยที่ {fmt(memory, 'vram_allocated_mib')} MiB"
        if leader == memory else f"ตัวนำ AP ใช้ allocated VRAM มากกว่าค่าต่ำสุด {leader['vram_allocated_mib']-memory['vram_allocated_mib']:.3f} MiB")
    totals = {model: {key: sum(next(item for item in case["models"] if item["model"] == model)["counts"][key] for case in cases)
        for key in ["tp", "fp", "fn", "ignored_predictions"]} for model in tier["models"]}
    note = "Box AP/P/R/F1 แสดงเป็นเปอร์เซ็นต์ Forward และ adapter เป็น ms/ภาพ, FPS = 1000 / mean adapter ms และ VRAM เป็น peak allocated MiB"
    caveats = "GT ใช้ visible bbox ของคนด้านยาว ≥50 px รวม 769 คน การรับรอง person inventory อิงการตรวจรับของผู้จัดทำ dataset ยังไม่มี independent inventory review ครบชุด ขนาดโมเดลและข้อมูล pretraining ต่างกัน การเปรียบเทียบนี้ยังไม่แยกสาเหตุทาง architecture ไม่มี uncertainty analysis และ Person Mask AP ต้องมี GT silhouette เพิ่ม"
    summary = f"""# สรุปผล {title} บน CCTV600

## สรุปใน 1 นาที

- ทดสอบ {len(rows)} pretrained checkpoints บน 600 ภาพชุดเดียวกันและ GT เข้าเกณฑ์ 769 คน
- Input 640×640, batch 1, FP32 บน Tesla T4 ไม่มี fine-tuning ใน benchmark รอบนี้
- คุณภาพครบ 600 IDs ต่อโมเดล และเวลาครบ 3 clean rounds ต่อโมเดล
- Box AP สูงสุดเป็น {leader['model']} ที่ {fmt(leader, 'box_ap50_95')}%
- Adapter time ต่ำสุดเป็น {fast['model']} ที่ {fmt(fast, 'adapter_mean_ms')} ms/ภาพ
- Allocated VRAM ต่ำสุดเป็น {memory['model']} ที่ {fmt(memory, 'vram_allocated_mib')} MiB

## ผลลัพธ์หลัก

{results_table(rows)}

{note}

## ผู้ชนะของแต่ละด้าน

{winner_table(rows)}

## สิ่งที่ตัวเลขบอกเรา

- {leader['model']} มี AP สูงกว่า {second['model']} {(leader['box_ap50_95']-second['box_ap50_95'])*100:.3f} จุดเปอร์เซ็นต์
- ตัวนำ AP มี TP {leader['tp']}, FP {leader['fp']} และ FN {leader['fn']} ที่ confidence ≥0.25 / bbox IoU ≥0.50
- {speed_finding}
- {memory_finding}

## บทบาทของแต่ละโมเดล

{role_table(rows)}

## ข้อแลกเปลี่ยนหลัก

### คุณภาพกับความเร็ว

ตัวนำ AP มี AP {fmt(leader,'box_ap50_95')}% ส่วนตัว adapter เร็วสุดมี AP {fmt(fast,'box_ap50_95')}% ทั้งสองเป้าหมายอาจได้ checkpoint เดียวกัน ให้พิจารณา forward กับ adapter แยกกัน เพราะ native mask postprocessing มีต้นทุนอยู่หลัง model forward

### คุณภาพกับหน่วยความจำ

ตัว allocated VRAM ต่ำสุดมี AP {fmt(memory,'box_ap50_95')}% และใช้ {fmt(memory,'vram_allocated_mib')} MiB ค่า allocator peak นี้ครอบคลุม resident model ใน benchmark หลัง warmup ส่วน memory รวมในระบบให้ตรวจ CUDA context และบริการที่รันร่วมด้วย

## ข้อควรระวังในการตีความ

{caveats} Adapter time ครอบคลุม preprocessing/H2D, forward และ postprocessing/native-mask transfer โดย input I/O, RLE, writing, model loading และ queue time มีขอบเขตแยก

## อ่านต่อ

[รายงานเทคนิค](REPORT.md) · [การวิเคราะห์ภาพและข้อผิดพลาด](PRESENTATION_SUMMARY_TH.md) · [ตารางเต็ม](metrics/TIER_RESULTS.csv) · [สูตรตัวชี้วัด]({MASTER_DOC}METRIC_GUIDE_TH.md)
"""
    document(repo / "RESULTS_SUMMARY_TH.md", summary)
    document(repo / "README.md", f"""# {title} YOLO Segmentation Benchmark — CCTV600

## ภาพรวม

ประเมิน pretrained segmentation checkpoints กลุ่ม {title} บน CCTV Human Parsing 600 ด้วยคะแนนกรอบบุคคล ทุกรุ่นใช้ input 640, FP32, batch 1 และ population เดียวกันบน Tesla T4

## สถานะ

**COMPLETE** คุณภาพครบ {len(rows)}/{len(rows)} โมเดล ตัวละ 600 ภาพ เวลาครบ 3 clean rounds ต่อโมเดล อ่านแหล่งผลและประวัติ timing continuation จาก [สถานะรวม]({MASTER_DOC}metrics/CLOSEOUT.json)

## ผลหลัก

{results_table(rows)}

{note}

## รายงาน

| เอกสาร | เนื้อหา |
| --- | --- |
| [RESULTS_SUMMARY_TH.md](RESULTS_SUMMARY_TH.md) | คะแนน ผู้ชนะรายด้าน บทบาทและข้อแลกเปลี่ยนของแต่ละโมเดล |
| [PRESENTATION_SUMMARY_TH.md](PRESENTATION_SUMMARY_TH.md) | กราฟและหลักฐานข้อผิดพลาดจาก fixed sample เดิม |
| [REPORT.md](REPORT.md) | Models, compatibility, ผลเต็ม, timing, source และข้อจำกัด |
| [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) | เงื่อนไขที่ใช้รัน |

## ภาพรวมคุณภาพ

![Box AP](metrics/plots/01_box_ap50_95.png)

## การนำทางทั้ง study

{navigation(tiers)}

## ตรวจย้อนกลับ

[ผลและ timings](metrics/) · [Config](configs/benchmark.json) · [Bindings]({MASTER_DOC}metrics/RESULT_INDEX.json) · [คู่มือตัวชี้วัด]({MASTER_DOC}METRIC_GUIDE_TH.md)

ภาพ CCTV และ per-frame records เก็บ private ชุดหลักฐานภาพของรอบตรวจเอกสารอยู่ใน Master ที่ `outputs/{revision}/cases/{tier['id']}/index.html` ใช้คู่กับรายงานภาพ โดยเก็บภาพเต็มและ ROI พิกัดเดียวกันครบทุกโมเดล

คำสั่งจาก workspace สำหรับตรวจสถานะปิดรอบ

```bash
.venv/bin/python {STUDY.name}/analysis/closeout.py --status
```
""")
    group_table = markdown_table(["Model", "Day AP %", "Night AP %", "R 50–99 px %", "R 100–199 px %", "R ≥200 px %"], [
        [row["model"]] + [f"{next(g[key] for g in groups if g['model']==row['model'] and g['group']==group)*100:.3f}"
            for group,key in [("day","box_ap50_95"),("night","box_ap50_95"),("50to99","recall"),("100to199","recall"),("200plus","recall")]] for row in rows])
    counts_table = markdown_table(["Model", "Precision %", "Recall %", "F1 %", "TP", "FP", "FN", "Ignored", "TP-only box IoU"], [
        [row["model"]]+[fmt(row,k) for k in ["precision","recall","f1"]]+[row[k] for k in ["tp","fp","fn","ignored_predictions"]]+[f"{row['matched_box_iou_mean']:.6f}"] for row in rows])
    timing = load(STUDY / "metrics/RESULT_INDEX.json")
    distributions = markdown_table(["Model", "Pre mean ms", "Post mean ms", "Adapter P50 ms", "Adapter P95 ms", "Adapter std ms", "Reserved MiB"], [
        [row["model"]]+[f"{item['timing_primary'][stage][stat]:.3f}" for stage,stat in
            [("preprocess_ms","mean"),("postprocess_ms","mean"),("total_ms","median"),("total_ms","p95"),("total_ms","std")]]+[fmt(row,"vram_reserved_mib")]
        for row in rows for item in timing["models"] if item["model"]==row["model"]])
    compatibility = markdown_table(["Item", "Status and evidence"], [
        ["Dataset", "PASS · 600 exact IDs; dataset membership SHA in frozen config"],
        ["Evaluator", "PASS · GT-first bbox matching and shared ignore union"],
        ["Preprocessing", "PASS · frozen runtime source binding"],
        ["Input size", "1×3×640×640, centered letterbox"], ["Precision", "FP32; no AMP/TF32/extra TTA"],
        ["Thresholds", "AP floor .001, fixed conf .25, match IoU .50, NMS IoU .70"],
        ["maxDet", "Model 1000 / evaluator 200; cap parity PASS"],
        ["Timing", "3×100 clean observations, warmup 10 each round; sample order verified"],
        ["Environment", "Python 3.12.3 / torch 2.14.0+cu130 / Ultralytics 8.4.160 / Tesla T4"]])
    model_table = markdown_table(["Family", "Checkpoint", "Parameters", "GFLOPs estimate", "Weights MB"],
        [[family(r["model"]),r["model"],f"{r['parameters']:,}",fmt(r,"gflops_estimate"),fmt(r,"checkpoint_mb")] for r in rows])
    report = f"""# {title} YOLO Segmentation Benchmark — CCTV600

## 1. Experiment Status

COMPLETE for accuracy and dedicated clean timing. Each of {len(rows)} models has 600 accepted unique image packets. Evaluation includes 769 eligible person boxes. Dataset-quality and interpretation limitations remain documented below.

## 2. Models Tested

{model_table}

## 3. Protocol Compatibility

{compatibility}

Compatibility applies within this CCTV600 experiment. MOTS20 supplies the reporting structure and reused detector runtime. CCTV600 uses visible box GT, its ≥50 px eligibility and coarse rectangle-ignore policy. MOTS20 uses person silhouettes and mask matching.

## 4. Overall Results

{results_table(rows)}

{counts_table}

AP/P/R/F1 columns use percentages. TP-only box IoU is a fraction. Counts use confidence ≥0.25 and box IoU ≥0.50. AP uses 10 IoU thresholds, 101 recall points and maxDet 200. Source-precision values remain in CSV.

## 5. Tier Winners

{winner_table(rows)}

## 6. Key Findings

The highest Box AP is {leader['model']} ({fmt(leader,'box_ap50_95')}%), {(leader['box_ap50_95']-second['box_ap50_95'])*100:.3f} percentage points above the runner-up. The shortest adapter mean belongs to {fast['model']} ({fmt(fast,'adapter_mean_ms')} ms/image). Differences are descriptive; no significance or architecture-causality claim is made.

## 7. Per-group Observations

{group_table}

Day/night use existing metadata and contain different scenes and people. Pooled AP is read directly from the canonical full-population result. Size-band recall conditions on GT size and the original matching; precision/AP by size were not computed under a new policy.

## 8. Efficiency and Resource Observations

{distributions}

Means, median, P95 and population standard deviation use all 300 accepted observations. Adapter FPS excludes disk/decode, RLE, packet writing and load/warmup. Native postprocessing substage timers are inclusive subsets. Allocated/reserved peaks include the resident model and take the maximum across the three accepted rounds.

## 9. Warnings and Anomalies

The original study ended with incomplete Nano timing. A linked segment supplied four accepted clean rounds, retaining the original contaminated YOLO11n round and original completion unchanged. Accuracy was not rerun. Detailed accepted round paths and hashes are in the common result index. The owned web worker was restored after authorized temporary suspension.

## 10. Limitations

GT inventory relies on the dataset creator's 600-image acceptance. Independent all-person inventory adjudication remains unavailable. No full-person silhouette GT, mask AP, tracking or parsing/color score is reported here. TP-only IoU excludes missed people. Different pretrained capacities and training histories confound architecture attribution. Related frames and absent uncertainty analysis limit generalization.

## 11. Reproducibility and Source Artifacts

[Canonical tier table](metrics/TIER_RESULTS.csv), [timing statistics](metrics/TIMING_SUMMARY.csv), [shared source/protocol/checkpoint bindings]({MASTER_DOC}metrics/RESULT_INDEX.json), [closed-run checks]({MASTER_DOC}metrics/CLOSEOUT.json), [metric definitions]({MASTER_DOC}METRIC_GUIDE_TH.md) and [presentation revision evidence]({MASTER_DOC}metrics/PRESENTATION_REVISION.json).

## 12. Relation to Full Scaling Study

{navigation(tiers)}

## Qualitative Analysis

[PRESENTATION_SUMMARY_TH.md](PRESENTATION_SUMMARY_TH.md) discusses four diagnostic cases from the original fixed16. Full frames, identical ROIs and per-GT matching are retained in the private local gallery. Only aggregate charts and de-identified qualitative observations are published.
"""
    document(repo / "REPORT.md", report)
    selected_counts = markdown_table(["Model", "TP รวม 4 ภาพ", "FP รวม 4 ภาพ", "FN รวม 4 ภาพ", "IGN รวม 4 ภาพ"],
        [[model]+[totals[model][k] for k in ["tp","fp","fn","ignored_predictions"]] for model in tier["models"]])
    case_text = []
    for case in cases:
        item = case["models"]
        has_fn = [r["model"] for r in item if r["counts"]["fn"]]
        has_fp = [r["model"] for r in item if r["counts"]["fp"]]
        has_ignore = [r["model"] for r in item if r["counts"]["ignored_predictions"]]
        observation = "มี FN ใน " + ", ".join(has_fn) if has_fn else "ทุกโมเดลจับคู่ eligible GT ที่มีในกรณีนี้ได้ครบ"
        if case["gt_people"] == 0:
            observation = "กรณีนี้ไม่มี eligible GT ตาม inventory ที่รับรอง"
        observation += " มี unmatched FP ใน " + ", ".join(has_fp) if has_fp else " และไม่พบ unmatched FP ที่ operating point เดิม"
        if has_ignore: observation += " ส่วน " + ", ".join(has_ignore) + " มี prediction ที่เข้ากติกา ignore"
        interpretation = {
            "A": "GT ของคนที่ชนขอบเฟรมล้อมส่วนที่มองเห็น การคืนพิกัดและการจับคู่ต้องใช้พิกัดภาพต้นฉบับเดียวกัน กรอบที่เห็นอยู่บนคนอาจผ่าน confidence แต่ยังไม่ผ่าน IoU จึงต้องอ่าน candidate/matched IoU ประกอบ",
            "B": "คนถูกโครงสร้างพาดผ่านและมีมากกว่าหนึ่ง GT ในภาพ การได้ TP ครบต้องดูกรอบส่วนเกินควบคู่กัน FP อาจอยู่บนคนที่มีคู่แล้ว จึงไม่ใช้จำนวน output แทนจำนวนคนที่เก็บได้",
            "C": "มี GT หลายขนาดและพื้นที่ ignore ในภาพเดียวกัน ข้อผิดพลาดร่วมในภาพนี้แสดงกรณีที่คะแนนเฉลี่ยสูงยังพลาดได้ ภาพนี้ยังไม่แยกสาเหตุด้านแสง ขนาด และการบดบังออกจากกัน Ignored prediction ต้องแยกจาก FP และยังไม่ชดเชย FN ของ valid GT",
            "D": "เป็น negative control ตาม inventory ที่รับรอง เกณฑ์ confidence ยังคง 0.25 การไม่เหลือกรอบผ่านเกณฑ์ในภาพนี้ใช้ตรวจการทำงานบน negative frame แต่ยังไม่ยืนยันอัตรา false alarm ของฉากประเภทนี้ทั้งชุด"
        }[case["case"]]
        if case["case"] == "B" and tier["id"] == "small":
            interpretation = "ภาพมี GT คนสองคนบนทางเดิน เห็นความต่างระหว่างการเก็บคนให้ครบกับกรอบเกิน ตัวนำ AP รวมอาจมี FP ในภาพนี้ ขณะที่โมเดลอื่นอาจเก็บ GT ครบโดยไม่มี FP เป็นกรณีที่ช่วยตรวจข้อแลกเปลี่ยนโดยไม่สรุปจากอันดับรวมอย่างเดียว"
        duplicates = [r["model"] for r in item if r["extra_boxes_overlapping_already_matched_gt_at_iou50"]]
        if duplicates:
            interpretation += " ตรวจพบกรอบ FP ที่มี bbox IoU ≥0.50 กับ GT ที่ถูกจับคู่แล้วใน " + ", ".join(duplicates) + " จึงมีหลักฐานว่าเป็น output เกินบน GT เดิมในความหมายของ bbox matching"
        case_text.append(f"""## Case {case['case']} — {case['title']}

**เหตุผลเลือก** {case['reason']} ใช้ภาพและ ROI เดียวกันครบทุกโมเดลในกลุ่ม

**สิ่งที่ตรวจจาก saved output** {observation} ผลราย GT และ candidate IoU ตรวจเทียบกับ counts เดิมแล้วอยู่ในหลักฐาน private

**การตีความ** {interpretation}

**ใช้ประกอบการเลือก** ดูความครบถ้วนของคนพร้อมผลส่วนเกินที่ downstream pipeline ต้องรับ ไม่ใช้กรณีนี้แทนคะแนน 600 ภาพ หรือใช้ภาพประเมิน latency/VRAM
""")
    presentation = f"""# {title} — การวิเคราะห์ภาพและข้อผิดพลาด

## 1. ภาพรวมผลการทดลอง

{len(rows)} โมเดลครบคุณภาพ 600 ภาพและ clean timing กลุ่มนี้มี AP สูงสุดใน {leader['model']} อ่านตัวเลขและบทบาทใน [สรุปผล](RESULTS_SUMMARY_TH.md)

![Box AP](metrics/plots/01_box_ap50_95.png)

![AP50/AP75](metrics/plots/02_ap50_ap75.png)

## 2. วิธีอ่านหลักฐานภาพ

เลือกสี่กรณีหลังปิด benchmark จาก fixed sample 16 ภาพเดิม โดยดู coverage ของ GT/ฉากและรูปแบบ TP/FP/FN เพื่ออธิบายความต่างของ output มีทั้งกรณีที่แยกโมเดลได้ ข้อผิดพลาดร่วม และ negative control การเลือกภาพประกอบนี้ไม่เปลี่ยน population, settings หรือคะแนนหลัก

Case A/C/D ใช้ภาพร่วมกันทุก tier ส่วน Case B ของ Small ใช้ภาพทางเดินที่แยกพฤติกรรมของโมเดลได้ชัดกว่าอีกกรณีซึ่งผลเหมือนกันทุกตัว กลุ่มอื่นใช้ภาพที่มีโครงสร้างบังคน รวม 20 case slots จากภาพต้นทางต่างกันห้าภาพ รายการและเหตุผลอยู่ใน private selection manifest

ทุกภาพสร้างจาก original RGB และ lossless saved native masks เรียง Original/GT ตามด้วย YOLO26, YOLO11, YOLOv9 เฉพาะ X/E และ L/C แล้ว YOLOv8 มีทั้ง full frame และ ROI พิกัดเดียวกัน สีเขียว TP, แดง FP, ส้ม FN และเทา ignored prediction ใช้ confidence 0.25 และ bbox IoU 0.50 เดิม ภาพขยายปรับขนาดเพื่อแสดงผลแบบเดียวกันทุกโมเดล รายละเอียดที่เห็นยังจำกัดด้วยความละเอียดภาพต้นฉบับ

Person Mask AP ยังไม่มี GT รองรับ ภาพ mask ช่วยตรวจรูปแบบ output และ map-back ส่วนการตัดสิน TP/FN ในเอกสารนี้ยึด bbox matching ราย GT ที่ตรวจตรงกับ canonical per-image CSV

## 3. ผลรวมของชุดภาพประกอบ

{selected_counts}

ตัวเลขนี้รวมสี่ภาพต้นทางเดียวกัน GT เข้าเกณฑ์รวม {sum(case['gt_people'] for case in cases)} รายการ ใช้เป็น diagnostics ที่อ่านคู่กับภาพ ส่วนผล headline ยังคงประชากร 600 ภาพ ภาพและ IDs จริงเก็บ private ตามนโยบาย dataset

{chr(10).join(case_text)}

## 4. สิ่งที่ตัวเลขรวมกับภาพบอกต่างกัน

AP รวมสรุป ranking และคุณภาพกรอบหลาย IoU ส่วนภาพสี่กรณีช่วยตรวจว่า FN/FP อยู่บนคนใดและบริเวณใด TP เท่ากันอาจจับคนต่างชุดและ mask รูปร่างต่างกัน การชนะ AP จึงไม่ได้รับรองว่าเก็บ GT ทุกคนในทุกภาพได้ดีกว่า

## 5. ความเร็วและหน่วยความจำ

![Forward](metrics/plots/03_inference_latency.png)

![Adapter FPS](metrics/plots/04_pipeline_fps.png)

![Allocated VRAM](metrics/plots/05_peak_vram.png)

![AP vs adapter](metrics/plots/06_accuracy_vs_latency.png)

กราฟนี้มาจาก clean benchmark measurements ภาพ segmentation ใช้อธิบายเวลาหรือ memory โดยตรงไม่ได้ รายละเอียด stage distributions อยู่ใน REPORT.md

## 6. Near-tie และข้อเสนอการเลือก

{role_table(rows)}

อ่านคู่ near-tie จาก [ตารางความต่าง]({MASTER_DOC}metrics/NEAR_TIES.csv) เมื่อมีคู่เข้าเกณฑ์ AP ห่างไม่เกิน 0.1 pp ให้ตรวจ per-GT output ในภาพเดียวกันเพิ่ม การที่ไม่เข้าเกณฑ์ near-tie ยังไม่ใช่หลักฐาน statistical significance

## 7. ข้อจำกัด

{caveats} ชุดภาพประกอบมีสี่ภาพที่เลือกตามประเภทกรณี จึงไม่ใช้ประมาณความถี่ข้อผิดพลาดของ dataset ทั้งชุด และการใช้ภาพเดิมข้าม tier ไม่เพิ่มจำนวนตัวอย่างอิสระ

## 8. เปิดหลักฐานภาพจริงในเครื่อง

จาก workspace เปิด `../{STUDY.name}/outputs/{revision}/cases/{tier['id']}/index.html` เมื่อเริ่มจากโฟลเดอร์ repo นี้ หรือเปิด `PRESENTATION_CASES_TH.md` ในโฟลเดอร์เดียวกันเพื่ออ่าน counts และ IoU ราย GT ภาพทั้ง 16 เดิมยังอยู่ที่ `outputs/<original_run_id>/gallery/index.html`

ใน GitHub แสดงกราฟรวมและคำอธิบายที่ตัด IDs จริงออก ส่วนภาพ CCTV ใช้ private gallery ตามนโยบายเดิมของงานนี้

[ผลเชิงตัวเลข](RESULTS_SUMMARY_TH.md) · [รายงานเทคนิค](REPORT.md) · [Master]({MASTER_URL})
"""
    document(repo / "PRESENTATION_SUMMARY_TH.md", presentation)


def derived_outputs(rows, tiers, timing):
    winners = []
    for tier in tiers:
        winners.extend(winner_rows([row for row in rows if row["tier"] == tier["id"]]))
    write_csv(STUDY / "metrics/TIER_WINNERS.csv", winners)
    composition, plot_ready = [], []
    flags = {name: {item["model"]: item["nondominated"] for item in read_csv(STUDY / "metrics" / name)}
        for name in ["PARETO_FRONTIER.csv", "PARETO_PIPELINE.csv", "PARETO_VRAM.csv"]}
    for index, row in enumerate(rows):
        primary = next(item["timing_primary"] for item in timing["models"] if item["model"] == row["model"])
        item = {"model": row["model"], **{key: primary[key]["mean"] for key in
            ["preprocess_ms", "inference_ms", "postprocess_ms", "total_ms", "rle_preparation_ms", "ultralytics_postprocess_inclusive_ms"]}}
        for key in ["preprocess_ms", "inference_ms", "postprocess_ms"]:
            item[key + "_share"] = item[key] / item["total_ms"]
        assert abs(sum(item[k] for k in ["preprocess_ms", "inference_ms", "postprocess_ms"]) - item["total_ms"]) < 1e-8
        composition.append(item)
        plot_ready.append({**row, "display_order": index+1, "family": family(row["model"]),
            "forward_pareto": flags["PARETO_FRONTIER.csv"][row["model"]], "pipeline_pareto": flags["PARETO_PIPELINE.csv"][row["model"]],
            "vram_pareto": flags["PARETO_VRAM.csv"][row["model"]], **{k: item[k] for k in ["preprocess_ms","postprocess_ms","rle_preparation_ms"]}})
    write_csv(STUDY / "metrics/TIMING_COMPOSITION.csv", composition)
    write_csv(STUDY / "metrics/PLOT_READY_RESULTS.csv", plot_ready)
    near = []
    by_model = {row["model"]: row for row in rows}
    for pair in read_csv(STUDY / "metrics/NEAR_TIES.csv"):
        a, b = by_model[pair["model_a"]], by_model[pair["model_b"]]
        for key in ["box_ap50_95", "recall", "inference_mean_ms", "adapter_mean_ms", "vram_allocated_mib", "parameters", "checkpoint_mb"]:
            with localcontext() as ctx:
                ctx.prec = 50
                av, bv = Decimal(str(a[key])), Decimal(str(b[key]))
                delta = bv - av
                near.append({"model_a": a["model"], "model_b": b["model"], "metric": key,
                    "a": av, "b": bv, "delta_b_minus_a": delta, "relative_percent_against_a": delta / av * 100,
                    "delta_pp": delta * 100 if key in ["box_ap50_95", "recall"] else None})
    write_csv(STUDY / "metrics/NEAR_TIE_RESOURCE_DELTAS.csv", near)


def master_documents(rows, tiers, groups, revision):
    lead = max(rows,key=lambda r:r["box_ap50_95"])
    recall = max(rows,key=lambda r:r["recall"])
    forward = min(rows,key=lambda r:r["inference_mean_ms"])
    adapter = min(rows,key=lambda r:r["adapter_mean_ms"])
    memory = min(rows,key=lambda r:r["vram_allocated_mib"])
    frontiers = {name:[item["model"] for item in read_csv(STUDY/"metrics"/name) if item["nondominated"]=="True"]
        for name in ["PARETO_FRONTIER.csv","PARETO_PIPELINE.csv","PARETO_VRAM.csv"]}
    deltas = read_csv(STUDY / "metrics/SCALING_DELTAS.csv")
    xl = {r["metric"]: r for r in deltas if r["larger"]=="yolo26x-seg.pt" and r["smaller"]=="yolo26l-seg.pt"}
    sn_rows = []
    for prefix in ["yolo26", "yolo11", "yolov8"]:
        selected = {r["metric"]: r for r in deltas if r["larger"] == prefix+"s-seg.pt" and r["smaller"] == prefix+"n-seg.pt"}
        sn_rows.append([family(prefix+"s-seg.pt"), f"{float(selected['box_ap50_95']['delta_percentage_points']):.3f}"] +
            [f"{float(selected[key]['relative_percent']):.2f}" for key in ["inference_mean_ms", "adapter_mean_ms", "vram_allocated_mib"]])
    group_winners = []
    for tier in tiers:
        subset = [r for r in rows if r["tier"] == tier["id"]]
        pool_leader = max(subset, key=lambda r:r["box_ap50_95"])["model"]
        day_leader = max([g for g in groups if g["model"] in tier["models"] and g["group"] == "day"], key=lambda g:g["box_ap50_95"])["model"]
        night_leader = max([g for g in groups if g["model"] in tier["models"] and g["group"] == "night"], key=lambda g:g["box_ap50_95"])["model"]
        group_winners.append([TITLES[tier["id"]], pool_leader, day_leader, night_leader])
    near_models = {r[k] for r in read_csv(STUDY/"metrics/NEAR_TIES.csv") for k in ["model_a","model_b"]}
    context = "ครบ 17 checkpoints ตัวละ 600 ภาพ GT เข้าเกณฑ์ 769 คน พร้อม 3 clean timing rounds ต่อโมเดล ใช้ pretrained, input 640, batch 1, FP32 บน Tesla T4"
    observations = f"""## 1. ตัวนำคุณภาพและ coverage เป็นตัวเดียวกันหรือไม่

{lead['model']} มี Box AP {fmt(lead,'box_ap50_95')}% และ F1 {fmt(lead,'f1')}% ส่วน {recall['model']} มี Recall สูงสุด {fmt(recall,'recall')}% ตัวแรกมี FP {lead['fp']} / FN {lead['fn']} เทียบกับ FP {recall['fp']} / FN {recall['fn']} ของตัวหลัง เป็นข้อแลกเปลี่ยนที่ operating point เดิม การเลือกโมเดลสำหรับส่งคนเข้า parser จึงต้องระบุว่าจะรับผลเกินหรือพลาดคนได้เท่าไร

## 2. ลดขนาดแล้วได้อะไร

YOLO26x → YOLO26l ทำให้ AP เปลี่ยน {float(xl['box_ap50_95']['delta_percentage_points']):.3f} pp, forward เปลี่ยน {float(xl['inference_mean_ms']['relative_percent']):.2f}% และ adapter เปลี่ยน {float(xl['adapter_mean_ms']['relative_percent']):.2f}% ที่ค่าประเมินเดิม เป็นจุดพิจารณาต้นทุนหนึ่งจุด ยังไม่มีคะแนนถ่วงน้ำหนักที่ตัดสินว่าคุ้มสำหรับทุกงาน

![Family accuracy](metrics/plots/family_accuracy_scaling.png)

### ส่วนต่างเมื่อเปลี่ยนจาก S เป็น N

{markdown_table(['Family', 'AP change pp', 'Forward change %', 'Adapter change %', 'Allocated change %'], sn_rows)}

ค่าทั้งหมดเป็น N−S ใช้ S เป็นฐานร้อยละสัมพัทธ์ การลด forward จึงอาจเกิดร่วมกับ adapter หรือ allocated peak ที่สูงขึ้นได้ งานหลัง forward และจำนวน retained masks เป็นส่วนของ workload ที่ต้องอ่านประกอบ โดยยังไม่สรุปสาเหตุจาก architecture

![Adjacent size deltas](metrics/plots/adjacent_scaling_deltas.png)

## 3. Forward เร็วขึ้นส่งผลถึง pipeline เพียงใด

{forward['model']} forward ต่ำสุด {fmt(forward,'inference_mean_ms')} ms/ภาพ แต่ {adapter['model']} มี adapter ต่ำสุด {fmt(adapter,'adapter_mean_ms')} ms/ภาพ ความต่างของ native mask postprocessing ต้องอ่านจาก measured stage composition การอธิบายว่าเกิดจาก architecture โดยตรงต้องมีการทดลองแยกเพิ่ม

![Timing composition](metrics/plots/timing_composition.png)

## 4. ตัวเลือกบน Pareto เปลี่ยนตามเป้าหมายอย่างไร

| แกนเปรียบเทียบ | Nondominated checkpoints |
| --- | --- |
| AP / forward | {', '.join(frontiers['PARETO_FRONTIER.csv'])} |
| AP / adapter | {', '.join(frontiers['PARETO_PIPELINE.csv'])} |
| AP / allocated VRAM | {', '.join(frontiers['PARETO_VRAM.csv'])} |

ใช้ dominance จากสถิติเต็มก่อนปัดเศษ แต่ละ frontier ตอบวัตถุประสงค์สองแกน ค่า allocated ต่ำสุดรวมเป็น {memory['model']} ที่ {fmt(memory,'vram_allocated_mib')} MiB และยังต้องรวม CUDA context/บริการอื่นเมื่อประเมิน deployment memory

## 5. Near-tie ในกลุ่ม M

YOLOv8m มี AP สูงกว่า YOLO26m เพียง 0.052 pp ส่วน Recall และ resource ต่างกัน ค่าที่ต่างน้อยนี้เป็นการคัดกรองเชิงพรรณนา ยังไม่มี uncertainty เพื่อยืนยันความแตกต่างหรือ equivalence ตาราง [NEAR_TIE_RESOURCE_DELTAS.csv](metrics/NEAR_TIE_RESOURCE_DELTAS.csv) ระบุทิศทาง B−A และฐาน A ชัดเจน

{results_table([row for row in rows if row['model'] in near_models])}

## 6. Day/night และขนาดคน

{markdown_table(['กลุ่มขนาด', 'Pooled AP winner', 'Day AP winner', 'Night AP winner'], group_winners)}

![Day/night](metrics/plots/day_night_ap_heatmap.png)

![GT size recall](metrics/plots/size_recall_heatmap.png)

ผล day/night แยก metadata ที่มีอยู่ ภาพแต่ละกลุ่มมีสถานที่ มุม และจำนวนคนต่างกัน ความต่างจึงยังแยกสาเหตุด้านแสงไม่ได้ ส่วน size-band recall แสดงการเก็บ GT ที่ขนาดต่างกันภายใต้ matching เดิม AP รวมเป็น pooled AP ที่อ่านจาก canonical result โดยตรง

## 7. สิ่งที่หลักฐานภาพช่วยอธิบาย

ตรวจสี่กรณีต่อ tier จาก fixed16 เดิม รวม 20 comparison panels จากภาพต้นทางห้าภาพ มีทั้ง visible person ที่ถูกตัดขอบ คนหลังโครงสร้างบัง คนบนทางเดิน คนหลายขนาดร่วมกับ ignore และ negative frame พร้อมภาพเต็ม/ROI ตรงกันและ per-GT matching เทียบ canonical counts Case A/C/D ใช้ร่วมกันทุก tier ส่วน B ของ Small ใช้กรณีที่แยกผลโมเดลได้ แทนกรณีที่ทั้งสามตัวให้ counts เหมือนกัน

รายงานภาพของแต่ละ tier แยก FN ซึ่งอาจมี candidate ที่ IoU ไม่ผ่านออกจากการไม่สร้าง prediction และแยก FP จาก ignored predictions การดู mask รูปร่างใกล้คนไม่ได้รับรองว่าผ่านเกณฑ์ bbox หรือว่าขอบ mask ถูกต้องทั้งตัว ภาพจริงและ per-image IDs คงเป็น private

## 8. ขอบเขตคำอ้าง

ผลนี้เปรียบเทียบ pretrained checkpoint พร้อม configuration บน CCTV600 ยังไม่แยก architecture/pretraining causality, ไม่มี significance analysis และยังไม่มี GT silhouette สำหรับ Person Mask AP การรับรอง inventory อิงการตรวจรับของผู้จัดทำ dataset ทั้ง 600 ภาพ ยังขาด independent full-inventory adjudication

Adapter FPS มีขอบเขต preprocess/H2D + forward + postprocess/native masks/transfer ส่วน input I/O, RLE, write/load/warmup มี timer และขอบเขตแยก โมเดลที่ส่งกรอบให้ parsing ได้ดีต้องประเมิน parsing downstream เพิ่มในงานถัดไป
"""
    document(STUDY / "RESEARCH_INSIGHTS_TH.md", "# วิเคราะห์ผล YOLO CCTV600 ข้ามขนาด\n\n" + context + "\n\n" + observations + "\n\n[รายงานหลัก](MASTER_RESULTS.md) · [นิยามตัวชี้วัด](METRIC_GUIDE_TH.md)")
    document(STUDY / "MEETING_SUMMARY_TH.md", f"""# สรุปสำหรับนำเสนอ YOLO CCTV600

## 1. ทดสอบอะไร

{context} คะแนนหลักเป็น visible person Box AP ของคนด้านยาวตั้งแต่ 50 px

## 2. ผลรวม 17 โมเดล

![Overall](metrics/plots/box_ap_by_model.png)

## 3. ใครนำแต่ละด้าน

{winner_table(rows)}

## 4. ลดขนาดแล้วต้นทุนเปลี่ยนอย่างไร

![Family scaling](metrics/plots/family_accuracy_scaling.png)

YOLO26x → l เปลี่ยน AP {float(xl['box_ap50_95']['delta_percentage_points']):.3f} pp แลก forward {float(xl['inference_mean_ms']['relative_percent']):.2f}% และ adapter {float(xl['adapter_mean_ms']['relative_percent']):.2f}% ดู 13 คู่เต็มใน SCALING_DELTAS.csv

## 5. ทำไมต้องแยก forward กับ adapter

![Timing](metrics/plots/timing_composition.png)

Forward ต่ำสุดคือ {forward['model']} แต่ adapter ต่ำสุดคือ {adapter['model']} เพราะ timer ครอบคลุมงานหลัง model forward ด้วย ค่าเหล่านี้ใช้ขอบเขตที่ระบุในคู่มือตัวชี้วัด

## 6. ขนาดคนและฉากมีผลอย่างไร

![Size recall](metrics/plots/size_recall_heatmap.png)

ขนาด GT เป็นช่วง 50–99, 100–199 และ ≥200 px มี 178, 342 และ 249 คน การแยกกลุ่มอธิบายว่าคนช่วงใดถูกเก็บได้มากน้อยต่างกัน ส่วนการระบุว่าเกิดจากแสงหรือมุมกล้องต้องมีหลักฐานควบคุมเพิ่ม

## 7. ภาพตัวอย่างช่วยตอบอะไร

รายงานภาพราย tier ใช้สี่กรณีจาก fixed16 เดิม มีทั้งคนถูกตัดขอบ การบดบัง หลายขนาด/ignore และภาพไม่มี eligible GT ตรวจ counts และ matching จาก saved outputs ไม่มี model calls เพิ่มจากงานเอกสารนี้ ภาพอยู่ private พร้อมดัชนีในเครื่อง

## 8. ข้อเสนอการเลือก

พิจารณา YOLOv9e เมื่อเน้น AP/F1, YOLO26x เมื่อเน้น Recall และ YOLO26l เมื่อยอมลด AP เพื่อ adapter time ที่ต่ำลง ตัวเลือกนี้อิง CCTV600 ที่ตรวจรับอยู่ ต้องตรวจข้อกำหนดระบบก่อน deployment

## 9. ข้อจำกัดและสถานะ

ครบคุณภาพและเวลาทั้ง 17 โมเดลแล้ว GT รองรับคะแนนกรอบคน Person Mask AP ยังต้องมี GT silhouette เพิ่ม ยังไม่มี uncertainty หรือข้อสรุปสาเหตุทาง architecture งานนี้จบที่สรุปผลจากหลักฐานเดิม

[รายละเอียดวิจัย](RESEARCH_INSIGHTS_TH.md) · [ผลและ provenance](MASTER_RESULTS.md) · [สูตร metric](METRIC_GUIDE_TH.md)
""")
    readme = f"""# YOLO Instance Segmentation — CCTV600 Scaling Study

**COMPLETE** {context} ประกอบด้วย YOLO26/YOLO11/YOLOv8 อย่างละห้าขนาด และ YOLOv9 ขนาด e/c รวม 17 checkpoints

## อ่านผลตามวัตถุประสงค์

| เอกสาร | ใช้อ่านเรื่องใด |
| --- | --- |
| [MASTER_RESULTS.md](MASTER_RESULTS.md) | ผลเต็ม 17 ตัว คุณภาพ/เวลา/VRAM กลุ่มข้อมูล และประวัติการตรวจปิดรอบ |
| [RESEARCH_INSIGHTS_TH.md](RESEARCH_INSIGHTS_TH.md) | คำถามข้ามขนาด scaling, Pareto, near-tie และการตีความ |
| [EXECUTIVE_SUMMARY_TH.md](EXECUTIVE_SUMMARY_TH.md) | ข้อเสนอการเลือกแบบสั้น |
| [MEETING_SUMMARY_TH.md](MEETING_SUMMARY_TH.md) | ลำดับนำเสนอพร้อมกราฟและข้อจำกัด |
| [METRIC_GUIDE_TH.md](METRIC_GUIDE_TH.md) | สูตร หน่วย denominator ขอบเขตเวลา และ schema ตาราง |
| [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) | เงื่อนไขที่ตรึงก่อนรัน |

## การนำทางทั้ง study

{navigation(tiers)}

## ผู้ได้ผลดีที่สุดตามแต่ละเป้าหมาย

{winner_table(rows)}

![ผลรวม](metrics/plots/box_ap_by_model.png)

## ตารางที่นำไปวิเคราะห์ต่อได้

- [MASTER_17_MODELS.csv](metrics/MASTER_17_MODELS.csv) เก็บค่าความละเอียดเต็มของทุก checkpoint
- [TIER_WINNERS.csv](metrics/TIER_WINNERS.csv) มีเจ็ดเป้าหมายต่อ tier แยก maximize/minimize
- [SCALING_DELTAS.csv](metrics/SCALING_DELTAS.csv) มี 13 คู่ติดกัน × 11 metrics ใช้ smaller−larger
- [PARETO_FRONTIER.csv](metrics/PARETO_FRONTIER.csv), [PARETO_PIPELINE.csv](metrics/PARETO_PIPELINE.csv), [PARETO_VRAM.csv](metrics/PARETO_VRAM.csv) แยกแกนต้นทุน
- [TIMING_COMPOSITION.csv](metrics/TIMING_COMPOSITION.csv), [TIMING_MASTER.csv](metrics/TIMING_MASTER.csv) และ [ACTIVE_600_TIMING.csv](metrics/ACTIVE_600_TIMING.csv) แยก component, distribution และเวลาใน quality pass
- [MODEL_COMPLEXITY_MASTER.csv](metrics/MODEL_COMPLEXITY_MASTER.csv) ให้ parameters, GFLOPs estimate และ checkpoint MB
- [PER_GROUP_RESULTS.csv](metrics/PER_GROUP_RESULTS.csv) ให้ day/night AP และ Recall ตามขนาด GT
- [PLOT_READY_RESULTS.csv](metrics/PLOT_READY_RESULTS.csv), [NEAR_TIES.csv](metrics/NEAR_TIES.csv) และ [NEAR_TIE_RESOURCE_DELTAS.csv](metrics/NEAR_TIE_RESOURCE_DELTAS.csv) รองรับการทำกราฟและอ่านคู่คะแนนใกล้กัน
- [RESULT_INDEX.json](metrics/RESULT_INDEX.json), [CLOSEOUT.json](metrics/CLOSEOUT.json) และ [PRESENTATION_REVISION.json](metrics/PRESENTATION_REVISION.json) ระบุ sources, hashes, membership และการปรับเอกสารครั้งนี้

## กราฟหลัก

![Scaling](metrics/plots/family_accuracy_scaling.png)

![Adjacent scaling changes](metrics/plots/adjacent_scaling_deltas.png)

![Pipeline composition](metrics/plots/timing_composition.png)

![Pareto](metrics/plots/accuracy_pipeline_tradeoff.png)

กราฟ forward/FPS/VRAM scaling, parameters และ day/night/size อยู่ใน [metrics/plots](metrics/plots/) ส่วนแต่ละ tier มีกราฟมาตรฐานหกรูปพร้อมรายงานภาพ

## หลักฐานภาพ

แต่ละ tier มี PRESENTATION_SUMMARY_TH.md อธิบายกราฟ การเลือกกรณี และผลรวมชุดภาพประกอบ มี private comparison gallery สี่ภาพจาก fixed16 เดิมสำหรับเทียบ full frame/ROI และ matching ราย GT ครบทุกโมเดล ภาพ CCTV และข้อมูลระบุภาพคงเก็บในเครื่องตามนโยบายเดิม เปิด Master `outputs/{revision}/index.html` หรืออ่านดัชนีจาก PRESENTATION_REVISION.json

## รูปแบบที่นำมาจาก MOTS20

ตรวจ Master และทั้งห้า tier ของ [MOTS20 study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study) จริง รายการเอกสาร revision และ SHA256 ที่ตรวจอยู่ใน [presentation evidence](metrics/PRESENTATION_REVISION.json) นำบทบาท README/REPORT/RESULTS/PRESENTATION, เจ็ด winners, scaling, Pareto, stage composition และตัวอย่าง matching มาปรับกับข้อมูลชุดนี้

CCTV600 ใช้ Box AP กับ visible bbox ≥50 px และ rectangle-ignore union ส่วน MOTS20 มี GT person silhouettes และ Mask AP จึงต้องระบุชนิด metric และ dataset กำกับ ภาพจริงของ CCTV600 ใช้ private gallery ตามนโยบายเดิม รายละเอียดร่วมและส่วนต่างอยู่ใน METRIC_GUIDE_TH.md

## ตรวจสถานะและสร้างเอกสาร

จาก workspace เดิม

```bash
.venv/bin/python {STUDY.name}/analysis/closeout.py --status
.venv/bin/python {STUDY.name}/analysis/presentation.py --check
```

สร้าง derived presentation ใหม่จากผลเดิม

```bash
.venv/bin/python {STUDY.name}/analysis/presentation.py
```

คำสั่งสุดท้ายอ่าน canonical statistics กับ saved packets เฉพาะสี่ภาพต่อโมเดล เก็บ snapshot เอกสารเดิมใน private outputs และตรวจ hashes ของ accepted results ว่าคงเดิม การรัน `analysis/closeout.py` แบบสร้างรายงานเป็นการสร้างเอกสารรุ่นปิดรอบก่อนจัด presentation ให้ใช้ presentation.py ต่อท้ายหากต้องสร้างใหม่

## ขอบเขตการตีความ

GT inventory อ้างการตรวจรับ 600 ภาพโดยผู้จัดทำ dataset ยังไม่มีการทวนทุกคนอย่างอิสระ ไม่มี uncertainty analysis และ full-person GT silhouette สำหรับ Mask AP ผลเป็น checkpoint/configuration comparison บน CCTV600 โดย pretraining/capacity ต่างกัน Adapter FPS มีขอบเขตตามคู่มือตัวชี้วัด การนำไป deployment ต้องตรวจตามเงื่อนไขระบบปลายทาง
"""
    document(STUDY / "README.md", readme)
    document(STUDY / "EXECUTIVE_SUMMARY_TH.md", f"""# ข้อเสนอการเลือก YOLO จากผล CCTV600

{context}

| เป้าหมาย | ตัวเลือก | หลักฐานและข้อแลกเปลี่ยน |
| --- | --- | --- |
| คุณภาพกรอบและ F1 | {lead['model']} | AP {fmt(lead,'box_ap50_95')}%, F1 {fmt(lead,'f1')}%, FP {lead['fp']}, FN {lead['fn']} |
| เก็บคนให้ได้มากที่ confidence เดิม | {recall['model']} | Recall {fmt(recall,'recall')}%, FP {recall['fp']}, FN {recall['fn']} |
| Forward latency | {forward['model']} | {fmt(forward,'inference_mean_ms')} ms/ภาพ แต่ adapter {fmt(forward,'adapter_mean_ms')} ms/ภาพ |
| Adapter latency | {adapter['model']} | {fmt(adapter,'adapter_mean_ms')} ms/ภาพ, AP {fmt(adapter,'box_ap50_95')}% |
| Allocated VRAM | {memory['model']} | {fmt(memory,'vram_allocated_mib')} MiB, AP {fmt(memory,'box_ap50_95')}% |
| AP กับ adapter budget | yolo26l-seg.pt | AP 65.099%, adapter 59.770 ms/ภาพ อยู่บน AP–adapter Pareto |

การให้โมเดลตรวจคนส่งเข้า parsing ต้องพิจารณาว่ายอมรับ FN/FP เท่าไร การชนะ AP, Recall และเวลาตอบคำถามคนละด้าน ยังไม่มี weighted score ที่กำหนดผู้ชนะทุกข้อจำกัด

คะแนนนี้วัด visible bbox ≥50 px ของ GT ที่ผู้จัดทำ dataset ตรวจรับ Person Mask AP ยังต้องมี GT silhouette เพิ่ม ยังไม่มี uncertainty analysis หรือข้อรับรอง deployment ค่า timing ครอบคลุม adapter และมี I/O/RLE/load/warmup แยกออกไป

[ผลเต็ม](MASTER_RESULTS.md) · [วิเคราะห์ข้ามขนาด](RESEARCH_INSIGHTS_TH.md) · [เนื้อหานำเสนอ](MEETING_SUMMARY_TH.md) · [สูตร metric](METRIC_GUIDE_TH.md)
""")


def validate(protected=None):
    revision = load(STUDY / "metrics/PRESENTATION_REVISION.json")
    protected = protected or revision["protected_before"]
    for path, digest in protected.items():
        assert sha(WORKSPACE / path) == digest, path
    for path, digest in revision["document_and_derived_hashes"].items():
        assert sha(WORKSPACE / path) == digest, path
    for source in revision["mots20_reference_documents"]:
        path = WORKSPACE / source["repository"] / source["file"]
        assert sha(path) == source["sha256"]
    tiers = load(STUDY / "configs/study.json")["tiers"]
    for repo in [STUDY] + [WORKSPACE / t["repository"] for t in tiers]:
        for path in repo.glob("*.md"):
            if path.name in ["AGENTS.md", "EXPERIMENT_PROTOCOL.md"]:
                continue
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                if target.startswith(("https://", "http://", "#")):
                    continue
                assert (path.parent / target.split("#")[0]).exists(), (path, target)
        subprocess.run(["git", "diff", "--check"], cwd=repo, check=True)
    winners = read_csv(STUDY / "metrics/TIER_WINNERS.csv")
    assert len(winners) == 35
    # These fixed expected names were independently checked against the accepted
    # results. They catch reversed cost objectives even if the builder agrees
    # with its own repeated calculations.
    expected_models = {
        "largest": ["yolov9e", "yolov9e", "yolo26x", "yolov9e", "yolov9e", "yolov9e", "yolo26x"],
        "second_largest": ["yolo26l", "yolo26l", "yolo11l", "yolo11l", "yolo11l", "yolo11l", "yolov9c"],
        "medium": ["yolov8m", "yolov8m", "yolov8m", "yolo26m", "yolo26m", "yolo26m", "yolo26m"],
        "small": ["yolo26s", "yolo26s", "yolo26s", "yolov8s", "yolo11s", "yolo11s", "yolo11s"],
        "nano": ["yolo26n", "yolo26n", "yolov8n", "yolov8n", "yolo11n", "yolo11n", "yolo11n"]}
    for tier, names in expected_models.items():
        assert [r["model"] for r in winners if r["tier"] == tier] == [name+"-seg.pt" for name in names]
    raw = load(STUDY / "metrics/MASTER_RESULTS.json")["models"]
    expected = [item for tier in tiers for item in winner_rows([row for row in raw if row["tier"] == tier["id"]])]
    for a,b in zip(winners, expected):
        assert a["model"] == b["model"] and float(a["value"]) == b["value"]
    assert len(read_csv(STUDY / "metrics/TIMING_COMPOSITION.csv")) == 17
    assert len(read_csv(STUDY / "metrics/PLOT_READY_RESULTS.csv")) == 17
    assert len(read_csv(STUDY / "metrics/NEAR_TIE_RESOURCE_DELTAS.csv")) == 7
    gallery = STUDY / revision["private_gallery"]
    cases = load(gallery.parent / "CASE_AUDIT.json")["tiers"]
    assert sum(len(item["cases"]) for item in cases) == 20
    assert len({case["image_id"] for item in cases for case in item["cases"]}) == 5
    for item in cases:
        for case in item["cases"]:
            image = gallery.parent / "cases" / item["tier"] / case["file"]
            assert sha(image) == case["sha256"]
            with Image.open(image) as content:
                assert content.size == (1760, 553*(len(case["models"])+1))
                content.verify()
            for model in case["models"]:
                assert model["counts"]["tp"] + model["counts"]["fn"] == case["gt_people"]
    print("PASS: MOTS20 reference hashes, protected measurements, document hashes/links, 35 winners, 17 timing compositions, 17 plot rows, near-tie deltas")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        validate()
        return
    run = Path(load(STUDY / "outputs/CURRENT.json")["run_dir"])
    prepared = load(run / "PREPARED.json")
    verify_source(prepared)
    assert load(STUDY / "metrics/CLOSEOUT.json")["state"] == "COMPLETE"
    rows = load(STUDY / "metrics/MASTER_RESULTS.json")["models"]
    assert len(rows) == 17 and all(row["timing_state"] == "COMPLETE" for row in rows)
    groups = [group for tier in prepared["config"]["tiers"] for item in load(WORKSPACE / tier["repository"] / "metrics/RESULTS.json")["models"]
        for group in [{"model": item["model"], **g} for g in item["groups"]]]
    protected = protected_files(prepared, run)
    revision_id = "presentation-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = STUDY / "outputs" / revision_id
    output.mkdir()
    tiers = prepared["config"]["tiers"]
    repositories = [STUDY] + [WORKSPACE / tier["repository"] for tier in tiers]
    for repo in repositories:
        for path in list(repo.glob("*.md")) + list((repo / "metrics").rglob("*")):
            if path.is_file():
                dest = output / "archive" / path.relative_to(WORKSPACE)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
    links = []
    all_cases = []
    for tier in tiers:
        selected = [row for row in rows if row["tier"] == tier["id"]]
        cases = render_cases(prepared, tier, output / "cases" / tier["id"])
        all_cases.append({"tier": tier["id"], "cases": cases})
        plots(WORKSPACE / tier["repository"], selected)
        tier_documents(tier, tiers, selected, groups, cases, revision_id)
        links.append("<li><a href='cases/"+tier["id"]+"/index.html'>"+TITLES[tier["id"]]+"</a></li>")
    document(output / "index.html", "<meta charset='utf-8'><title>CCTV600 diagnostic comparison</title><h1>Four fixed cases · five tiers</h1><ul>"+"".join(links)+"</ul>")
    derived_outputs(rows, tiers, load(STUDY / "metrics/RESULT_INDEX.json"))
    master_figures(rows, tiers)
    master_documents(rows, tiers, groups, revision_id)
    main_doc = STUDY / "MASTER_RESULTS.md"
    section = "\n\n## รูปแบบรายงานและการตรวจเทียบ MOTS20\n\nตรวจ Master และทั้งห้า tier ของ MOTS20 แล้วเพิ่มเติมเจ็ด winners รายขนาด ตารางบทบาทโมเดล รายงานเทคนิค กราฟมาตรฐานหกรูปต่อ tier และรายงานภาพจากสี่กรณีใน fixed16 เดิม ภาพจริงอยู่ private พร้อม matching ราย GT ที่ตรวจ counts ตรงกับผลเดิม รายงานรอบนี้ใช้ข้อมูลเดิมและไม่เพิ่ม model calls\n\n[วิเคราะห์ข้ามขนาด](RESEARCH_INSIGHTS_TH.md) · [ลำดับนำเสนอ](MEETING_SUMMARY_TH.md) · [รายการอ้างอิงและการตรวจเอกสาร](metrics/PRESENTATION_REVISION.json)\n"
    if "## รูปแบบรายงานและการตรวจเทียบ MOTS20" not in main_doc.read_text():
        document(main_doc, main_doc.read_text().rstrip()+section)
    guide = STUDY / "METRIC_GUIDE_TH.md"
    text = guide.read_text().replace("ผู้ได้ค่าสูงสุดแต่ละ metric ในแต่ละ tier รวม ties ถ้ามี", "เจ็ดเป้าหมายต่อ tier แยกทิศทาง max/min สำหรับคะแนน เวลา FPS และ allocated VRAM")
    if "## 10. ตารางประกอบการนำเสนอ" not in text:
        text += "\n## 10. ตารางประกอบการนำเสนอ\n\nTIMING_COMPOSITION.csv คง stage means เดิมและเพิ่มสัดส่วน preprocess/forward/postprocess ต่อ adapter total ส่วน PLOT_READY_RESULTS.csv เพิ่ม family, display order และ Pareto flags โดยคะแนน/เวลาคงเดิม NEAR_TIE_RESOURCE_DELTAS.csv ใช้ B−A และหาร A สำหรับร้อยละสัมพัทธ์ ทิศทางนี้ต่างจาก SCALING_DELTAS.csv ที่ใช้ smaller−larger\n\nTIER_WINNERS.csv มีเจ็ดเป้าหมายในแต่ละ tier ได้แก่ Box AP, AP75, Recall, forward time, adapter time, FPS และ allocated VRAM คะแนน/FPS ใช้ค่าสูงสุด ส่วนเวลา/VRAM ใช้ค่าต่ำสุด เก็บ exact ties ทุกตัวและเก็บค่าคะแนนต้นฉบับเป็น fraction\n"
    document(guide, text)
    for repo in repositories:
        status = repo / "STATUS.md"
        addition = "\n\n## รายงานหลังตรวจเทียบ MOTS20\n\nตรวจโครง Master และห้า tier ของ MOTS20 พร้อมเทียบไฟล์ต้นฉบับ GitHub แล้ว เพิ่มรายงานตามบทบาท กราฟมาตรฐาน และหลักฐานภาพจาก saved outputs โดยคะแนนและเวลาเดิมคงเดิม\n\n"
        addition += ("[วิเคราะห์ผล](RESEARCH_INSIGHTS_TH.md) · [ลำดับนำเสนอ](MEETING_SUMMARY_TH.md) · [revision evidence](metrics/PRESENTATION_REVISION.json)\n"
            if repo == STUDY else "[สรุปผล](RESULTS_SUMMARY_TH.md) · [รายงานภาพ](PRESENTATION_SUMMARY_TH.md) · [รายงานเทคนิค](REPORT.md)\n")
        if "## รายงานหลังตรวจเทียบ MOTS20" not in status.read_text():
            document(status, status.read_text().rstrip()+addition)
    generated = [path for repo in repositories for path in repo.glob("*.md") if path.name not in ["AGENTS.md", "EXPERIMENT_PROTOCOL.md"]]
    generated.extend(STUDY / "metrics" / name for name in ["TIER_WINNERS.csv", "TIMING_COMPOSITION.csv", "PLOT_READY_RESULTS.csv", "NEAR_TIE_RESOURCE_DELTAS.csv"])
    generated.extend(path for repo in repositories for path in (repo / "metrics/plots").glob("*.png"))
    evidence = {"state": "COMPLETE", "at_utc": now(), "revision_id": revision_id,
        "authoritative_quality_and_timing": "metrics/RESULT_INDEX.json and metrics/CLOSEOUT.json",
        "model_calls": 0, "timing_calls": 0, "full_population_rescores": 0,
        "saved_packets_distinct_read": 68,
        "source_images_distinct": len({case["image_id"] for group in all_cases for case in group["cases"]}), "case_slots": 20,
        "selection": "Post-study diagnostics in prior fixed16: 3/13/15/16; Small uses 3/9/15/16. Three shared anchors plus discriminating Small comparison. Full600 scores/settings unchanged",
        "gallery_publication": "PRIVATE_LOCAL_ONLY", "private_gallery": "outputs/"+revision_id+"/index.html",
        "protected_before": protected, "protected_after": protected_files(prepared, run),
        "mots20_reference_documents": reference_audit(),
        "github_reference_checks": remote_reference_checks(),
        "report_generator_sha256": sha(Path(__file__)),
        "document_and_derived_hashes": {str(p.relative_to(WORKSPACE)):sha(p) for p in generated},
        "adaptations": ["Person Box AP with visible GT >=50px and rectangle ignore replaces silhouette Mask AP",
            "Day/night and GT-size groups replace MOTS sequences", "Private CCTV images remain local",
            "Thai technical explanations retained in existing Master report; English per-tier REPORT follows MOTS20 role"],
        "equivalent_reporting_sections": ["Landing page results/navigation", "Seven tier winners", "Per-model roles/trade-offs",
            "Technical compatibility/results/group/timing/warnings/provenance report", "Six standard tier charts",
            "Four full-frame/same-ROI saved-output cases with per-GT matching", "Cross-tier research interpretation", "Reusable meeting narrative",
            "Timing composition", "Near-tie resource deltas", "Machine-readable plotting table"]}
    assert evidence["protected_before"] == evidence["protected_after"]
    write(STUDY / "metrics/PRESENTATION_REVISION.json", evidence)
    write(output / "CASE_AUDIT.json", {"tiers": all_cases})
    validate(protected)
    print("Private gallery", output / "index.html")


if __name__ == "__main__":
    main()
