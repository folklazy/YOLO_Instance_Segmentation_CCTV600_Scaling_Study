"""Private fixed-sample gallery and aggregate-only public figures."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask

from .common import STUDY, WORKSPACE, load, write
from .packets import read_packet
from .reporting import model_directory, model_binding, csv_write


def gallery(prepared, tier, limit=None):
    frames = {frame["image_id"]: frame for frame in prepared["frames"]}
    output = WORKSPACE / tier["repository"] / "outputs" / prepared["run_id"] / "gallery"
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for identifier in prepared["visual_sample"][:limit]:
        frame = frames[identifier]
        source = Image.open(frame["image_path"]).convert("RGB")
        panels = []
        gt = source.copy()
        draw = ImageDraw.Draw(gt)
        for person in frame["people"]:
            draw.rectangle(person["bbox"], outline=(0, 180, 80), width=3)
        for box in frame["ignore"]:
            draw.rectangle(box, outline=(160, 160, 160), width=2)
        panels.append(("GT visible boxes / ignore", gt))
        used = []
        for model in tier["models"]:
            packet = model_directory(prepared, tier, model) / "predictions" / (identifier + ".json.gz")
            if not packet.exists():
                continue
            record = read_packet(packet, frame, model_binding(prepared, model))
            image = np.asarray(source).copy()
            for prediction in record["predictions"]:
                if prediction["confidence"] < prepared["config"]["fixed_confidence"]:
                    continue
                rle = prediction["rle"]
                mask = coco_mask.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")}).astype(bool)
                image[mask] = (image[mask] * .65 + np.array([0, 160, 240]) * .35).astype(np.uint8)
            panel = Image.fromarray(image)
            draw = ImageDraw.Draw(panel)
            for prediction in record["predictions"]:
                if prediction["confidence"] >= prepared["config"]["fixed_confidence"]:
                    draw.rectangle(prediction["bbox_xyxy"], outline=(0, 160, 240), width=3)
                    draw.text(tuple(prediction["bbox_xyxy"][:2]), f"{prediction['confidence']:.3f}", fill="white", stroke_width=1, stroke_fill="black")
            panels.append((model, panel))
            used.append(model)
        if len(panels) == 1:
            continue
        width = 720
        height = round(source.height * width / source.width)
        canvas = Image.new("RGB", (width * len(panels), height + 36), "white")
        for index, (title, panel) in enumerate(panels):
            canvas.paste(panel.resize((width, height)), (index * width, 36))
            ImageDraw.Draw(canvas).text((index * width + 8, 10), title, fill="black")
        target = output / (identifier + ".jpg")
        canvas.save(target, quality=94)
        manifest.append({"image_id": identifier, "models": used, "file": target.name})
    write(output / "MANIFEST.json", {"fixed_sample": prepared["visual_sample"], "panels": manifest,
        "source_binding": prepared["scientific_binding"], "note": "Predicted native person masks shown as diagnostic; GT has bbox only for this task"})
    html = "<meta charset='utf-8'><title>CCTV600 fixed comparison</title><style>body{font-family:sans-serif}img{max-width:100%}</style>"
    for row in manifest:
        html += "<h2>" + row["image_id"] + "</h2><img src='" + row["file"] + "'>"
    (output / "index.html").write_text(html)
    return output


def plots(rows):
    complete = [row for row in rows if row["state"] == "COMPLETE"]
    if len(complete) != 17:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = STUDY / "metrics/plots"
    output.mkdir(parents=True, exist_ok=True)
    names = [row["model"].removesuffix("-seg.pt") for row in rows]
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.bar(names, [row["box_ap50_95"] * 100 for row in rows], color="#187e9d")
    axis.set_ylabel("Person Box AP@[.50:.95] (%)")
    axis.tick_params(axis="x", rotation=65)
    fig.tight_layout()
    fig.savefig(output / "box_ap_by_model.png", dpi=150)
    plt.close(fig)
    for tier in load(STUDY / "configs/study.json")["tiers"]:
        results = load(WORKSPACE / tier["repository"] / "metrics/RESULTS.json")["models"]
        for kind in ["pr", "f1"]:
            fig, axis = plt.subplots(figsize=(8, 5))
            for result in results:
                if kind == "pr":
                    curve = result["primary"]["pr_curve_iou50"]
                    axis.plot(curve["recall"], curve["precision"], label=result["model"])
                else:
                    curve = result["confidence_curve"]
                    axis.plot([r["confidence"] for r in curve], [r["f1"] for r in curve], label=result["model"])
            axis.set_xlabel("Recall" if kind == "pr" else "Confidence threshold")
            axis.set_ylabel("Precision at Box IoU0.50" if kind == "pr" else "Fixed-matcher F1")
            axis.set_ylim(0, 1.01)
            axis.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(output / (tier["id"] + "_" + kind + "_curve.png"), dpi=150)
            plt.close(fig)
    if not all(row["timing_state"] == "COMPLETE" for row in rows):
        return
    fig, axis = plt.subplots(figsize=(9, 6))
    for row in rows:
        axis.scatter(row["inference_mean_ms"], row["box_ap50_95"] * 100)
        axis.annotate(row["model"].removesuffix("-seg.pt"), (row["inference_mean_ms"], row["box_ap50_95"] * 100), fontsize=8)
    axis.set_xlabel("Clean forward mean (ms/image)")
    axis.set_ylabel("Person Box AP@[.50:.95] (%)")
    fig.tight_layout()
    fig.savefig(output / "accuracy_forward_tradeoff.png", dpi=150)
    plt.close(fig)
    frontier = []
    for row in rows:
        dominated = any(other["box_ap50_95"] >= row["box_ap50_95"] and
            other["inference_mean_ms"] <= row["inference_mean_ms"] and
            (other["box_ap50_95"] > row["box_ap50_95"] or other["inference_mean_ms"] < row["inference_mean_ms"])
            for other in rows)
        frontier.append({"model": row["model"], "box_ap50_95": row["box_ap50_95"],
                         "inference_mean_ms": row["inference_mean_ms"], "nondominated": not dominated})
    csv_write(STUDY / "metrics/PARETO_FRONTIER.csv", frontier)
    summaries = {}
    for tier in load(STUDY / "configs/study.json")["tiers"]:
        for item in load(WORKSPACE / tier["repository"] / "metrics/RESULTS.json")["models"]:
            summaries[item["model"]] = item["timing"]["primary"]
    fig, axis = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(rows))
    for key, label in [("preprocess_ms", "Preprocess/H2D"), ("inference_ms", "Forward"), ("postprocess_ms", "Postprocess/native masks")]:
        values = np.array([summaries[row["model"]][key]["mean"] for row in rows])
        axis.bar(names, values, bottom=bottom, label=label)
        bottom += values
    axis.set_ylabel("Adapter time (ms/image)")
    axis.tick_params(axis="x", rotation=65)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "timing_composition.png", dpi=150)
    plt.close(fig)
