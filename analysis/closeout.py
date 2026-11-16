"""Derive public closeout tables from accepted evidence without model calls.

Run from the workspace using .venv/bin/python <study>/analysis/closeout.py.
This post-study reporter lives outside the frozen execution-source namespace.
It preserves canonical predictions, metrics, audits and timing attempts.
"""
import csv
from collections import Counter
from datetime import datetime
from decimal import Decimal, localcontext
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

STUDY = Path(__file__).resolve().parents[1]
WORKSPACE = STUDY.parent
sys.path.insert(0, str(STUDY / "src"))

import numpy as np

from cctv600.common import load, now, sha, verify_source, write
from cctv600.reporting import model_binding, model_directory, statistics


def csv_save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def link_path(path):
    return str(path.relative_to(WORKSPACE))


def percentage(value):
    return "—" if value is None else f"{value * 100:.3f}"


def number(value):
    return "—" if value is None else f"{value:.3f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"] +
        ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def pareto(rows, cost):
    eligible = [row for row in rows if row[cost] is not None]
    return [{"model": row["model"], "box_ap50_95": row["box_ap50_95"],
        cost: row[cost], "comparison_models": len(eligible),
        "nondominated": not any(other["box_ap50_95"] >= row["box_ap50_95"]
            and other[cost] <= row[cost]
            and (other["box_ap50_95"] > row["box_ap50_95"] or other[cost] < row[cost])
            for other in eligible)} for row in eligible]


def snapshot(repositories, target):
    """Retain the public result snapshot before regenerating derived documents."""
    for repo in repositories:
        for path in list(repo.glob("*.md")) + list((repo / "metrics").rglob("*")):
            if path.is_file():
                dest = target / link_path(path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)


def collect(prepared, run):
    expected = {frame["image_id"] for frame in prepared["frames"]}
    assert len(expected) == prepared["config"]["expected_images"] == 600
    assert len(set(prepared["sample"])) == 100
    gt_people = sum(len(frame["people"]) for frame in prepared["frames"])
    acceptance = load(STUDY / "manifests/CPU_ACCEPTANCE.json")
    assert acceptance["state"] == "PASS" and acceptance["sources"] == prepared["sources"]
    verify_source(prepared)
    continuation = None
    pointer = run / "closeout/TIMING_CONTINUATION.json"
    if pointer.exists():
        segment = Path(load(pointer)["segment_dir"])
        if (segment / "COMPLETION.json").exists():
            completion = load(segment / "COMPLETION.json")
            manifest = load(segment / "MANIFEST.json")
            assert completion["state"] == "COMPLETE"
            assert completion["parent_run_id"] == prepared["run_id"]
            assert manifest["parent_prepared_sha256"] == sha(run / "PREPARED.json")
            assert completion["scientific_binding"] == prepared["scientific_binding"]
            local = load(segment / "PREPARED.json")
            assert {k: v for k, v in local.items() if k != "run_id"} == {k: v for k, v in prepared.items() if k != "run_id"}
            assert len(completion["completed"]) == len(manifest["pending"]) == 4
            continuation = {(item["model"], item["round"]): item for item in completion["completed"]}
    rows, canonical, groups, active, timings, ledger, caps = [], [], [], [], [], [], []
    for path in sorted((run / "attempts").glob("*/ledger.jsonl")):
        for line in path.read_text().splitlines():
            ledger.append(json.loads(line))
    events = Counter((item["model"], item["event"]) for item in ledger)
    for tier in prepared["config"]["tiers"]:
        for model in tier["models"]:
            directory = model_directory(prepared, tier, model)
            metrics_path = directory / "METRICS.json"
            metrics = load(metrics_path)
            audit_path = directory / "PACKET_AUDIT.json"
            audit = load(audit_path)
            binding = model_binding(prepared, model)
            assert metrics["state"] == "COMPLETE" and audit["state"] == "PASS"
            assert metrics["binding"] == audit["binding"] == binding
            assert metrics["membership"] == audit["membership"] == {
                "expected": 600, "completed": 600, "missing": 0,
                "unexpected": 0, "duplicate": 0, "invalid": 0}
            paths = {p.name.removesuffix(".json.gz"): p for p in (directory / "predictions").glob("*.json.gz")}
            assert paths.keys() == expected == audit["packet_sha256"].keys()
            # Reuse accepted packet hashes. Detect post-audit changes by filesystem
            # timestamps, then verify only those potentially changed packets.
            audit_time = audit_path.stat().st_mtime_ns
            changed = [p for p in paths.values() if max(p.stat().st_mtime_ns, p.stat().st_ctime_ns) > audit_time]
            for path in changed:
                assert sha(path) == audit["packet_sha256"][path.name.removesuffix(".json.gz")]
            for identifier, digest in audit["packet_sha256"].items():
                receipt = load(directory / "receipts" / (identifier + ".json"))
                assert receipt["packet_sha256"] == digest
                assert receipt["binding"] == binding
            with (directory / "PER_IMAGE.csv").open(newline="") as stream:
                per_image = list(csv.DictReader(stream))
            assert len(per_image) == len({item["image_id"] for item in per_image}) == 600
            assert {item["image_id"] for item in per_image} == expected
            primary = metrics["primary"]
            assert primary["gt_people"] == gt_people == primary["tp"] + primary["fn"]
            for key in ["tp", "fp", "fn", "gt_people"]:
                assert sum(int(item[key]) for item in per_image) == primary[key]
            assert events[model, "ACCEPTED_PACKET"] == 600
            assert events[model, "STARTED_FRAME"] == 600
            for key, expected_value in {
                "precision": primary["tp"] / (primary["tp"] + primary["fp"]),
                "recall": primary["tp"] / gt_people,
                "f1": 2 * primary["tp"] / (2 * primary["tp"] + primary["fp"] + primary["fn"])
            }.items():
                assert abs(primary[key] - expected_value) < 1e-12
            cap_path = directory / "CAP_CHECK.json"
            cap = load(cap_path)
            assert cap["state"] == "PASS" and cap["binding"] == binding
            assert cap["sample_membership"] == prepared["sample"]
            assert max(cap["absolute_differences"].values()) < prepared["config"]["ap_cap_tolerance"]
            caps.append({"model": model, "state": cap["state"],
                "maxdet": cap["maxdet"], "reference_maxdet": cap["reference_maxdet"],
                **cap["absolute_differences"]})
            accepted, round_records, original_round_records = [], [], []
            for round_number in range(1, 4):
                path = directory / "timing" / f"round{round_number}.json"
                record = load(path) if path.exists() else None
                original_round_records.append({"round": round_number,
                    "state": record["state"] if record else "NOT_ATTEMPTED",
                    "path": link_path(path) if record else None,
                    "sha256": sha(path) if record else None})
                reused = continuation.get((model, round_number)) if continuation else None
                if reused:
                    assert record is None or record["state"] != "CLEAN_COMPLETE"
                    assert reused["original_sha256"] == (sha(path) if record else None)
                    path = Path(reused["path"])
                    assert sha(path) == reused["sha256"]
                    record = load(path)
                    assert record["state"] == "CLEAN_COMPLETE"
                if record is not None:
                    assert record["binding"] == binding
                    assert record["round"] == round_number
                    if record["state"] == "CLEAN_COMPLETE":
                        assert [item["image_id"] for item in record["observations"]] == prepared["sample"]
                        assert record["warmup_calls"] == 10
                        assert all(not item["external_processes"] for item in record["gpu_samples"])
                        assert record["gpu_samples"][0]["utilization_percent"] <= 5
                        accepted.append(record)
                round_records.append({"round": round_number,
                    "state": record["state"] if record else "NOT_ATTEMPTED",
                    "path": link_path(path) if record else None,
                    "sha256": sha(path) if record else None,
                    "measured_calls": len(record["observations"]) if record else 0,
                    "warmup_calls": record["warmup_calls"] if record else 0})
            timing = None
            if len(accepted) == 3:
                observations = [item for record in accepted for item in record["observations"]]
                timing = {key: statistics([item[key] for item in observations])
                    for key in observations[0] if key.endswith("_ms")}
                for item in observations:
                    assert abs(item["preprocess_ms"] + item["inference_ms"] + item["postprocess_ms"] - item["total_ms"]) < 1e-5
                timing["fps"] = 1000 / timing["total_ms"]["mean"]
                for key in ["peak_gpu_allocated_mib", "peak_gpu_reserved_mib"]:
                    timing[key] = max(record["memory"][key] for record in accepted)
                timing["model_loading_seconds"] = statistics([record["load_seconds"] for record in accepted])
                for key, value in timing.items():
                    if isinstance(value, dict):
                        timings.append({"model": model, "scope": "CLEAN_3x100",
                            "stage": key, "observations": 300 if key.endswith("_ms") else 3, **value})
            metadata = load(sorted(directory.glob("MODEL_*.json"))[0])
            row = {"tier": tier["id"], "model": model, "state": "COMPLETE", "images": 600,
                **{key: primary[key] for key in primary if key != "pr_curve_iou50"},
                "timing_state": "COMPLETE" if timing else "INCOMPLETE",
                "clean_rounds": len(accepted),
                "inference_mean_ms": timing["inference_ms"]["mean"] if timing else None,
                "adapter_mean_ms": timing["total_ms"]["mean"] if timing else None,
                "fps": timing["fps"] if timing else None,
                "vram_allocated_mib": timing["peak_gpu_allocated_mib"] if timing else None,
                "vram_reserved_mib": timing["peak_gpu_reserved_mib"] if timing else None,
                "checkpoint_mb": prepared["checkpoints"][model]["bytes"] / 1e6,
                "parameters": metadata["checkpoint_parameters"],
                "gflops_estimate": prepared["checkpoints"][model]["reference_gflops"]}
            assert row["parameters"] == prepared["checkpoints"][model]["reference_parameters"]
            rows.append(row)
            for group in metrics["groups"]:
                groups.append({"model": model, **{key: value for key, value in group.items() if key != "pr_curve_iou50"}})
            for key, total in metrics["active_timing"].items():
                if key.endswith("_seconds"):
                    coverage = metrics["active_timing"][key.removesuffix("_seconds") + "_coverage"]
                    active.append({"model": model, "stage": key.removesuffix("_seconds"),
                        "images_with_timer": coverage, "expected_images": 600,
                        "seconds_sum": total, "mean_ms_per_image": total * 1000 / 600 if total is not None else None})
            canonical.append({"model": model, "quality_state": "COMPLETE",
                "membership": metrics["membership"], "metrics_path": link_path(metrics_path),
                "metrics_sha256": sha(metrics_path), "packet_audit_path": link_path(audit_path),
                "packet_audit_sha256": sha(audit_path), "per_image_sha256": sha(directory / "PER_IMAGE.csv"),
                "cap_check_sha256": sha(cap_path), "binding": binding,
                "post_audit_packets_rehashed": len(changed), "receipt_hash_bindings_checked": 600,
                "quality_primary_source": "METRICS.json primary",
                "timing_state": row["timing_state"], "timing_primary": timing,
                "timing_rounds": round_records, "original_timing_rounds": original_round_records})
    return rows, canonical, groups, active, timings, caps, Counter(item["event"] for item in ledger)


def derived_tables(rows, groups, active, timings, caps):
    output = STUDY / "metrics"
    csv_save(output / "MASTER_17_MODELS.csv", rows)
    csv_save(output / "PER_GROUP_RESULTS.csv", groups)
    csv_save(output / "ACTIVE_600_TIMING.csv", active)
    csv_save(output / "TIMING_MASTER.csv", timings)
    csv_save(output / "PREFLIGHT_MAXDET.csv", caps)
    csv_save(output / "MODEL_COMPLEXITY_MASTER.csv", [{key: row[key] for key in
        ["model", "tier", "checkpoint_mb", "parameters", "gflops_estimate"]} for row in rows])
    winners = []
    for tier in dict.fromkeys(row["tier"] for row in rows):
        selected = [row for row in rows if row["tier"] == tier]
        for metric in ["box_ap50_95", "box_ap50", "box_ap75", "precision", "recall", "f1"]:
            best = max(row[metric] for row in selected)
            winners.extend({"tier": tier, "metric": metric, "model": row["model"], "value": best}
                for row in selected if row[metric] == best)
    csv_save(output / "TIER_WINNERS.csv", winners)
    deltas = []
    for prefix in ["yolo26", "yolo11", "yolov9", "yolov8"]:
        family = [row for row in rows if row["model"].startswith(prefix)]
        for larger, smaller in zip(family, family[1:]):
            for key in ["box_ap50_95", "box_ap50", "box_ap75", "precision", "recall", "f1",
                        "inference_mean_ms", "adapter_mean_ms", "vram_allocated_mib", "parameters", "gflops_estimate"]:
                left, right = larger[key], smaller[key]
                with localcontext() as ctx:
                    ctx.prec = 50
                    delta = Decimal(str(right)) - Decimal(str(left)) if left is not None and right is not None else None
                    relative = delta / Decimal(str(left)) * 100 if delta is not None and left else None
                deltas.append({"family": prefix, "larger": larger["model"], "smaller": smaller["model"],
                    "metric": key, "larger_value": left, "smaller_value": right,
                    "delta_smaller_minus_larger": delta, "relative_percent": relative,
                    "delta_percentage_points": delta * 100 if delta is not None and key in
                        ["box_ap50_95", "box_ap50", "box_ap75", "precision", "recall", "f1"] else None})
    csv_save(output / "SCALING_DELTAS.csv", deltas)
    near = []
    for a, b in itertools.combinations(rows, 2):
        delta = b["box_ap50_95"] - a["box_ap50_95"]
        if abs(delta) <= .001:
            near.append({"model_a": a["model"], "model_b": b["model"],
                "ap_a": a["box_ap50_95"], "ap_b": b["box_ap50_95"], "delta_b_minus_a_pp": delta * 100,
                "screen_absolute_ap_fraction": .001})
    csv_save(output / "NEAR_TIES.csv", near)
    for name, cost in [("PARETO_FRONTIER", "inference_mean_ms"),
                       ("PARETO_PIPELINE", "adapter_mean_ms"), ("PARETO_VRAM", "vram_allocated_mib")]:
        csv_save(output / (name + ".csv"), pareto(rows, cost))
    return deltas, near


def plots(rows, groups, canonical):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .15})
    output = STUDY / "metrics/plots"
    output.mkdir(exist_ok=True)
    names = [row["model"].removesuffix("-seg.pt") for row in rows]
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(names, [row["box_ap50_95"] * 100 for row in rows], color="#187e9d")
    ax.set_ylabel("Person Box AP@[.50:.95] (%)")
    ax.tick_params(axis="x", rotation=65)
    save(fig, "box_ap_by_model.png")
    for metric, filename, ylabel in [
        ("box_ap50_95", "family_accuracy_scaling.png", "Person Box AP (%)"),
        ("inference_mean_ms", "family_forward_scaling.png", "Clean forward mean (ms/image)"),
        ("fps", "family_fps_scaling.png", "Clean adapter FPS"),
        ("vram_allocated_mib", "family_vram_scaling.png", "Peak allocated VRAM (MiB)")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for prefix in ["yolo26", "yolo11", "yolov9", "yolov8"]:
            family = [row for row in rows if row["model"].startswith(prefix)]
            ax.plot([list(dict.fromkeys(row["tier"] for row in rows)).index(row["tier"]) for row in family],
                [row[metric] * (100 if metric == "box_ap50_95" else 1) if row[metric] is not None else np.nan for row in family],
                "o-", label=prefix)
        ax.set_xticks(range(5), ["X/E", "L/C", "M", "S", "N"])
        ax.set_ylabel(ylabel)
        ax.legend()
        if metric != "box_ap50_95":
            ax.set_title(f"{sum(row['timing_state'] == 'COMPLETE' for row in rows)}/17 complete clean timings")
        save(fig, filename)
    for cost, filename, xlabel in [
        ("parameters", "accuracy_parameters.png", "Unfused checkpoint parameters"),
        ("inference_mean_ms", "accuracy_forward_tradeoff.png", "Clean forward mean (ms/image)"),
        ("adapter_mean_ms", "accuracy_pipeline_tradeoff.png", "Clean adapter mean (ms/image)"),
        ("vram_allocated_mib", "accuracy_vram_tradeoff.png", "Peak allocated VRAM (MiB)")]:
        fig, ax = plt.subplots(figsize=(11, 6))
        accepted = [row for row in rows if row[cost] is not None]
        frontier = {row["model"] for row in pareto(rows, cost) if row["nondominated"]}
        for row in accepted:
            ax.scatter(row[cost], row["box_ap50_95"] * 100, color="#d86924" if row["model"] in frontier else "#187e9d")
            ax.annotate(row["model"].removesuffix("-seg.pt"), (row[cost], row["box_ap50_95"] * 100),
                fontsize=8, xytext=(3, 4), textcoords="offset points")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Person Box AP (%)")
        ax.set_title(f"{len(accepted)}/17 eligible; orange = nondominated within eligible models")
        save(fig, filename)
    complete = [(row, item["timing_primary"]) for row, item in zip(rows, canonical) if item["timing_primary"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(complete))
    for key, label in [("preprocess_ms", "Preprocess/H2D"), ("inference_ms", "Forward"), ("postprocess_ms", "Postprocess/native masks")]:
        values = np.array([timing[key]["mean"] for _, timing in complete])
        ax.bar([row["model"].removesuffix("-seg.pt") for row, _ in complete], values, bottom=bottom, label=label)
        bottom += values
    ax.set_ylabel("Clean adapter mean (ms/image)")
    ax.set_title(f"{len(complete)}/17 models with complete clean timing")
    ax.tick_params(axis="x", rotation=65)
    ax.legend()
    save(fig, "timing_composition.png")
    for kind, selected_groups, key, filename in [
        ("metadata_day_night", ["day", "night"], "box_ap50_95", "day_night_ap_heatmap.png"),
        ("gt_long_side_recall", ["50to99", "100to199", "200plus"], "recall", "size_recall_heatmap.png")]:
        values = np.array([[next(g[key] for g in groups if g["model"] == row["model"] and
            g["group_type"] == kind and g["group"] == group) * 100 for group in selected_groups] for row in rows])
        fig, ax = plt.subplots(figsize=(7, 9))
        image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(selected_groups)), selected_groups)
        ax.set_yticks(range(17), names)
        for (y, x), value in np.ndenumerate(values):
            ax.text(x, y, f"{value:.1f}", ha="center", va="center", color="white" if value > 60 else "black", fontsize=9)
        fig.colorbar(image, ax=ax, label="Box AP (%)" if key == "box_ap50_95" else "Recall at conf .25 / IoU .50 (%)")
        save(fig, filename)


def documents(prepared, run, rows, canonical, groups, active, evidence):
    checked = datetime.fromisoformat(evidence["checked_utc"]).astimezone(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
    timing_complete = sum(row["timing_state"] == "COMPLETE" for row in rows)
    quality_table = table(["Model", "Box AP %", "AP50 %", "AP75 %", "P %", "R %", "F1 %", "TP", "FP", "FN"],
        [[r["model"]] + [percentage(r[k]) for k in ["box_ap50_95", "box_ap50", "box_ap75", "precision", "recall", "f1"]] +
            [r[k] for k in ["tp", "fp", "fn"]] for r in rows])
    timing_table = table(["Model", "Clean rounds", "Forward ms/ภาพ", "Adapter ms/ภาพ", "Adapter FPS", "Allocated MiB"],
        [[r["model"], f"{r['clean_rounds']}/3"] + [number(r[k]) for k in
            ["inference_mean_ms", "adapter_mean_ms", "fps", "vram_allocated_mib"]] for r in rows])
    winners = sorted(rows, key=lambda row: row["box_ap50_95"], reverse=True)
    best, second = winners[:2]
    recall = max(rows, key=lambda row: row["recall"])
    eligible = [row for row in rows if row["timing_state"] == "COMPLETE"]
    fastest = min(eligible, key=lambda row: row["adapter_mean_ms"])
    comparisons = []
    for prefix in ["yolo26", "yolo11", "yolov9", "yolov8"]:
        family = [row for row in rows if row["model"].startswith(prefix)]
        comparisons.extend(zip(family, family[1:]))
    pipeline_increases = sum(smaller["adapter_mean_ms"] > larger["adapter_mean_ms"]
        for larger, smaller in comparisons if larger["adapter_mean_ms"] is not None and smaller["adapter_mean_ms"] is not None)
    memory_increases = sum(smaller["vram_allocated_mib"] > larger["vram_allocated_mib"]
        for larger, smaller in comparisons if larger["vram_allocated_mib"] is not None and smaller["vram_allocated_mib"] is not None)
    tier_winners = []
    for tier in prepared["config"]["tiers"]:
        selected = sorted([row for row in rows if row["tier"] == tier["id"]], key=lambda row: row["box_ap50_95"], reverse=True)
        tier_winners.append([tier["id"], selected[0]["model"], percentage(selected[0]["box_ap50_95"]),
            f"{(selected[0]['box_ap50_95'] - selected[1]['box_ap50_95']) * 100:.3f}"])
    ap_gap = (best["box_ap50_95"] - second["box_ap50_95"]) * 100
    lead = f"**{best['model']} ได้ Box AP สูงสุด {percentage(best['box_ap50_95'])}%** สูงกว่า {second['model']} {ap_gap:.3f} จุดเปอร์เซ็นต์ ส่วน Recall สูงสุดเป็น {recall['model']} ที่ {percentage(recall['recall'])}%"
    status = f"คะแนนคุณภาพครบ **17/17 โมเดล × 600 ภาพ = 10,200 ผลรายภาพ** ผ่านการตรวจรับ เวลาหลักครบ **{timing_complete}/17 โมเดล**"
    run_info = f"Run `{prepared['run_id']}` · ตรวจปิดรอบเมื่อ {checked} (เวลาไทย)"
    limitations = """## ขอบเขตการตีความ

คะแนนใช้ visible person bbox ที่ด้านยาวตั้งแต่ 50 px มี GT ที่เข้าเกณฑ์ 769 กรอบ คนเล็กอีก 5 กรอบและ ignore rectangles 172 รายการใช้เป็นพื้นที่ยกเว้น ฐานการรับรองว่าตรวจครบทุกคนมาจากการตรวจรับของผู้จัดทำ dataset ทั้ง 600 ภาพ ยังขาดการตรวจบัญชีบุคคลซ้ำอย่างอิสระครบทั้งชุด

โมเดลทุกตัวเป็น segmentation checkpoint แต่ GT รอบนี้รองรับคะแนนกรอบคน การประเมิน Person Mask AP ต้องมี GT silhouette รายคนเพิ่มเติม รูป mask ที่แสดงใน gallery ใช้ตรวจ output และ alignment เท่านั้น การเปรียบเทียบนี้ยังครอบคลุมเฉพาะ detector ไม่มีคะแนน human parsing สี่ส่วน สีเสื้อผ้า หรือ tracking

ทุกตัวใช้ input 640, FP32, batch 1 และตัวประเมินเดียวกัน แต่จำนวน parameters ข้อมูล pretraining และรายละเอียด checkpoint ต่างกัน ผลจึงอธิบาย checkpoint พร้อม application configuration บนชุด CCTV นี้ ประวัติ pretraining exposure ที่ตรวจยืนยันไม่ได้ยังเป็น unknown ข้อมูลบางส่วนเคยใช้พัฒนา annotation helper

ภาพจากกล้องหรือเหตุการณ์เดียวกันอาจสัมพันธ์กัน ยังไม่มี uncertainty หรือ significance analysis ส่วนต่างคะแนนและ near-tie เป็นผลเชิงพรรณนา ห้ามอ้างสาเหตุจาก architecture เพียงอย่างเดียวหรือรับรอง deployment จากชุดนี้

ตาราง MOTS20 เดิมวัด mask บนข้อมูลและ GT อีกแบบ การเทียบแนวโน้มต้องระบุ dataset และชนิด AP กำกับทุกครั้ง
"""
    timing_note = """## สถานะเวลาที่เหลือ

Nano ขาด clean timing 4 รอบ ได้แก่ YOLO26n รอบ 3, YOLO11n รอบ 2 และ 3, YOLOv8n รอบ 3 โดย YOLO11n รอบ 2 เดิมมี 100 observations แต่เป็น `CONTAMINATED` เก็บหลักฐานนั้นไว้และตัดออกจากเวลาหลัก อีกสามรอบยังไม่มีไฟล์ผล ไม่พบ OOM ใน ledger คุณภาพของ run นี้

ขณะตรวจปิดรอบยังมี GPU processes ของบริการเว็บ Body Extraction ในบทบาท detector และ parser เกณฑ์ clean timing เดิมให้เว้นรอบที่มี external GPU compute process แม้ utilization ขณะตรวจเป็นศูนย์ จึงพักการเติมเวลาโดยรักษาบริการเดิม เวลาหลัก Nano เป็น null คะแนนคุณภาพ Nano ครบแล้ว

งานที่เหลือคือเปิด fresh controlled timing segment เฉพาะ 4 รอบนี้เมื่อ GPU ผ่าน admission ใช้ 100 ภาพเดิมและ warmup 10 calls ต่อรอบ จากนั้นตรวจ round/source bindings และรวมสาม clean rounds ต่อโมเดล เก็บ original completion และ contaminated round ไว้ ประวัติอยู่ใน derived result index โดยไม่มีการเปลี่ยน threshold หรือรัน quality 600 ภาพซ้ำ

Pareto และคำแนะนำด้านเวลาในรายงานนี้จำกัดที่ 14 โมเดลที่เวลาครบ เมื่อ Nano ครบอาจได้ frontier ต่างออกไป
"""
    if timing_complete == 17:
        segment = Path(load(run / "closeout/TIMING_CONTINUATION.json")["segment_dir"])
        finished = load(segment / "COMPLETION.json")
        timing_note = f"""## การเติมรอบเวลาและประวัติเดิม

เวลาหลักครบทั้ง 17 ตัวแล้ว ใช้ 47 clean rounds จาก run เดิม ร่วมกับ 4 clean rounds จาก segment `{segment.name}` ตามการอนุมัติของผู้ใช้ รวม 51 รอบ × 100 ภาพ = 5,100 measured observations และ warmup ที่ตรวจรับ 510 calls

ส่วนที่เติมคือ YOLO26n รอบ 3, YOLO11n รอบ 2 และ 3, YOLOv8n รอบ 3 ใช้ภาพ ลำดับ input, checkpoint, environment, preprocessing, source, batch, FP32, guards และ timer เดิม เปลี่ยนเฉพาะ run ID/output path และคิว 4 รอบที่ขาด เก็บ manifest ระบุ compatibility และ hashes การเติมครั้งนี้ใช้ 400 measured calls กับ warmup 40 calls เพิ่มเติม ไม่มี quality calls

YOLO11n รอบ 2 เดิมเป็น `CONTAMINATED` มี 100 observations และ warmup 10 calls เก็บไว้ครบโดยตัดออกจากสถิติเวลาหลัก ค่าใช้จ่าย model calls ของ run หลักและ continuation รวมคุณภาพ 10,200 + timing ที่ตรวจรับ 5,610 + timing ที่ปนเปื้อน 110 = 15,920 calls

ผู้ใช้อนุมัติพักเฉพาะ worker เว็บ Body Extraction ชั่วคราว ตรวจแล้วไม่มีงาน RUNNING ก่อนพัก API และฐานข้อมูลคงทำงาน หลัง CUDA contexts เดิมคืนทรัพยากรจึงเปิด timing แบบ detached ใน tmux จากนั้นเปิด worker เว็บคืน หลักฐาน restoration เก็บแยก การตรวจ GPU ทุกช่วงที่ timer เดิมเก็บไว้ผ่านเกณฑ์ไม่มี external compute PID

Continuation จบเมื่อ {datetime.fromisoformat(finished['at_utc']).astimezone(ZoneInfo('Asia/Bangkok')).isoformat(timespec='seconds')} (เวลาไทย) Original `COMPLETION.json` ที่เคยระบุ timing incomplete ยังคงเดิม สถานะรวมหลัง reconciliation อ่านจาก `metrics/CLOSEOUT.json` และ `metrics/RESULT_INDEX.json`
"""
    group_table = table(["Model", "Day AP %", "Night AP %", "R 50–99 px %", "R 100–199 px %", "R ≥200 px %"], [
        [r["model"]] + [percentage(next(g[key] for g in groups if g["model"] == r["model"] and g["group"] == group))
            for group, key in [("day", "box_ap50_95"), ("night", "box_ap50_95"), ("50to99", "recall"),
                ("100to199", "recall"), ("200plus", "recall")]] for r in rows])
    master = f"""# ผลประเมิน YOLO บน CCTV Human Parsing 600

{run_info}

{status}

{lead}

## 1. สิ่งที่ทดสอบและตรวจปิดรอบ

ทดสอบ pretrained checkpoints 17 ตัวจาก YOLO26, YOLO11, YOLOv9 และ YOLOv8 แบ่งเป็น X/E → L/C → M → S → N ใช้ NVIDIA Tesla T4 จริง ไม่มี fine-tuning ในรอบ benchmark นี้ CPU acceptance ผ่านก่อนเปิดงาน ผูก config/source/checkpoint/environment ไว้ใน PREPARED และ LAUNCH

Environment ใช้ Python 3.12.3, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, Ultralytics 8.4.160 และ pycocotools 2.0.11 ตรึง population SHA256 `{prepared['config']['dataset_membership_sha256']}` และ scientific binding `{prepared['scientific_binding']}` รายการ weights พร้อม SHA256 อยู่ใน [CHECKPOINTS.json](manifests/CHECKPOINTS.json)

งานเริ่ม 16 พฤศจิกายน 2026 เวลา 14:38:19 น. คุณภาพเขียน packet สุดท้ายเวลา 15:23:33 น. ตัวควบคุมจบรอบรวมเวลา 15:43:16 น. ทุกเวลาเป็นเวลาไทย ระยะเหล่านี้รวมงาน orchestration จึงแสดงแยกจากเวลาเฉลี่ยต่อภาพ

ตรวจ membership จากชื่อ packet จริงเทียบ population และ packet audit ทั้ง 17 ตัว อ่าน receipts ผูก hash ครบ 10,200 รายการ ตรวจ per-image CSV กับ TP/FP/FN/GT และ ledger ผล canonical ครบ 600 IDs ไม่ซ้ำต่อโมเดล missing/unexpected/invalid เป็น 0 ทั้งหมด นำ audit ที่ตรวจ readback และ saved-packet scoring แล้วกลับมาใช้ ตรวจ hash ใหม่เฉพาะไฟล์ที่ filesystem timestamps ใหม่กว่า audit พร้อมเก็บ hash ของ metrics/audit เพื่ออ้างย้อนกลับ

ตรวจ source/config binding เดิมผ่านก่อนสร้างรายงาน derived ไม่ได้เรียกโมเดลหรือ full rescore เพื่อจัดตาราง ตรวจ maxDet 200 เทียบ 1000 บน 100 ภาพเดิมผ่านทั้ง 17 ตัว ผลต่าง AP ทั้งสามค่าเป็นศูนย์

## 2. ผลคุณภาพรวม

ตารางเรียง frozen roster order ค่าต้นฉบับเก็บความละเอียดเต็มใน CSV แสดงผลนี้ปัดเศษสามตำแหน่ง P/R/F1 ใช้ confidence ≥0.25 และ bbox IoU ≥0.50 ส่วน Box AP ใช้ IoU 0.50 ถึง 0.95 ทีละ 0.05, 101 recall points, maxDet 200

{quality_table}

### ตัวชี้วัดประกอบและความซับซ้อน

{table(['Model', 'AR@200 %', 'IoU เฉพาะ TP %', 'Parameters', 'GFLOPs estimate', 'Weights MB'], [
    [r['model'], percentage(r['box_ar_maxdet']), percentage(r['matched_box_iou_mean']),
        f"{r['parameters']:,}", number(r['gflops_estimate']), number(r['checkpoint_mb'])] for r in rows])}

AR เฉลี่ย maximum recall ข้าม IoU 10 ระดับ ส่วน IoU เฉพาะ TP เฉลี่ยบนคู่ที่จับสำเร็จที่ operating point เดิม จึงต้องอ่านควบคู่กับ Recall ของแต่ละตัว นิยาม หน่วย และแหล่งคำนวณอธิบายใน [คู่มือตัวชี้วัด](METRIC_GUIDE_TH.md)

![ผล Box AP ครบ 17 โมเดล](metrics/plots/box_ap_by_model.png)

## 3. ผลตามขนาดและการเลือกโมเดล

{table(['กลุ่ม', 'Box AP สูงสุด', 'AP %', 'ห่างอันดับสอง pp'], tier_winners)}

YOLOv9e มี TP {best['tp']} / FP {best['fp']} / FN {best['fn']} ที่ operating point เดิม เทียบกับ YOLO26x ที่ TP {recall['tp']} / FP {recall['fp']} / FN {recall['fn']} จึงมีข้อแลกเปลี่ยนระหว่างการเก็บคนและ false positives

![แนวโน้มตามขนาด](metrics/plots/family_accuracy_scaling.png)

ในกลุ่ม M ค่า AP ของ YOLOv8m สูงกว่า YOLO26m เพียง 0.052 จุดเปอร์เซ็นต์ การเรียกว่าชนะอย่างมีนัยสำคัญต้องมีหลักฐานเพิ่ม ตัวอย่าง near-tie ทั้งหมดอยู่ใน [ตาราง near-tie](metrics/NEAR_TIES.csv) เกณฑ์ส่วนต่างไม่เกิน 0.1 จุดเปอร์เซ็นต์ใช้เพื่อชี้คู่ที่ควรอ่านรายละเอียด

ตาราง [การลดขนาด 13 คู่](metrics/SCALING_DELTAS.csv) คำนวณ smaller − larger จากสถิติเต็มก่อนปัดเศษ ทั้งค่า accuracy จุดเปอร์เซ็นต์และต้นทุนสัมพัทธ์ ตรวจว่าค่า accuracy และเวลาของทั้ง 13 คู่มีแหล่งผลครบแล้ว

การลดขนาดทั้ง 13 คู่ทำให้ AP และ forward time ลดลง แต่ adapter time เพิ่มใน {pipeline_increases} คู่ และ allocated peak เพิ่มใน {memory_increases} คู่ ข้อมูลนี้แสดงว่าชื่อขนาดกับเวลา/หน่วยความจำของ native-mask pipeline ไม่ได้เรียงตามกันเสมอ ต้องพิจารณา postprocessing และ retained outputs ร่วมด้วย การระบุสาเหตุที่แน่นอนต้องมี diagnostic เพิ่ม

ข้อเสนอเบื้องต้นจากข้อมูลชุดนี้คือ YOLOv9e สำหรับ Box AP และ F1, YOLO26x สำหรับ Recall และ YOLO26l สำหรับสมดุล AP กับ adapter time ในกลุ่มที่จับเวลาครบ ทั้งสามยังต้องตรวจความเหมาะสมกับงานจริง ไม่มีคะแนนถ่วงน้ำหนักรวมใหม่

## 4. ความเร็วและทรัพยากร

{timing_table}

### องค์ประกอบและการกระจายของเวลาที่ตรวจรับ

{table(['Model', 'Pre mean ms', 'Forward mean ms', 'Post mean ms', 'Adapter median ms', 'Adapter P95 ms', 'Adapter std ms', 'Reserved MiB'], [
    [r['model']] + ([number(item['timing_primary'][stage]['mean']) for stage in ['preprocess_ms', 'inference_ms', 'postprocess_ms']] +
        [number(item['timing_primary']['total_ms'][stat]) for stat in ['median', 'p95', 'std']] + [number(r['vram_reserved_mib'])]
        if item['timing_primary'] else ['—'] * 7) for r, item in zip(rows, canonical)])}

เวลาหลักใช้ 100 ภาพเดิม × 3 รอบที่ผ่านเกณฑ์ รวม 300 measured observations ต่อโมเดล หลัง warmup 10 ภาพต่อรอบ Forward คือ model call ภายใน adapter ส่วน adapter รวม preprocessing/H2D, forward และ postprocessing/native masks/compact transfer Input read/decode, RLE, packet write, load, warmup และเวลารอคิวแยกออกมา FPS = 1000 / mean adapter ms

Mean, median, P95 และ population std อยู่ใน [TIMING_MASTER.csv](metrics/TIMING_MASTER.csv) รวมรายละเอียด native postprocessing ย่อยที่ timer เดิมเก็บไว้ Substage ที่อยู่ภายใน postprocessing ห้ามบวกซ้ำเข้ารวม pipeline

![องค์ประกอบเวลา](metrics/plots/timing_composition.png)

ใน {timing_complete} ตัวที่เวลาครบ {fastest['model']} มี adapter mean ต่ำสุด {fastest['adapter_mean_ms']:.3f} ms/ภาพ ({fastest['fps']:.3f} FPS) ส่วน forward ต่ำสุดอาจเป็นคนละตัว ภาพ input และจำนวน masks ทำให้ต้นทุน postprocessing ต่างกัน ข้อมูลนี้ยังแยกสาเหตุทาง architecture ไม่ได้

![AP และ adapter time](metrics/plots/accuracy_pipeline_tradeoff.png)

ค่า peak allocated/reserved ใช้ค่าสูงสุดของสาม clean rounds หลัง reset peak เมื่อจบ warmup รวม resident model ด้วย ค่านี้แยกจาก VRAM รวมที่ nvidia-smi เห็น Checkpoint MB ใช้ฐาน 10^6, MiB ใช้ 2^20, parameters เป็นจำนวนก่อน fusion และ GFLOPs เป็นค่าประมาณที่ input 640 จาก exact checkpoint เดียวกับงาน MOTS20

![AP และ parameters](metrics/plots/accuracy_parameters.png)

เวลางานจริง 600 ภาพอยู่ใน [ACTIVE_600_TIMING.csv](metrics/ACTIVE_600_TIMING.csv) มีผลรวมวินาที ค่าเฉลี่ยต่อภาพ และ coverage ของแต่ละ stage รวม cold/session effects ค่า application เป็นช่วงอ่านภาพจนเขียน prediction packet สำเร็จ ห้ามนำ adapter time ไปบวก application ซ้ำ หรือเรียกผลรวมช่วงทำงานว่า elapsed wall time ของทั้งการทดลอง

{timing_note}

## 5. ผลตามบริบทและขนาดคน

{group_table}

Day 243 ภาพ / GT 349 คน และ night 357 ภาพ / GT 420 คน ใช้ metadata ที่มีอยู่ การแยกนี้ยังไม่ควบคุม scene, camera หรือความยากให้เท่ากัน AP รวม 600 ภาพคำนวณแบบ pooled จาก ranking เดียว ค่าเฉลี่ยของ day/night AP ใช้แทน pooled AP ไม่ได้

กลุ่มด้านยาว GT 50–99 px มี 178 คน, 100–199 px มี 342 คน และ ≥200 px มี 249 คน ตารางรายงาน Recall จาก matching เดิมโดยนับ GT ที่อยู่ในช่วงนั้น ไม่ได้คำนวณ AP หรือ Precision ต่อ size band เพราะยังไม่ได้กำหนดการแจก FP ลงช่วงขนาด

![ผลตาม day/night](metrics/plots/day_night_ap_heatmap.png)

![Recall ตามขนาดคน](metrics/plots/size_recall_heatmap.png)

## 6. ภาพเปรียบเทียบ

ใช้ fixed sample 16 ภาพที่กำหนดก่อนเห็นผล ทั้งห้า tiers ใช้ IDs เดียวกัน มี 80 comparison panels ตรวจ manifest/model membership และขนาดไฟล์ภาพครบ ตรวจด้วยสายตาบนภาพลำดับ 1 และ 3 ข้ามห้า tiers รวม 10 panels ภาพลำดับ 1 เห็นคนใกล้ขอบภาพกับคนไกลในแนวบันได ผลของโมเดลต่อคนไกลต่างกัน ภาพลำดับ 3 เห็นบุคคลถูกตัดด้วยขอบเฟรม การเปรียบเทียบทั้งสองเป็นตัวอย่าง output ไม่ใช้รับรองความถูกต้องของขอบ mask ทั้งชุด

ภาพต้นฉบับและ gallery มีข้อมูล CCTV ส่วนบุคคล จัดเก็บในเครื่อง เปิดจาก `outputs/<run_id>/gallery/index.html` ของแต่ละ tier หรือเปิด [ดัชนีในเครื่อง](outputs/{prepared['run_id']}/closeout/index.html) ซึ่งรวมลิงก์ทั้งห้ากลุ่ม ไฟล์ภาพชุดนี้เก็บตามนโยบาย private ส่วนกราฟรวมใน Git เปิดดูได้โดยตรง

## 7. การจัดผลตามงาน MOTS20

{table(['ส่วน', 'ผลส่งมอบ CCTV600'], [
    ['ตารางหลักและผู้ชนะรายขนาด', 'MASTER_17_MODELS.csv และ TIER_WINNERS.csv'],
    ['AP / P / R / F1 / TP / FP / FN / IoU / AR', 'ตารางหลักและคู่มือนิยามตัวชี้วัด'],
    ['PR / F1-confidence curves', 'กราฟเดิมราย tier จาก saved predictions'],
    ['Scaling / near ties', '13 adjacent pairs และคู่ AP ต่างกันไม่เกิน 0.1 pp'],
    ['Forward / pipeline / VRAM Pareto', f'ตารางและกราฟเฉพาะ {timing_complete} โมเดลเวลาครบ'],
    ['Stage timing / complexity', 'Mean / median / P95 / std, parameters, GFLOPs, MB'],
    ['ผลตามกลุ่มข้อมูล', 'Day/night AP และ Recall ตามขนาด GT'],
    ['ภาพจริง', 'Fixed sample 16 ภาพทุก tier เก็บ private'],
    ['Provenance / closure', 'RESULT_INDEX.json และ CLOSEOUT.json']])}

## 8. ไฟล์สำหรับตรวจต่อ

- [บทสรุปเพื่อเลือกโมเดล](EXECUTIVE_SUMMARY_TH.md)
- [นิยามตัวชี้วัด สูตร ขอบเขตเวลา และ schema ตาราง](METRIC_GUIDE_TH.md)
- [ตารางรวมความละเอียดเต็ม](metrics/MASTER_17_MODELS.csv)
- [สถานะและ hashes ของผลที่ตรวจรับ](metrics/RESULT_INDEX.json)
- [ผลตรวจปิดรอบ](metrics/CLOSEOUT.json)
- [Protocol ที่ใช้รัน](EXPERIMENT_PROTOCOL.md)

{limitations}
"""
    (STUDY / "MASTER_RESULTS.md").write_text(master.rstrip() + "\n", encoding="utf-8")
    summary = f"""# สรุปสำหรับนำเสนอผล YOLO CCTV600

{run_info}

{status}

{lead}

## ข้อเสนอการเลือก

| เป้าหมาย | ตัวเลือกจากผลนี้ | สิ่งที่ต้องพิจารณา |
| --- | --- | --- |
| Box AP และ F1 | YOLOv9e-Seg | AP 67.808%, F1 85.356%, FP 53, FN 157 |
| เก็บคนให้ได้มากที่ confidence 0.25 | YOLO26x-Seg | Recall 81.144%, TP 624, FN 145, FP 113 |
| สมดุล AP กับ adapter time | YOLO26l-Seg | AP 65.099%, adapter mean 59.770 ms/ภาพ |
| คะแนนกลุ่ม M | YOLOv8m-Seg / YOLO26m-Seg | AP ต่างกันเพียง 0.052 จุดเปอร์เซ็นต์ ให้พิจารณา route และ resource ร่วม |
| คะแนนกลุ่ม S / N | YOLO26s-Seg / YOLO26n-Seg | นำ AP ในขนาดของตน แต่ Recall ต่ำกว่ารุ่นใหญ่ |

การเลือกข้างต้นอิงผล CCTV ชุดนี้ ต้องยืนยันกับข้อกำหนดระบบก่อน deployment ตารางใช้ pretrained checkpoint และ input 640 FP32 batch 1 บน Tesla T4 ค่า Box AP บอกคุณภาพกรอบตรวจคน ส่วน F1 ใช้ threshold ที่ตรึงไว้ก่อนรัน

![ภาพรวมผล](metrics/plots/box_ap_by_model.png)

![แนวโน้มตามขนาด](metrics/plots/family_accuracy_scaling.png)

![องค์ประกอบเวลา](metrics/plots/timing_composition.png)

{timing_note}

อ่านผลครบทั้ง 17 ตัว ตัวเลขต่อกลุ่ม และแหล่งหลักฐานใน [รายงานหลัก](MASTER_RESULTS.md) สูตรและความหมายอยู่ใน [คู่มือตัวชี้วัด](METRIC_GUIDE_TH.md)

{limitations}
"""
    (STUDY / "EXECUTIVE_SUMMARY_TH.md").write_text(summary.rstrip() + "\n", encoding="utf-8")
    state_text = ("สถานะ `COMPLETE` คุณภาพครบ 17 ตัวและเวลาครบ 51 clean rounds เติมเฉพาะ Nano 4 รอบผ่าน fresh timing continuation แล้วเปิด worker เว็บคืน ไม่มีการรัน quality ซ้ำ\n\n"
        if timing_complete == 17 else "สถานะ `QUALITY_CLOSED_TIMING_INCOMPLETE` ตัวควบคุมเดิมสิ้นสุดแล้ว Nano เหลือ clean timing 4 รอบ เวลาหลัก Nano คงเป็น null\n\n")
    (STUDY / "STATUS.md").write_text(f"# สถานะปิดรอบ YOLO CCTV600\n\n{run_info}\n\n{status}\n\n" + state_text +
        "อ่าน [รายงานหลัก](MASTER_RESULTS.md), [สรุปเลือกโมเดล](EXECUTIVE_SUMMARY_TH.md) และ [ผลตรวจปิดรอบ](metrics/CLOSEOUT.json)\n",
        encoding="utf-8")
    for tier in prepared["config"]["tiers"]:
        repository = WORKSPACE / tier["repository"]
        selected = [row for row in rows if row["tier"] == tier["id"]]
        best_tier = max(selected, key=lambda row: row["box_ap50_95"])
        base = "https://github.com/folklazy/" + STUDY.name + "/blob/main/"
        tier_text = f"# ผลประเมิน YOLO กลุ่ม {tier['id']} บน CCTV600\n\n{run_info}\n\n"
        tier_text += f"คุณภาพครบ {len(selected)}/{len(selected)} โมเดล ตัวละ 600 ภาพ GT เข้าเกณฑ์ 769 คน คะแนน Box AP สูงสุดในกลุ่มคือ **{best_tier['model']} {percentage(best_tier['box_ap50_95'])}%**\n\n"
        tier_text += "## คุณภาพกรอบบุคคล\n\n" + table(
            ["Model", "AP %", "AP50 %", "AP75 %", "P %", "R %", "F1 %", "TP", "FP", "FN"],
            [[r["model"]] + [percentage(r[k]) for k in ["box_ap50_95", "box_ap50", "box_ap75", "precision", "recall", "f1"]] + [r[k] for k in ["tp", "fp", "fn"]] for r in selected]) + "\n\n"
        tier_text += "AP ใช้ IoU 0.50–0.95, 101 recall points และ maxDet 200 P/R/F1 ใช้ confidence ≥0.25 และ IoU ≥0.50 จับ valid GT ก่อนพิจารณา ignore union\n\n"
        tier_text += "## ความเร็ว\n\n" + table(["Model", "Clean rounds", "Forward ms/ภาพ", "Adapter ms/ภาพ", "FPS", "Allocated MiB"],
            [[r["model"], f"{r['clean_rounds']}/3"] + [number(r[k]) for k in ["inference_mean_ms", "adapter_mean_ms", "fps", "vram_allocated_mib"]] for r in selected]) + "\n\n"
        tier_text += "เวลาหลักมาจาก 100 ภาพเดิม × 3 clean rounds หลัง warmup 10 calls ต่อรอบ Adapter รวม preprocess/H2D + forward + postprocess/native-mask transfer ส่วน input read/decode, RLE และเขียนไฟล์วัดแยก FPS = 1000 / mean adapter ms\n\n"
        if tier["id"] == "nano":
            tier_text += timing_note + "\n"
        tier_text += f"## กราฟและภาพตัวอย่าง\n\n![PR curve](https://raw.githubusercontent.com/folklazy/{STUDY.name}/main/metrics/plots/{tier['id']}_pr_curve.png)\n\n"
        tier_text += f"![F1-confidence curve](https://raw.githubusercontent.com/folklazy/{STUDY.name}/main/metrics/plots/{tier['id']}_f1_curve.png)\n\n"
        tier_text += f"เส้น F1 แสดงผลของ threshold ต่อ trade-off โดย main threshold คง 0.25 ภาพเปรียบเทียบ 16 ภาพเดิมเปิดในเครื่องที่ `outputs/{prepared['run_id']}/gallery/index.html` ของ repo นี้\n\n"
        tier_text += f"## แหล่งผลและข้อจำกัด\n\n[ตารางกลุ่ม](metrics/TIER_RESULTS.csv) และ [canonical summary](metrics/RESULTS.json) มาจาก saved packets ที่ตรวจรับแล้ว Source/scientific bindings คงเดิม อ่าน [รายงานรวม]({base}MASTER_RESULTS.md), [สูตรตัวชี้วัด]({base}METRIC_GUIDE_TH.md) และ [ผลตรวจปิดรอบ]({base}metrics/CLOSEOUT.json)\n\n"
        tier_text += "งานนี้วัด visible bbox บน CCTV600 ของ pretrained segmentation checkpoints ส่วน native masks เก็บเป็น output diagnostics ยังไม่มี GT silhouette สำหรับ Person Mask AP ไม่มี human parsing/color/tracking score และยังไม่มี uncertainty analysis\n"
        (repository / "RESULTS_SUMMARY_TH.md").write_text(tier_text.rstrip() + "\n", encoding="utf-8")
        (repository / "STATUS.md").write_text(f"# สถานะ {tier['id']}\n\n{run_info}\n\n"
            f"Quality COMPLETE {len(selected)}/{len(selected)} models × 600 images. Clean timing COMPLETE "
            f"{sum(r['timing_state'] == 'COMPLETE' for r in selected)}/{len(selected)} models.\n\n"
            "อ่าน [ผลประเมินและภาพประกอบ](RESULTS_SUMMARY_TH.md) หลักฐาน run เดิมเก็บไว้ใน outputs\n", encoding="utf-8")
        csv_save(repository / "metrics/TIER_RESULTS.csv", selected)
        csv_save(repository / "metrics/TIMING_SUMMARY.csv", [item for item in evidence["timing_statistics"] if item["model"] in tier["models"]])


def main():
    if sys.argv[1:] == ["--status"]:
        evidence = load(STUDY / "metrics/CLOSEOUT.json")
        print(json.dumps({key: evidence[key] for key in ["state", "checked_utc", "run_id",
            "quality_models_complete", "clean_timing_models_complete", "timing_rounds_pending"]}, ensure_ascii=False, indent=2))
        return
    run = Path(load(STUDY / "outputs/CURRENT.json")["run_dir"])
    prepared = load(run / "PREPARED.json")
    original_hashes = {link_path(path): sha(path) for path in
        [run / "PREPARED.json", run / "COMPLETION.json", run / "STATUS.json"]}
    repositories = [STUDY] + [WORKSPACE / tier["repository"] for tier in prepared["config"]["tiers"]]
    archive = run / "closeout" / ("snapshot-" + datetime.now().strftime("%Y%m%dT%H%M%S%f"))
    snapshot(repositories, archive)
    rows, canonical, groups, active, timings, caps, events = collect(prepared, run)
    derived_tables(rows, groups, active, timings, caps)
    plots(rows, groups, canonical)
    panels = []
    for tier in prepared["config"]["tiers"]:
        directory = WORKSPACE / tier["repository"] / "outputs" / prepared["run_id"] / "gallery"
        manifest = load(directory / "MANIFEST.json")
        assert manifest["fixed_sample"] == prepared["visual_sample"]
        assert manifest["source_binding"] == prepared["scientific_binding"]
        assert [item["image_id"] for item in manifest["panels"]] == prepared["visual_sample"]
        from PIL import Image
        for item in manifest["panels"]:
            assert item["models"] == tier["models"]
            path = directory / item["file"]
            with Image.open(path) as image:
                assert image.width == 720 * (len(tier["models"]) + 1)
                image.verify()
            panels.append({"tier": tier["id"], "image_id": item["image_id"], "file": link_path(path), "sha256": sha(path)})
    write(run / "closeout/GALLERY_AUDIT.json", {"panels": panels, "visual_review_sample_positions_1based": [1, 3]})
    html = "<meta charset='utf-8'><title>CCTV600 comparison galleries</title><h1>Fixed sample · 16 images × 5 tiers</h1><ul>"
    import os
    for tier in prepared["config"]["tiers"]:
        target = WORKSPACE / tier["repository"] / "outputs" / prepared["run_id"] / "gallery/index.html"
        html += "<li><a href='" + os.path.relpath(target, run / "closeout") + "'>" + tier["id"] + "</a></li>"
    (run / "closeout/index.html").write_text(html + "</ul>", encoding="utf-8")
    process = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], capture_output=True, text=True, check=True)
    write(run / "closeout/GPU_PROCESS_CHECK.json", {"at_utc": now(), "compute_processes": process.stdout})
    pointer = run / "closeout/TIMING_CONTINUATION.json"
    continuation = Path(load(pointer)["segment_dir"]) if pointer.exists() else None
    recovered = load(continuation / "COMPLETION.json") if continuation and (continuation / "COMPLETION.json").exists() else None
    timing_complete = sum(row["timing_state"] == "COMPLETE" for row in rows)
    evidence = {"schema_version": "1.0.0", "checked_utc": now(),
        "state": "COMPLETE" if timing_complete == 17 else "QUALITY_CLOSED_TIMING_INCOMPLETE", "run_id": prepared["run_id"],
        "scientific_binding": prepared["scientific_binding"], "source_binding_verified": True,
        "quality_models_expected": 17, "quality_models_complete": len(rows), "accepted_model_image_records": len(rows) * 600,
        "membership_totals": {key: sum(item["membership"][key] for item in canonical)
            for key in ["missing", "unexpected", "duplicate", "invalid"]},
        "gt_people": sum(len(frame["people"]) for frame in prepared["frames"]),
        "quality_model_calls_in_closeout": 0,
        "timing_model_calls_in_closeout": recovered["measured_model_calls"] + recovered["warmup_calls"] if recovered else 0,
        "continuation": {"run_id": continuation.name, "manifest_sha256": sha(continuation / "MANIFEST.json"),
            "completion_sha256": sha(continuation / "COMPLETION.json"),
            "controller_sha256": sha(STUDY / "analysis/timing_continuation.py"),
            "web_worker_restoration": load(continuation / "WEB_WORKER_RESTORATION.json")} if recovered else None,
        "full_rescores_in_closeout": 0, "ledger_events": dict(events),
        "clean_timing_models_complete": sum(row["timing_state"] == "COMPLETE" for row in rows),
        "accepted_clean_rounds": sum(row["clean_rounds"] for row in rows),
        "timing_rounds_expected": 51,
        "timing_rounds_pending": [{"model": item["model"], "round": record["round"],
            "original_state": record["state"]} for item in canonical for record in item["timing_rounds"] if record["state"] != "CLEAN_COMPLETE"],
        "original_run_hashes": original_hashes,
        "reporter_sha256": sha(Path(__file__)), "gallery_panels_checked": len(panels),
        "checks_performed": ["Frozen scientific source/config identity", "Exact 600-ID membership per model",
            "Accepted packet audit reuse and post-audit timestamp change detection",
            "10,200 receipt-to-packet-hash bindings", "Per-image counts and immutable ledger consistency",
            "Unrounded precision/recall/F1 formulas", "17 maxDet parity records",
            "51 clean timing round bindings, sample order and resource observations",
            "Per-observation adapter stage sum", "Original completion/quality metric immutability",
            "80 fixed-gallery manifest memberships and image geometry"] if timing_complete == 17 else [],
        "packet_audit_reuse": "Reuse accepted per-packet readback/hash/scoring audits; compare exact filenames, receipts and per-image counts; rehash only post-audit timestamp changes",
        "gpu_timing_blocker": None if recovered else "External GPU compute PIDs of the Body Extraction detector/parser service at closeout",
        "timing_statistics": timings}
    write(STUDY / "metrics/RESULT_INDEX.json", {"schema_version": "1.0.0", "run_id": prepared["run_id"],
        "scientific_binding": prepared["scientific_binding"], "sources": prepared["sources"], "models": canonical})
    write(STUDY / "metrics/MASTER_RESULTS.json", {"run_id": prepared["run_id"],
        "scientific_binding": prepared["scientific_binding"], "models": rows,
        "result_index": "RESULT_INDEX.json", "summary_scope": "DERIVED_AFTER_CONTROLLED_TIMING_CONTINUATION"})
    for tier in prepared["config"]["tiers"]:
        public_path = WORKSPACE / tier["repository"] / "metrics/RESULTS.json"
        public = load(public_path)
        for item in public["models"]:
            derived = next(model for model in canonical if model["model"] == item["model"])
            assert item["primary"] == load(WORKSPACE / derived["metrics_path"])["primary"]
            item["timing"] = {"state": derived["timing_state"],
                "clean_rounds": sum(r["state"] == "CLEAN_COMPLETE" for r in derived["timing_rounds"]),
                "primary": derived["timing_primary"], "rounds": derived["timing_rounds"]}
        public["summary_scope"] = "DERIVED_AFTER_CONTROLLED_TIMING_CONTINUATION"
        write(public_path, public)
    documents(prepared, run, rows, canonical, groups, active, evidence)
    evidence.pop("timing_statistics")
    write(STUDY / "metrics/CLOSEOUT.json", evidence)
    for path, digest in original_hashes.items():
        assert sha(WORKSPACE / path) == digest
    verify_source(prepared)
    for item in canonical:
        assert sha(WORKSPACE / item["metrics_path"]) == item["metrics_sha256"]
        assert sha(WORKSPACE / item["packet_audit_path"]) == item["packet_audit_sha256"]
    print(json.dumps({"quality_models": len(rows), "clean_timing_models": evidence["clean_timing_models_complete"],
        "pending_timing_rounds": evidence["timing_rounds_pending"], "gallery_panels": len(panels)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
