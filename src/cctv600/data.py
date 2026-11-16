"""Read-only bbox GT export and deterministic workload selection."""
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import platform

from PIL import Image

from .common import config, identity, load, now, sha, source_binding, STUDY, WORKSPACE, write


def box_of(shape):
    points = shape["points"]
    if not points or not all(len(p) == 2 and all(math.isfinite(v) for v in p) for p in points):
        raise ValueError("Malformed/nonfinite annotation coordinates")
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]


def clip_box(box, width, height):
    return [max(0., min(float(width), box[0])), max(0., min(float(height), box[1])),
            max(0., min(float(width), box[2])), max(0., min(float(height), box[3]))]


def frame_from_row(row, number):
    image, annotation = Path(row["image_path"]), Path(row["annotation_path"])
    if sha(image) != row["image_sha256"] or sha(annotation) != row["annotation_sha256"]:
        raise ValueError("Source image/annotation changed from accepted600 membership")
    document = load(annotation)
    with Image.open(image) as decoded:
        decoded.load()
        width, height = decoded.size
    if (width, height) != (document["imageWidth"], document["imageHeight"]):
        raise ValueError("Annotation/image dimensions disagree")
    people, excluded, ignore = [], [], []
    for shape in document["shapes"]:
        if shape["label"] not in {"bbox", "ignore"}:
            continue
        raw = box_of(shape)
        box = clip_box(raw, width, height)
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Degenerate person/ignore bbox")
        if shape["label"] == "ignore":
            ignore.append(box)
            continue
        long_side = max(raw[2] - raw[0], raw[3] - raw[1])
        person = {"person_id": shape["group_id"], "bbox": box, "source_bbox": raw,
                  "long_side_px": long_side,
                  "size_band": "50to99" if long_side < 100 else "100to199" if long_side < 200 else "200plus"}
        if long_side >= config()["minimum_gt_long_side_px"]:
            people.append(person)
        else:
            excluded.append(person)
            ignore.append(box)
    identifiers = [p["person_id"] for p in people + excluded]
    if len(identifiers) != len(set(identifiers)) or any(type(v) is not int or v < 1 for v in identifiers):
        raise ValueError("Duplicate/invalid bbox owner")
    people.sort(key=lambda p: (p["bbox"][1], p["bbox"][0], p["bbox"][3], p["bbox"][2]))
    return {"image_id": row["image_id"], "coco_id": number, "image_path": str(image.resolve()),
            "annotation_path": str(annotation.resolve()), "image_sha256": row["image_sha256"],
            "annotation_sha256": row["annotation_sha256"], "width": width, "height": height,
            "people": people, "excluded_people": excluded, "ignore": ignore,
            "diurnal": row["diurnal_metadata"], "site": row["site_code"], "camera": row["camera_code"]}


def select_subset(frames, count, seed):
    groups = defaultdict(list)
    for frame in frames:
        groups[(frame["diurnal"], frame["site"], frame["camera"])].append(frame)
    for group in groups.values():
        group.sort(key=lambda f: identity([seed, f["image_id"]]))
    keys = sorted(groups, key=lambda key: identity([seed, key]))
    selected = []
    while len(selected) < count:
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0)["image_id"])
    return selected


def prepare(manifest):
    cfg = config()
    rows = sorted(csv.DictReader(Path(manifest).open()), key=lambda row: row["image_id"])
    if len(rows) != cfg["expected_images"] or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Expected exact600 unique image IDs")
    membership = [[r["image_id"], r["image_sha256"], r["annotation_sha256"]] for r in rows]
    body = __import__("json").dumps(membership, separators=(",", ":")).encode()
    if hashlib.sha256(body).hexdigest() != cfg["dataset_membership_sha256"]:
        raise ValueError("Dataset membership fingerprint mismatch")
    frames = [frame_from_row(row, index) for index, row in enumerate(rows, 1)]
    sample = select_subset(frames, cfg["timing_images"], cfg["seed"])
    checkpoints = {}
    complexity = load(STUDY / "manifests/COMPLEXITY_REFERENCE.json")["models"]
    for tier in cfg["tiers"]:
        for name in tier["models"]:
            path = WORKSPACE / "models" / name
            if not path.is_file():
                raise FileNotFoundError("Local checkpoint missing: " + name)
            checkpoints[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
            reference = complexity.get(name)
            if reference and reference["checkpoint_sha256"] == checkpoints[name]["sha256"]:
                checkpoints[name]["reference_gflops"] = reference["gflops_estimate"]
                checkpoints[name]["reference_parameters"] = reference["loaded_parameters"]
    identifier = "cctv600-" + now().replace("-", "").replace(":", "").replace(".", "").replace("+0000", "Z")
    directory = STUDY / "outputs" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    environment = {"python": platform.python_version(), "packages": {
        name: importlib.metadata.version(name) for name in
        ["torch", "torchvision", "ultralytics", "numpy", "pycocotools", "Pillow", "opencv-python", "PyYAML"]}}
    framework = Path(next(iter(importlib.util.find_spec("ultralytics").submodule_search_locations)))
    environment["ultralytics_sources"] = {str(path): sha(path) for path in sorted(framework.rglob("*.py"))}
    prepared = {"run_id": identifier, "created_utc": now(), "config": cfg,
                "config_sha256": sha(STUDY / "configs/study.json"), "frames": frames,
                "sample": sample, "visual_sample": sample[:16], "checkpoints": checkpoints,
                "sources": source_binding(), "environment": environment,
                "source_manifest_sha256": sha(manifest)}
    prepared["scientific_binding"] = identity({k: v for k, v in prepared.items()
        if k not in {"run_id", "created_utc", "scientific_binding"}})
    write(directory / "PREPARED.json", prepared)
    write(STUDY / "outputs/CURRENT.json", {"run_dir": str(directory)})
    summary = {"images": len(frames), "eligible_person_boxes": sum(len(f["people"]) for f in frames),
               "retained_under50_boxes": sum(len(f["excluded_people"]) for f in frames),
               "diurnal_counts": dict(Counter(f["diurnal"] for f in frames)),
               "timing_images": len(sample), "dataset_membership_sha256": cfg["dataset_membership_sha256"],
               "bbox_policy": cfg["gt_bbox_coordinates"], "ignore_policy": cfg["ignore_policy"],
               "person_inventory_basis": cfg["person_inventory_basis"]}
    if summary["eligible_person_boxes"] != 769 or summary["retained_under50_boxes"] != 5:
        raise ValueError("GT eligibility differs from accepted dataset closeout")
    write(directory / "GT_SUMMARY.json", summary)
    write(STUDY / "manifests/DATASET_SUMMARY.json", summary)
    write(STUDY / "manifests/CHECKPOINTS.json", checkpoints)
    return directory
