"""Person-first bbox matching and COCO-style AP with explicit coarse-ignore union."""
from contextlib import redirect_stdout
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def area(box):
    return max(0., box[2] - box[0]) * max(0., box[3] - box[1])


def intersection(left, right):
    return [max(left[0], right[0]), max(left[1], right[1]),
            min(left[2], right[2]), min(left[3], right[3])]


def box_iou(left, right):
    overlap = area(intersection(left, right))
    union = area(left) + area(right) - overlap
    return overlap / union if union else 0.


def ignore_ioa(box, rectangles):
    """Continuous rectangle-union area, avoiding double-counted ignore overlaps."""
    clipped = [intersection(box, rect) for rect in rectangles]
    clipped = [rect for rect in clipped if area(rect) > 0]
    xs = sorted({x for rect in clipped for x in [rect[0], rect[2]]})
    covered = 0.
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((r[1], r[3]) for r in clipped if r[0] < right and r[2] > left)
        y_length, bottom = 0., None
        for low, high in intervals:
            start = low if bottom is None else max(low, bottom)
            y_length += max(0., high - start)
            bottom = high if bottom is None else max(bottom, high)
        covered += (right - left) * y_length
    return covered / area(box) if area(box) else 0.


def ratios(tp, fp, fn):
    return {"precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def fixed(frame, predictions, confidence=.25, threshold=.5, ignore_threshold=.5):
    predictions = sorted((p for p in predictions if p["confidence"] >= confidence),
                         key=lambda p: (-p["confidence"], *p["bbox_xyxy"]))
    remaining = set(range(len(frame["people"])))
    matched, extra, ignored, events = [], [], [], []
    for number, prediction in enumerate(predictions):
        box = prediction["bbox_xyxy"]
        candidates = [(box_iou(box, frame["people"][j]["bbox"]), j) for j in remaining]
        candidates = [(value, j) for value, j in candidates if value >= threshold]
        if candidates:
            value, chosen = min(candidates, key=lambda item: (-item[0], *frame["people"][item[1]]["bbox"]))
            remaining.remove(chosen)
            matched.append({"prediction_index": number, "gt_index": chosen, "box_iou": value})
            events.append({"confidence": prediction["confidence"], "disposition": "TP"})
        elif ignore_ioa(box, frame["ignore"]) >= ignore_threshold:
            ignored.append(number)
            events.append({"confidence": prediction["confidence"], "disposition": "IGNORE"})
        else:
            extra.append(number)
            events.append({"confidence": prediction["confidence"], "disposition": "FP"})
    tp, fp, fn = len(matched), len(extra), len(remaining)
    return {"tp": tp, "fp": fp, "fn": fn, "ignored_predictions": len(ignored),
            "matched_box_iou_mean": float(np.mean([m["box_iou"] for m in matched])) if matched else None,
            **ratios(tp, fp, fn), "matches": matched, "fn_indices": sorted(remaining), "events": events}


class BboxEvaluator(COCOeval):
    def __init__(self, gt, dt, frames, ignore_threshold):
        super().__init__(gt, dt, "bbox")
        self.frames = {frame["coco_id"]: frame for frame in frames}
        self.ignore_threshold = ignore_threshold

    def computeIoU(self, image_id, category_id):
        matrix = super().computeIoU(image_id, category_id)
        if not len(matrix):
            return matrix
        gt = self._gts[image_id, category_id]
        dt = sorted(self._dts[image_id, category_id], key=lambda item: -item["score"])
        dt = dt[:self.params.maxDets[-1]]
        for column, item in enumerate(gt):
            if item["iscrowd"]:
                for row, detection in enumerate(dt):
                    x, y, width, height = detection["bbox"]
                    matrix[row, column] = float(ignore_ioa(
                        [x, y, x + width, y + height], self.frames[image_id]["ignore"]
                    ) >= self.ignore_threshold)
        return matrix


def ap(frames, prediction_lists, max_dets=200, ignore_threshold=.5, include_curve=False):
    annotations, detections = [], []
    for frame, predictions in zip(frames, prediction_lists):
        for person in frame["people"]:
            x1, y1, x2, y2 = person["bbox"]
            annotations.append({"id": len(annotations) + 1, "image_id": frame["coco_id"],
                "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        if frame["ignore"]:
            annotations.append({"id": len(annotations) + 1, "image_id": frame["coco_id"],
                "category_id": 1, "bbox": [0, 0, frame["width"], frame["height"]],
                "area": frame["width"] * frame["height"], "iscrowd": 1})
        ordered = sorted(predictions, key=lambda item: (-item["confidence"], *item["bbox_xyxy"]))
        for item in ordered:
            x1, y1, x2, y2 = item["bbox_xyxy"]
            detections.append({"image_id": frame["coco_id"], "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": item["confidence"]})
    with redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"info": {}, "images": [{"id": f["coco_id"], "width": f["width"],
            "height": f["height"]} for f in frames], "categories": [{"id": 1, "name": "person"}],
            "annotations": annotations}
        gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = {**gt.dataset, "annotations": []}
            dt.createIndex()
        evaluator = BboxEvaluator(gt, dt, frames, ignore_threshold)
        evaluator.params.maxDets = [1, 10, max_dets]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, 0, 0, -1]
    def mean_valid(values):
        valid = values[values >= 0]
        return float(valid.mean()) if valid.size else None
    result = {"box_ap50_95": mean_valid(precision),
            "box_ap50": mean_valid(precision[0]), "box_ap75": mean_valid(precision[5]),
            "box_ar_maxdet": mean_valid(evaluator.eval["recall"][:, 0, 0, -1])}
    if include_curve:
        result["pr_curve_iou50"] = {"recall": evaluator.params.recThrs.tolist(),
            "precision": [float(v) if v >= 0 else None for v in precision[0]]}
    return result


def aggregate(frames, prediction_lists, cfg):
    rows = [fixed(frame, predictions, cfg["fixed_confidence"], cfg["matching_iou"],
                  cfg["ignore_prediction_ioa"]) for frame, predictions in zip(frames, prediction_lists)]
    counts = {key: sum(row[key] for row in rows) for key in ["tp", "fp", "fn", "ignored_predictions"]}
    matches = [match["box_iou"] for row in rows for match in row["matches"]]
    return {"images": len(frames), "gt_people": sum(len(f["people"]) for f in frames), **counts,
            **ratios(counts["tp"], counts["fp"], counts["fn"]),
            "matched_box_iou_mean": float(np.mean(matches)) if matches else None,
            **ap(frames, prediction_lists, cfg["ap_max_dets"], cfg["ignore_prediction_ioa"], include_curve=True)}, rows
