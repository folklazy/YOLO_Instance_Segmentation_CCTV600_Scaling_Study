"""Lossless native masks and immutable original-grid prediction packets."""
from dataclasses import dataclass
from functools import cached_property
import gzip
import json
import math

import numpy as np
from pycocotools import mask as coco_mask

from .common import immutable_bytes, load, sha


@dataclass
class PackedPrediction:
    confidence: float
    class_id: int
    bbox_xyxy: list
    packed: np.ndarray
    shape: tuple

    @property
    def mask(self):
        pixels = np.unpackbits(self.packed, bitorder="little", count=int(np.prod(self.shape)))
        return pixels.reshape(self.shape).view(bool)

    @cached_property
    def rle(self):
        return coco_mask.encode(np.asfortranarray(self.mask.astype(np.uint8)))

    def compact(self):
        return {"class_id": self.class_id, "confidence": self.confidence,
                "bbox_xyxy": self.bbox_xyxy,
                "rle": {"size": self.rle["size"], "counts": self.rle["counts"].decode("ascii")}}


def save_packet(path, record):
    body = json.dumps(record, separators=(",", ":"), allow_nan=False).encode()
    immutable_bytes(path, gzip.compress(body, mtime=0))


def read_packet(path, frame, binding):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    if record["image_id"] != frame["image_id"] or record["binding"] != binding:
        raise ValueError("Prediction/input/model binding mismatch")
    if record["image_sha256"] != frame["image_sha256"]:
        raise ValueError("Prediction image hash mismatch")
    if record["original_wh"] != [frame["width"], frame["height"]]:
        raise ValueError("Prediction image geometry mismatch")
    if record["input_shape"] != [1, 3, 640, 640]:
        raise ValueError("Prediction network geometry mismatch")
    for number, item in enumerate(record["predictions"], 1):
        score, box = item["confidence"], item["bbox_xyxy"]
        if type(item.get("person_id")) is not int or item["person_id"] != number:
            raise ValueError("Native prediction ownership IDs invalid")
        if type(item["class_id"]) is not int or item["class_id"] != 0 or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid Person label/confidence")
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError("Invalid prediction bbox")
        if not (0 <= box[0] < box[2] <= frame["width"] and
                0 <= box[1] < box[3] <= frame["height"]):
            raise ValueError("Prediction bbox outside image or empty")
        rle = item["rle"]
        if rle["size"] != [frame["height"], frame["width"]]:
            raise ValueError("Native mask geometry mismatch")
        if int(coco_mask.area({"size": rle["size"], "counts": rle["counts"].encode("ascii")})) <= 0:
            raise ValueError("Empty retained native mask")
    if len(record["predictions"]) >= 1000:
        raise ValueError("Model cap saturated")
    receipt_path = path.parent.parent / "receipts" / (frame["image_id"] + ".json")
    record["timing"]["output_write_ms"] = None
    record["timing"]["application_ms"] = None
    if receipt_path.exists():
        receipt = load(receipt_path)
        if receipt["packet_sha256"] != sha(path) or receipt["binding"] != binding:
            raise ValueError("Execution receipt binding mismatch")
        record["timing"].update(receipt["timing"])
    return record
