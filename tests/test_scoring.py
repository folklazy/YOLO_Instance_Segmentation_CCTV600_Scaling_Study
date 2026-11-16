"""Analytical bbox expectations and actual packet-to-scorer checks on CPU."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask as coco_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cctv600.common import config, sha
from cctv600.data import frame_from_row
from cctv600.packets import PackedPrediction, read_packet, save_packet
from cctv600.reporting import model_binding, model_directory, score_model
from cctv600.scoring import ap, box_iou, fixed, ignore_ioa


def frame(people=None, ignore=None):
    return {"image_id": "synthetic", "coco_id": 1, "width": 130, "height": 90,
            "image_sha256": "imagehash", "diurnal": "day", "people": people if people is not None else [
                {"person_id": 1, "bbox": [0., 0., 10., 10.], "size_band": "50to99"}],
            "ignore": ignore or []}


def prediction(box=None, score=.9):
    return {"bbox_xyxy": box or [0., 0., 10., 10.], "confidence": score}


class BboxAcceptance(unittest.TestCase):
    def test_perfect_ap_and_fixed_counts(self):
        result = fixed(frame(), [prediction()])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))
        self.assertAlmostEqual(ap([frame()], [[prediction()]])["box_ap50_95"], 1.)

    def test_half_iou_passes_only_first_of_ten_ap_thresholds(self):
        p = prediction([0., 0., 5., 10.])
        self.assertEqual(box_iou(p["bbox_xyxy"], frame()["people"][0]["bbox"]), .5)
        self.assertEqual(fixed(frame(), [p])["tp"], 1)
        self.assertEqual(fixed(frame(), [p], threshold=.75)["tp"], 0)
        self.assertAlmostEqual(ap([frame()], [[p]])["box_ap50_95"], .1)

    def test_missed_and_wrong_localization(self):
        missed = fixed(frame(), [])
        self.assertEqual((missed["tp"], missed["fp"], missed["fn"]), (0, 0, 1))
        wrong = fixed(frame(), [prediction([20., 20., 30., 30.])])
        self.assertEqual((wrong["tp"], wrong["fp"], wrong["fn"]), (0, 1, 1))
        self.assertEqual(ap([frame()], [[]])["box_ap50_95"], 0.)

    def test_duplicate_is_fp_but_late_fp_does_not_reduce_single_gt_ap(self):
        predictions = [prediction(), prediction(score=.8)]
        result = fixed(frame(), predictions)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertEqual(result["precision"], .5)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertAlmostEqual(ap([frame()], [predictions])["box_ap50_95"], 1.)

    def test_high_confidence_false_positive_reduces_ap(self):
        predictions = [prediction([20., 20., 30., 30.], .95), prediction(score=.8)]
        self.assertAlmostEqual(ap([frame()], [predictions])["box_ap50_95"], .5)

    def test_ignore_union_and_valid_gt_priority(self):
        self.assertEqual(ignore_ioa([0, 0, 10, 10], [[0, 0, 6, 10], [4, 0, 10, 10]]), 1.)
        f = frame(ignore=[[0, 0, 10, 10], [20, 20, 30, 30]])
        predictions = [prediction([20, 20, 30, 30], .95), prediction()]
        result = fixed(f, predictions)
        self.assertEqual((result["tp"], result["fp"], result["ignored_predictions"]), (1, 0, 1))
        self.assertAlmostEqual(ap([f], [predictions])["box_ap50_95"], 1.)

    def test_empty_domain_null_and_id_renaming(self):
        result = fixed(frame(people=[]), [])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])
        self.assertIsNone(ap([frame(people=[])], [[]])["box_ap50_95"])
        renamed = frame()
        renamed["people"][0]["person_id"] = 87
        self.assertEqual(fixed(frame(), [prediction()]), fixed(renamed, [prediction()]))

    def test_maxdet_summary_uses_requested_cap(self):
        self.assertAlmostEqual(ap([frame()], [[prediction()]], max_dets=200)["box_ap50_95"], 1.)
        self.assertAlmostEqual(ap([frame()], [[prediction()]], max_dets=1000)["box_ap50_95"], 1.)

    def test_exact50_and_clipped_gt_keep_source_eligibility(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, annotation = root / "image.png", root / "annotation.json"
            Image.new("RGB", (130, 90)).save(image)
            document = {"imageWidth": 130, "imageHeight": 90, "shapes": [
                {"label": "bbox", "group_id": 1, "points": [[-.2, 0], [49.8, 20]]},
                {"label": "bbox", "group_id": 2, "points": [[70, 0], [100, 49.9]]}]}
            annotation.write_text(json.dumps(document))
            row = {"image_id": "synthetic", "image_path": str(image), "annotation_path": str(annotation),
                   "image_sha256": sha(image), "annotation_sha256": sha(annotation),
                   "diurnal_metadata": "day", "site_code": "S", "camera_code": "C"}
            result = frame_from_row(row, 1)
            self.assertEqual(len(result["people"]), 1)
            self.assertEqual(result["people"][0]["bbox"][0], 0.)
            self.assertEqual(len(result["excluded_people"]), 1)


class PacketAcceptance(unittest.TestCase):
    def fixture(self, directory):
        f = frame()
        mask = np.zeros((90, 130), dtype=np.uint8)
        mask[1:7, 2:9] = 1
        packed = PackedPrediction(.9, 0, [0., 0., 10., 10.],
            np.packbits(mask.ravel(), bitorder="little"), mask.shape)
        prepared = {"run_id": "synthetic", "frames": [f], "config": config(),
                    "scientific_binding": "binding", "checkpoints": {"synthetic.pt": {"sha256": "weights"}}}
        tier = {"repository": str(Path(directory)), "id": "synthetic", "models": ["synthetic.pt"]}
        binding = model_binding(prepared, "synthetic.pt")
        record = {"image_id": f["image_id"], "image_sha256": f["image_sha256"], "original_wh": [130, 90],
            "input_shape": [1, 3, 640, 640], "binding": binding, "predictions": [{"person_id": 1, **packed.compact()}],
            "post_nms_candidates": 1,
            "timing": {key: 1. for key in ["input_read_decode_ms", "preprocess_ms", "inference_ms",
                                          "postprocess_ms", "total_ms", "rle_preparation_ms"]}}
        path = model_directory(prepared, tier, "synthetic.pt") / "predictions/synthetic.json.gz"
        return f, mask, prepared, tier, record, path

    def test_asymmetric_native_packet_to_scorer_and_json_null(self):
        with tempfile.TemporaryDirectory() as directory:
            f, mask, prepared, tier, record, path = self.fixture(directory)
            save_packet(path, record)
            loaded = read_packet(path, f, record["binding"])
            rle = loaded["predictions"][0]["rle"]
            decoded = coco_mask.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")})
            np.testing.assert_array_equal(decoded, mask)
            result = score_model(prepared, tier, "synthetic.pt")
            self.assertEqual(result["state"], "COMPLETE")
            self.assertAlmostEqual(result["primary"]["box_ap50_95"], 1.)
            self.assertIsNone(result["active_timing"]["application_ms_seconds"])
            reread = json.loads((path.parent.parent / "METRICS.json").read_text())
            self.assertIsNone(reread["active_timing"]["application_ms_seconds"])
            self.assertTrue((path.parent.parent / "PER_IMAGE.csv").is_file())

    def test_immutable_packet_and_invalid_native_class(self):
        with tempfile.TemporaryDirectory() as directory:
            f, _, _, _, record, path = self.fixture(directory)
            save_packet(path, record)
            original = sha(path)
            with self.assertRaises(FileExistsError):
                save_packet(path, record)
            self.assertEqual(sha(path), original)
            invalid = copy.deepcopy(record)
            invalid["predictions"][0]["class_id"] = 255
            bad = path.parent / "bad.json.gz"
            save_packet(bad, invalid)
            with self.assertRaises(ValueError):
                read_packet(bad, f, record["binding"])

    def test_missing_image_keeps_primary_null(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, prepared, tier, _, _ = self.fixture(directory)
            result = score_model(prepared, tier, "synthetic.pt")
            self.assertEqual(result["membership"]["missing"], 1)
            self.assertIsNone(result["primary"])

    def test_reused_native_gpu_bitpack_logic_on_cpu(self):
        import torch
        from cctv600.yolo_runtime import pack_binary_masks
        masks = np.zeros((2, 9, 13), dtype=np.uint8)
        masks[0, 2, 7] = 1
        masks[1, 7, 2] = 1
        packed = pack_binary_masks(torch.from_numpy(masks)).numpy()
        for index in range(2):
            item = PackedPrediction(.9, 0, [0, 0, 13, 9], packed[index], (9, 13))
            np.testing.assert_array_equal(item.mask, masks[index].astype(bool))

    def test_production_packet_writer_receipt_scorer_and_public_report(self):
        from PIL import Image
        from cctv600 import engine, reporting
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            master = root / "master"
            master.mkdir()
            f = frame()
            image = root / "synthetic.png"
            Image.new("RGB", (130, 90)).save(image)
            f.update(image_path=str(image), image_sha256=sha(image))
            mask = np.zeros((90, 130), dtype=np.uint8)
            mask[:10, :10] = 1
            item = PackedPrediction(.9, 0, [0., 0., 10., 10.], np.packbits(mask.ravel(), bitorder="little"), mask.shape)
            tier = {"repository": "Tier", "id": "synthetic", "models": ["synthetic.pt"]}
            cfg = copy.deepcopy(config())
            cfg["tiers"] = [tier]
            prepared = {"run_id": "synthetic", "frames": [f], "config": cfg,
                "scientific_binding": "binding", "environment": {"packages": {}},
                "checkpoints": {"synthetic.pt": {"sha256": "weights", "bytes": 100}}}
            adapter = SimpleNamespace(checkpoint_parameters=10, runtime_parameters=10,
                predictor=SimpleNamespace(args=SimpleNamespace(device="cpu"), post_nms_candidates=1),
                predict=lambda image, path: ([item], {"input_height": 640, "input_width": 640,
                    "preprocess_ms": 1., "inference_ms": 2., "postprocess_ms": 3., "total_ms": 6.}))
            run_dir = master / "run"
            attempt = run_dir / "attempt"
            attempt.mkdir(parents=True)
            with patch.object(reporting, "WORKSPACE", root), patch.object(reporting, "STUDY", master), \
                 patch.object(engine, "verify_source"), patch.object(engine, "create_adapter", return_value=adapter), \
                 patch.object(engine, "release"), patch.object(engine, "gpu", return_value={}):
                engine.quality(prepared, tier, "synthetic.pt", [f], run_dir, attempt, time.monotonic())
                result = score_model(prepared, tier, "synthetic.pt")
                rows = reporting.write_reports(prepared)
            self.assertEqual(result["active_timing"]["application_ms_coverage"], 1)
            self.assertAlmostEqual(rows[0]["box_ap50_95"], 1.)
            public = json.loads((root / "Tier/metrics/RESULTS.json").read_text())
            self.assertAlmostEqual(public["models"][0]["primary"]["box_ap50_95"], 1.)
            self.assertIsNone(rows[0]["fps"])


if __name__ == "__main__":
    unittest.main()
