"""Explicit, single-attempt timing continuation using the frozen measurement code.

The operator pauses the owned web worker first. This detached controller always
requests restoration of that worker on exit. It never starts quality inference.
"""
import copy
import fcntl
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "src"))

import numpy as np

from cctv600.common import load, now, sha, verify_source, write
from cctv600.engine import clean_timing, gpu, NMSWarningGuard, process_token, check_output_budget
from cctv600.reporting import model_binding, model_directory


def main():
    directory = Path(sys.argv[1]).resolve()
    manifest = load(directory / "MANIFEST.json")
    parent = Path(manifest["parent_run_dir"])
    prepared = load(parent / "PREPARED.json")
    try:
        lock = (STUDY / "outputs/controller.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_source(prepared)
        assert sha(parent / "PREPARED.json") == manifest["parent_prepared_sha256"]
        assert sha(Path(__file__)) == manifest["controller_sha256"]
        assert prepared["scientific_binding"] == manifest["scientific_binding"]
        for path, digest in prepared["environment"]["ultralytics_sources"].items():
            assert sha(Path(path)) == digest
        state = gpu()
        assert not state["external_processes"], "External GPU processes still present"
        assert state["free_mib"] >= prepared["config"]["minimum_free_vram_mib"]
        check_output_budget(prepared)
        local = copy.deepcopy(prepared)
        local["run_id"] = directory.name
        assert {key: value for key, value in local.items() if key != "run_id"} == {
            key: value for key, value in prepared.items() if key != "run_id"}
        write(directory / "PREPARED.json", local)
        os.environ["YOLO_CONFIG_DIR"] = str(directory / "ultralytics_config")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        import torch
        from ultralytics.utils import LOGGER
        LOGGER.addHandler(NMSWarningGuard())
        random.seed(prepared["config"]["seed"])
        np.random.seed(prepared["config"]["seed"])
        torch.manual_seed(prepared["config"]["seed"])
        torch.backends.cudnn.benchmark = prepared["config"]["cudnn_benchmark"]
        torch.backends.cuda.matmul.allow_tf32 = prepared["config"]["allow_tf32"]
        torch.backends.cudnn.allow_tf32 = prepared["config"]["allow_tf32"]
        launch = {"at_utc": now(), "pid": os.getpid(), "session_id": os.getsid(0),
            "process_start_token": process_token(os.getpid()), "command": sys.argv,
            "launch_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=STUDY, text=True).strip(),
            "gpu": state, "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_num_threads": torch.get_num_threads(), "matmul_tf32": False, "cudnn_tf32": False}
        write(directory / "LAUNCH.json", launch)
        attempt = directory / "attempts" / "explicit-timing-continuation-1"
        attempt.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        completed = []
        for item in manifest["pending"]:
            tier = next(t for t in local["config"]["tiers"] if item["model"] in t["models"])
            original = model_directory(prepared, tier, item["model"]) / "timing" / f"round{item['round']}.json"
            if original.exists():
                assert load(original)["state"] != "CLEAN_COMPLETE", "Refuse to repeat accepted timing"
                assert sha(original) == item["original_sha256"]
            else:
                assert item["original_sha256"] is None
            assert clean_timing(local, tier, item["model"], item["round"], directory, attempt, started), "Timing deferred or contaminated; no automatic retry"
            output = model_directory(local, tier, item["model"]) / "timing" / f"round{item['round']}.json"
            record = load(output)
            assert record["binding"] == model_binding(prepared, item["model"])
            completed.append({**item, "path": str(output), "sha256": sha(output)})
            check_output_budget(local)
        write(directory / "COMPLETION.json", {"state": "COMPLETE", "at_utc": now(),
            "scientific_binding": prepared["scientific_binding"], "parent_run_id": prepared["run_id"],
            "completed": completed, "measured_model_calls": 400, "warmup_calls": 40,
            "quality_calls": 0, "automatic_retries": 0})
    except BaseException:
        write(directory / "FAILURE.json", {"at_utc": now(), "traceback": traceback.format_exc(),
            "automatic_retries": 0})
        raise
    finally:
        restored = subprocess.run(["systemctl", "--user", "start", "body-extraction-worker.service"],
            capture_output=True, text=True)
        write(directory / "WEB_WORKER_RESTORATION.json", {"at_utc": now(), "returncode": restored.returncode,
            "stdout": restored.stdout, "stderr": restored.stderr})


if __name__ == "__main__":
    main()
