"""One-worker quality batch, saved-packet scoring and separate clean timing."""
import csv
import fcntl
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from .common import config, event, identity, immutable_bytes, load, now, sha, STUDY, WORKSPACE, verify_source, write
from .packets import read_packet, save_packet
from .reporting import model_binding, model_directory, score_model, write_reports
from .scoring import ap


def gpu():
    state = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    first = [part.strip() for part in state.stdout.splitlines()[0].split(",")]
    external = [line for line in processes.stdout.splitlines() if line and int(line.split(",")[0]) != os.getpid()]
    return {"at_utc": now(), "name": first[1], "free_mib": int(first[2]),
            "utilization_percent": int(first[3]), "external_processes": external,
            "raw_gpu": state.stdout, "raw_processes": processes.stdout}


def process_token(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def live(record):
    return record.get("pid") is not None and process_token(record["pid"]) == record.get("process_start_token")


def update(directory, **values):
    path = directory / "STATUS.json"
    previous = load(path) if path.exists() else {}
    previous.update(values, at_utc=now(), pid=os.getpid(), process_start_token=process_token(os.getpid()))
    write(path, previous)
    print(json.dumps({key: previous.get(key) for key in
        ["at_utc", "state", "tier", "model", "completed", "expected", "phase", "round"]}), flush=True)


def guard(directory, prepared, start):
    cfg = prepared["config"]
    if (directory / "STOP_REQUEST.json").exists():
        raise RuntimeError("Controlled stop requested at image boundary")
    if time.monotonic() - start > cfg["maximum_execution_hours"] * 3600:
        raise RuntimeError("Study execution budget exceeded")
    if shutil.disk_usage(STUDY).free < cfg["minimum_free_disk_gib"] * 1024**3:
        raise RuntimeError("Free disk guard failed")


def check_output_budget(prepared):
    directories = [STUDY / "outputs" / prepared["run_id"]]
    directories.extend(WORKSPACE / tier["repository"] / "outputs" / prepared["run_id"]
                       for tier in prepared["config"]["tiers"])
    size = sum(path.stat().st_size for directory in directories for path in directory.rglob("*") if path.is_file())
    if size > prepared["config"]["maximum_study_output_gib"] * 1024**3:
        raise RuntimeError("Study output disk budget exceeded")


class NMSWarningGuard(logging.Handler):
    def emit(self, record):
        message = record.getMessage()
        if "NMS" in message and ("time" in message.lower() or "limit" in message.lower()):
            raise RuntimeError("NMS candidate retention warning: " + message)


def create_adapter(prepared, model, framework_directory):
    import torch
    from .detector import Adapter
    if gpu()["free_mib"] < prepared["config"]["minimum_free_vram_mib"]:
        raise RuntimeError("Free VRAM admission guard failed")
    if sha(WORKSPACE / "models" / model) != prepared["checkpoints"][model]["sha256"]:
        raise RuntimeError("Checkpoint changed")
    adapter = Adapter(WORKSPACE / "models" / model, prepared["config"], framework_directory)
    adapter.sync()
    expected_parameters = prepared["checkpoints"][model].get("reference_parameters")
    if expected_parameters is not None and adapter.checkpoint_parameters != expected_parameters:
        raise RuntimeError("Loaded architecture parameter count differs from checkpoint binding")
    return adapter


def release():
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def decode_frame(frame):
    import cv2
    body = Path(frame["image_path"]).read_bytes()
    if hashlib.sha256(body).hexdigest() != frame["image_sha256"]:
        raise ValueError("Actual inference input hash differs")
    image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (frame["height"], frame["width"]):
        raise ValueError("Input image decode/geometry failed")
    return image


def quality(prepared, tier, model, selected, directory, attempt, started):
    import cv2
    cfg = prepared["config"]
    output = model_directory(prepared, tier, model)
    output.mkdir(parents=True, exist_ok=True)
    binding = model_binding(prepared, model)
    pending = []
    for frame in selected:
        path = output / "predictions" / (frame["image_id"] + ".json.gz")
        if path.exists():
            read_packet(path, frame, binding)
            event(attempt / "ledger.jsonl", {"at_utc": now(), "event": "REUSE_VERIFIED_PACKET",
                "model": model, "image_id": frame["image_id"], "packet_sha256": sha(path)})
        else:
            pending.append(frame)
    if not pending:
        return
    verify_source(prepared)
    for path, expected_hash in prepared["environment"].get("ultralytics_sources", {}).items():
        if sha(path) != expected_hash:
            raise RuntimeError("Ultralytics source changed")
    import importlib.metadata
    for package, version in prepared["environment"]["packages"].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError("Environment package binding changed: " + package)
    update(directory, state="LOADING", tier=tier["id"], model=model, phase="QUALITY",
           completed=len(list((output / "predictions").glob("*.json.gz"))), expected=len(prepared["frames"]))
    load_started = time.perf_counter()
    adapter = create_adapter(prepared, model, output / "framework" / (attempt.name + "-" + str(time.time_ns())))
    write(output / ("MODEL_" + attempt.name + "-" + str(time.time_ns()) + ".json"), {"binding": binding,
        "load_seconds": time.perf_counter() - load_started,
        "checkpoint_parameters": adapter.checkpoint_parameters, "runtime_parameters": adapter.runtime_parameters,
        "effective_predictor_args": json.loads(json.dumps(vars(adapter.predictor.args), default=str)), "gpu": gpu()})
    try:
        for frame in pending:
            guard(directory, prepared, started)
            event(attempt / "ledger.jsonl", {"at_utc": now(), "event": "STARTED_FRAME",
                "model": model, "image_id": frame["image_id"]})
            app_start = time.perf_counter()
            image = decode_frame(frame)
            read_end = time.perf_counter()
            predictions, timing = adapter.predict(image, Path(frame["image_path"]))
            if [timing["input_height"], timing["input_width"]] != [640, 640]:
                raise ValueError("Actual network input changed")
            rle_start = time.perf_counter()
            compact = [{"person_id": index + 1, **prediction.compact()} for index, prediction in enumerate(predictions)]
            timing["rle_preparation_ms"] = (time.perf_counter() - rle_start) * 1000
            timing["input_read_decode_ms"] = (read_end - app_start) * 1000
            record = {"image_id": frame["image_id"], "image_sha256": frame["image_sha256"],
                "original_wh": [frame["width"], frame["height"]], "input_shape": [1, 3, 640, 640],
                "binding": binding, "at_utc": now(), "attempt": attempt.name,
                "post_nms_candidates": adapter.predictor.post_nms_candidates,
                "predictions": compact, "timing": timing}
            path = output / "predictions" / (frame["image_id"] + ".json.gz")
            write_start = time.perf_counter()
            save_packet(path, record)
            written = time.perf_counter()
            packet_hash = sha(path)
            receipt = {"binding": binding, "packet_sha256": packet_hash, "at_utc": now(),
                "timing": {"output_write_ms": (written - write_start) * 1000,
                           "application_ms": (written - app_start) * 1000}}
            receipt_path = output / "receipts" / (frame["image_id"] + ".json")
            immutable_bytes(receipt_path, json.dumps(receipt, allow_nan=False).encode())
            read_packet(path, frame, binding)
            event(attempt / "ledger.jsonl", {"at_utc": now(), "event": "ACCEPTED_PACKET",
                "model": model, "image_id": frame["image_id"], "packet_sha256": packet_hash})
            count = len(list((output / "predictions").glob("*.json.gz")))
            if count % 25 == 0:
                check_output_budget(prepared)
            update(directory, state="QUALITY_RUNNING", completed=count,
                   latest_image=frame["image_id"], gpu=gpu() if count % 25 == 0 else None)
            del predictions, compact, image
    except BaseException as error:
        event(attempt / "ledger.jsonl", {"at_utc": now(), "event": "FAILED_FRAME",
            "model": model, "image_id": frame["image_id"], "error": repr(error)})
        raise
    finally:
        del adapter
        release()


def cap_check(prepared, tier, model):
    frames_by_id = {frame["image_id"]: frame for frame in prepared["frames"]}
    frames = [frames_by_id[key] for key in prepared["sample"]]
    output = model_directory(prepared, tier, model)
    predictions = [read_packet(output / "predictions" / (frame["image_id"] + ".json.gz"),
        frame, model_binding(prepared, model))["predictions"] for frame in frames]
    cfg = prepared["config"]
    actual = ap(frames, predictions, cfg["ap_max_dets"], cfg["ignore_prediction_ioa"])
    reference = ap(frames, predictions, cfg["ap_cap_reference"], cfg["ignore_prediction_ioa"])
    differences = {key: abs(actual[key] - reference[key]) for key in
        ["box_ap50_95", "box_ap50", "box_ap75"] if actual[key] is not None}
    passed = len(differences) == 3 and all(v < cfg["ap_cap_tolerance"] for v in differences.values())
    write(output / "CAP_CHECK.json", {"state": "PASS" if passed else "FAIL", "maxdet": cfg["ap_max_dets"],
        "reference_maxdet": cfg["ap_cap_reference"], "absolute_differences": differences,
        "sample_membership": prepared["sample"], "binding": model_binding(prepared, model)})
    if not passed:
        raise RuntimeError("Common AP maxDet200 failed reference1000 convergence; protocol review required")


def clean_timing(prepared, tier, model, round_number, directory, attempt, started):
    import cv2
    output = model_directory(prepared, tier, model)
    path = output / "timing" / f"round{round_number}.json"
    if path.exists():
        existing = load(path)
        if existing["state"] == "CLEAN_COMPLETE" and existing["binding"] == model_binding(prepared, model):
            return True
        raise RuntimeError("Existing incomplete/contaminated timing requires fresh controlled segment")
    before = gpu()
    idle_started = time.monotonic()
    for _ in range(prepared["config"]["timing_idle_settle_seconds"]):
        if before["external_processes"] or before["utilization_percent"] <= 5:
            break
        time.sleep(1)
        before = gpu()
    if before["external_processes"] or before["utilization_percent"] > 5:
        update(directory, state="TIMING_DEFERRED_GPU_BUSY", phase="TIMING", tier=tier["id"], model=model,
               round=round_number, gpu=before)
        return False
    verify_source(prepared)
    frames_by_id = {frame["image_id"]: frame for frame in prepared["frames"]}
    frames = [frames_by_id[key] for key in prepared["sample"]]
    update(directory, state="TIMING_LOADING", phase="TIMING", tier=tier["id"], model=model,
           round=round_number, completed=0, expected=len(frames))
    idle_wait_seconds = time.monotonic() - idle_started
    load_start = time.perf_counter()
    adapter = create_adapter(prepared, model, output / "timing_framework" / f"round{round_number}")
    record = {"binding": model_binding(prepared, model), "round": round_number,
              "state": "RUNNING", "load_seconds": time.perf_counter() - load_start,
              "idle_wait_seconds": idle_wait_seconds,
              "observations": [], "gpu_samples": [before], "warmup_calls": 0,
              "parameters": adapter.checkpoint_parameters}
    try:
        for frame in frames[:prepared["config"]["warmup_iterations"]]:
            guard(directory, prepared, started)
            image = decode_frame(frame)
            predictions, _ = adapter.predict(image, Path(frame["image_path"]))
            record["warmup_calls"] += 1
            del predictions, image
        adapter.sync()
        adapter.reset_peak_memory()
        contaminated = False
        for index, frame in enumerate(frames):
            guard(directory, prepared, started)
            if index % 10 == 0:
                state = gpu()
                record["gpu_samples"].append(state)
                contaminated |= bool(state["external_processes"])
            image = decode_frame(frame)
            predictions, timing = adapter.predict(image, Path(frame["image_path"]))
            start = time.perf_counter()
            compact = [prediction.compact() for prediction in predictions]
            timing["rle_preparation_ms"] = (time.perf_counter() - start) * 1000
            record["observations"].append({"image_id": frame["image_id"], **timing})
            update(directory, state="TIMING_RUNNING", completed=index + 1)
            del predictions, compact, image
        record["memory"] = adapter.peak_memory()
        after = gpu()
        record["gpu_samples"].append(after)
        contaminated |= bool(after["external_processes"])
        record["state"] = "CONTAMINATED" if contaminated else "CLEAN_COMPLETE"
        immutable_bytes(path, json.dumps(record, allow_nan=False).encode())
        return not contaminated
    except BaseException:
        record.update(state="FAILED", traceback=traceback.format_exc())
        immutable_bytes(path, json.dumps(record, allow_nan=False).encode())
        raise
    finally:
        del adapter
        release()


def publish(prepared):
    repositories = [STUDY] + [WORKSPACE / t["repository"] for t in prepared["config"]["tiers"]]
    for repository in repositories:
        files = [name for name in ["STATUS.md", "MASTER_RESULTS.md", "RESULTS_SUMMARY_TH.md", "metrics"]
                 if (repository / name).exists()]
        subprocess.run(["git", "add", "--", *files], cwd=repository, check=True)
        changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repository)
        if changed.returncode:
            subprocess.run(["git", "commit", "-m", "Record accepted CCTV600 benchmark results"], cwd=repository, check=True)
            subprocess.run(["git", "push", "origin", "main"], cwd=repository, check=True)


def run(directory, publish_results=False, tier_scope=None, resume=False):
    directory = Path(directory).resolve()
    prepared = load(directory / "PREPARED.json")
    lock = (STUDY / "outputs/controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = load(directory / "STATUS.json") if (directory / "STATUS.json").exists() else {}
    if live(previous):
        raise RuntimeError("Existing controller is alive")
    if previous and not resume:
        raise RuntimeError("Explicit controlled continuation is required for an existing attempt")
    if resume and not list(directory.glob("RECONCILIATION-*.json")):
        raise RuntimeError("Read-only reconciliation required before controlled continuation")
    if (directory / "STOP_REQUEST.json").exists():
        raise RuntimeError("Outstanding operator stop request requires acknowledgement before continuation")
    verify_source(prepared)
    acceptance = load(STUDY / "manifests/CPU_ACCEPTANCE.json")
    if prepared["config"]["quality_warmup_calls"] != 0:
        raise RuntimeError("Quality warmup policy mismatch")
    if acceptance["state"] != "PASS" or acceptance["sources"] != prepared["sources"]:
        raise RuntimeError("CPU acceptance binding mismatch")
    selected_tiers = [tier for tier in prepared["config"]["tiers"]
                      if tier_scope is None or tier_scope in {tier["id"], tier["repository"]}]
    if not selected_tiers:
        raise ValueError("Unknown tier scope")
    expected_models = sum(len(tier["models"]) for tier in selected_tiers)
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
    attempt = directory / "attempts" / ("attempt-" + str(time.time_ns()))
    attempt.mkdir(parents=True)
    started = time.monotonic()
    update(directory, state="STARTING", phase="QUALITY", completed=0, expected=600,
           attempt=attempt.name, gpu=gpu())
    write(attempt / "LAUNCH.json", {"at_utc": now(), "pid": os.getpid(), "ppid": os.getppid(),
        "session_id": os.getsid(0), "process_start_token": process_token(os.getpid()),
        "command": sys.argv, "scientific_binding": prepared["scientific_binding"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": gpu(),
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "torch_num_threads": torch.get_num_threads(), "matmul_tf32": False, "cudnn_tf32": False,
        "launch_commits": {str(path.name): subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
            for path in [STUDY] + [WORKSPACE / t["repository"] for t in prepared["config"]["tiers"]]}})
    try:
        check_output_budget(prepared)
        sample_ids = set(prepared["sample"])
        by_id = {frame["image_id"]: frame for frame in prepared["frames"]}
        sample = [by_id[key] for key in prepared["sample"]]
        for tier in selected_tiers:
            for model in tier["models"]:
                quality(prepared, tier, model, sample, directory, attempt, started)
                cap_check(prepared, tier, model)
            pending_frames = [frame for frame in prepared["frames"] if frame["image_id"] not in sample_ids]
            for model in tier["models"]:
                quality(prepared, tier, model, pending_frames, directory, attempt, started)
            for model in tier["models"]:
                update(directory, state="SAVED_PACKET_SCORING", tier=tier["id"], model=model, phase="SCORING")
                result = score_model(prepared, tier, model)
                if result["state"] != "COMPLETE":
                    raise RuntimeError("STRICT_COMPLETE quality gate failed")
        update(directory, state="QUALITY_COMPLETE", phase="TIMING", completed=expected_models, expected=expected_models)
        timing_ok = True
        for tier in selected_tiers:
            models = list(tier["models"])
            random.Random(prepared["config"]["seed"]).shuffle(models)
            for number in range(1, prepared["config"]["timing_repeats"] + 1):
                order = models[number - 1:] + models[:number - 1]
                for model in order:
                    if not clean_timing(prepared, tier, model, number, directory, attempt, started):
                        timing_ok = False
                        break
                if not timing_ok:
                    break
            if not timing_ok:
                break
        verify_source(prepared)
        rows = write_reports(prepared)
        from .visuals import gallery, plots
        for tier in selected_tiers:
            gallery(prepared, tier)
        plots(rows)
        state = "COMPLETE" if timing_ok else "QUALITY_COMPLETE_TIMING_INCOMPLETE"
        update(directory, state=state, phase="DONE", completed=expected_models, expected=expected_models,
               tier_scope=[tier["id"] for tier in selected_tiers])
        write(directory / "COMPLETION.json", {"state": state, "at_utc": now(),
            "quality_models_complete": expected_models, "timing_complete": timing_ok,
            "tier_scope": [tier["id"] for tier in selected_tiers],
            "scientific_binding": prepared["scientific_binding"], "attempt": attempt.name})
        if publish_results:
            try:
                publish(prepared)
                write(directory / "PUBLICATION.json", {"state": "PUSHED", "at_utc": now()})
            except Exception as publication_error:
                write(directory / "PUBLICATION.json", {"state": "FAILED", "at_utc": now(),
                    "error": repr(publication_error), "scientific_results_state": state})
    except BaseException as error:
        try:
            failure_gpu = gpu()
        except Exception as capture_error:
            failure_gpu = {"capture_error": repr(capture_error)}
        failure = {"at_utc": now(), "exception": repr(error), "traceback": traceback.format_exc(),
                   "gpu": failure_gpu, "status": load(directory / "STATUS.json")}
        write(attempt / "FAILURE.json", failure)
        update(directory, state="STOPPED_REQUIRES_RECONCILIATION", error=repr(error), phase="STOPPED")
        write_reports(prepared)
        if publish_results:
            try:
                publish(prepared)
            except Exception as publication_error:
                write(directory / "PUBLICATION.json", {"state": "FAILED", "at_utc": now(), "error": repr(publication_error)})
        raise
    finally:
        lock.close()
