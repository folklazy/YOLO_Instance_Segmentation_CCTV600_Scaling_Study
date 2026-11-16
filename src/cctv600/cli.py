"""Preparation, detached launch and read-only progress commands."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .common import config, load, now, sha, STUDY, WORKSPACE, verify_source, write
from .data import prepare
from .engine import live, run
from .reporting import model_directory, write_reports


def current(value):
    return Path(value).resolve() if value else Path(load(STUDY / "outputs/CURRENT.json")["run_dir"])


def status(directory, default_tier=None):
    prepared = load(directory / "PREPARED.json")
    record = load(directory / "STATUS.json") if (directory / "STATUS.json").exists() else {"state": "PREPARED"}
    active = live(record)
    terminal = {"COMPLETE", "QUALITY_COMPLETE_TIMING_INCOMPLETE", "STOPPED_REQUIRES_RECONCILIATION"}
    state = record["state"]
    if state not in terminal | {"PREPARED"} and not active:
        state = "INTERRUPTED_REQUIRES_RECONCILIATION"
    print("YOLO CCTV600 |", now(), "|", state)
    print("Run", prepared["run_id"], "| PID", record.get("pid"), "| alive", active)
    print("Phase", record.get("phase"), "|", record.get("tier"), record.get("model"),
          "|", record.get("completed"), "/", record.get("expected"))
    for tier in prepared["config"]["tiers"]:
        if default_tier and tier["repository"] != default_tier:
            continue
        for model in tier["models"]:
            directory_model = model_directory(prepared, tier, model)
            count = len(list((directory_model / "predictions").glob("*.json.gz")))
            result = load(directory_model / "METRICS.json") if (directory_model / "METRICS.json").exists() else {}
            print(tier["id"], model, str(count) + "/600 packets", "| audit", result.get("state", "PENDING"))
    print("Log", directory / "controller.log")
    print("Status", directory / "STATUS.json")
    print("Completion", directory / "COMPLETION.json")


def launch(directory, tier_scope=None, resume=False):
    prepared = load(directory / "PREPARED.json")
    verify_source(prepared)
    record = load(directory / "STATUS.json") if (directory / "STATUS.json").exists() else {}
    if live(record):
        status(directory)
        return
    if record and not resume:
        raise RuntimeError("Existing attempt requires explicit reconciliation before another launch")
    test_record = load(STUDY / "manifests/CPU_ACCEPTANCE.json")
    if test_record["state"] != "PASS" or test_record["sources"] != prepared["sources"]:
        raise RuntimeError("CPU acceptance/source binding mismatch")
    session = "yolo_cctv600_" + prepared["run_id"].removeprefix("cctv600-").split("T")[1][:6]
    command = [sys.executable, "-B", "-u", str(STUDY / "scripts/study.py"),
               "run", "--run-dir", str(directory), "--publish-results"]
    if tier_scope:
        command.extend(["--tier", tier_scope])
    if resume:
        command.append("--resume")
    shell = "cd " + shlex.quote(str(WORKSPACE)) + " && exec " + shlex.join(command)
    shell += " > " + shlex.quote(str(directory / "controller.log")) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    write(directory / "DETACHED_LAUNCH.json", {"at_utc": now(), "session": session,
        "command": command, "shell": shell, "python": sys.executable})
    print("Detached tmux session", session)
    print("Log", directory / "controller.log")


def main(default_tier=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--manifest", required=True, type=Path)
    for name in ["launch", "run", "status", "report", "stop", "reconcile", "visualize"]:
        child = sub.add_parser(name)
        child.add_argument("--run-dir", type=Path)
        if name == "run":
            child.add_argument("--publish-results", action="store_true")
        if name in {"run", "launch"}:
            child.add_argument("--tier", default=default_tier)
            child.add_argument("--resume", action="store_true")
        if name == "visualize":
            child.add_argument("--tier", default=default_tier or "largest")
            child.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.manifest))
        return
    directory = current(args.run_dir)
    if args.command == "launch":
        launch(directory, args.tier, args.resume)
    elif args.command == "run":
        run(directory, args.publish_results, args.tier, args.resume)
    elif args.command == "status":
        status(directory, default_tier)
    elif args.command == "report":
        record = load(directory / "STATUS.json") if (directory / "STATUS.json").exists() else {}
        if live(record):
            raise RuntimeError("Active controller owns report generation")
        write_reports(load(directory / "PREPARED.json"))
    elif args.command == "stop":
        write(directory / "STOP_REQUEST.json", {"at_utc": now(), "reason": "operator_controlled_stop"})
        print("Stop requested at next image boundary")
    elif args.command == "visualize":
        prepared = load(directory / "PREPARED.json")
        from .visuals import gallery
        tier = next(t for t in prepared["config"]["tiers"] if args.tier in {t["id"], t["repository"]})
        print(gallery(prepared, tier, args.limit))
    elif args.command == "reconcile":
        record = load(directory / "STATUS.json") if (directory / "STATUS.json").exists() else {}
        if live(record):
            raise RuntimeError("Controller still active")
        prepared = load(directory / "PREPARED.json")
        from .packets import read_packet
        from .reporting import model_binding
        counts = []
        for tier in prepared["config"]["tiers"]:
            for model in tier["models"]:
                valid = 0
                for frame in prepared["frames"]:
                    path = model_directory(prepared, tier, model) / "predictions" / (frame["image_id"] + ".json.gz")
                    if path.exists():
                        read_packet(path, frame, model_binding(prepared, model))
                        valid += 1
                counts.append({"model": model, "valid": valid, "missing": 600 - valid})
        target = directory / ("RECONCILIATION-" + str(__import__("time").time_ns()) + ".json")
        write(target, {"at_utc": now(), "models": counts, "state": "READ_ONLY_RECONCILIATION"})
        print(target)
