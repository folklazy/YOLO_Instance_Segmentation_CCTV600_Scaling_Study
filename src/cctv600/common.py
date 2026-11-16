"""Paths, immutable artifacts and study bindings."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

STUDY = Path(__file__).resolve().parents[2]
WORKSPACE = STUDY.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def write(path, value):
    """Atomic replacement for explicitly mutable status and derived summaries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def immutable_bytes(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def event(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def config():
    return load(STUDY / "configs/study.json")


def source_binding():
    paths = list((STUDY / "src").rglob("*.py")) + list((STUDY / "scripts").glob("*.py"))
    paths.append(STUDY / "configs/study.json")
    paths.append(STUDY / "EXPERIMENT_PROTOCOL.md")
    paths.extend(STUDY / "manifests" / name for name in ["RUNTIME_ORIGIN.json", "COMPLEXITY_REFERENCE.json"])
    for tier in config()["tiers"]:
        paths.extend((WORKSPACE / tier["repository"] / "configs").glob("*.json"))
        paths.extend((WORKSPACE / tier["repository"] / "src").glob("*.py"))
        paths.append(WORKSPACE / tier["repository"] / "EXPERIMENT_PROTOCOL.md")
    return {str(path.relative_to(WORKSPACE)): sha(path) for path in sorted(paths)}


def verify_source(prepared):
    if source_binding() != prepared["sources"]:
        raise RuntimeError("Scientific source/config bindings changed")
