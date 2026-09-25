"""Freeze the completed SocialMemBench Experiment 1 with reproducibility metadata."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / ".research_runs" / "socialmembench_full_official_v1"
DATA_DIR = Path("/tmp/socialmembench")
FREEZE_ID = "socialmembench_full_official_v1_20260911_184037_MSK"
DESTINATION = ROOT / ".research_runs" / "frozen" / FREEZE_ID


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as source:
        return sum(1 for _ in source)


def command(*args: str) -> str:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout + result.stderr


def validate_completed_run() -> dict[str, int]:
    counts = {
        "session_assertions": line_count(SOURCE_RUN / "session_assertions.jsonl"),
        "predictions": line_count(SOURCE_RUN / "predictions.jsonl"),
        "results": line_count(SOURCE_RUN / "results.jsonl"),
    }
    expected = {"session_assertions": 348, "predictions": 1031, "results": 3093}
    if counts != expected:
        raise RuntimeError(f"run is incomplete: expected {expected}, got {counts}")
    results = [json.loads(line) for line in (SOURCE_RUN / "results.jsonl").read_text().splitlines()]
    unique = {(row["qa_key"], row["condition"]) for row in results}
    if len(unique) != 3093:
        raise RuntimeError(f"results identity check failed: {len(unique)} unique rows")
    if "score: final report ready" not in (SOURCE_RUN / "run.log").read_text():
        raise RuntimeError("run log has no completion marker")
    return counts


def freeze() -> Path:
    counts = validate_completed_run()
    if DESTINATION.exists():
        raise RuntimeError(f"freeze already exists: {DESTINATION}")

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_RUN, DESTINATION / "run")
    shutil.copytree(ROOT / "research", DESTINATION / "source" / "research")
    shutil.copytree(ROOT / "tests", DESTINATION / "source" / "tests")
    shutil.copytree(ROOT / "docs", DESTINATION / "source" / "docs")
    for relative in ("config.py", "embeddings.py", "llm/groq_client.py", "requirements.txt"):
        target = DESTINATION / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    dataset_target = DESTINATION / "dataset"
    dataset_target.mkdir()
    for name in ("conversations.parquet", "qa.parquet", "networks.parquet", "personas.parquet"):
        shutil.copy2(DATA_DIR / name, dataset_target / name)

    metadata = DESTINATION / "metadata"
    metadata.mkdir()
    (metadata / "pip-freeze.txt").write_text(command(sys.executable, "-m", "pip", "freeze"))
    (metadata / "git-status.txt").write_text(command("git", "status", "--short"))
    (metadata / "git-diff.patch").write_text(command("git", "diff", "--binary"))
    (metadata / "git-ls-files.txt").write_text(command("git", "ls-files", "-s"))
    system = {
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_head": command("git", "rev-parse", "HEAD").strip(),
        "git_branch": command("git", "rev-parse", "--abbrev-ref", "HEAD").strip(),
    }
    (metadata / "system.json").write_text(json.dumps(system, indent=2, ensure_ascii=False) + "\n")

    readme = f"""# Frozen SocialMemBench Experiment 1

- Freeze ID: `{FREEZE_ID}`
- Completed run: 2026-09-11 18:40:37 MSK
- Sessions: {counts['session_assertions']}
- QA predictions: {counts['predictions']}
- Scored condition rows: {counts['results']}
- Conditions: RAW, RAW+FLAT, RAW+VERSIONED
- This directory is a read-only preservation copy. Do not use it as a writable cache.

Contents:

- `run/`: every completed run artifact, including embeddings and API cache
- `dataset/`: exact public parquet inputs used by the run
- `source/`: research code, tests, docs, and direct runtime modules
- `metadata/`: Python environment, git state/diff, system information
- `MANIFEST.json` and `SHA256SUMS`: size and SHA-256 for every preserved payload file

Secrets and `.env` are intentionally excluded.
"""
    (DESTINATION / "README.md").write_text(readme)

    payload_files = sorted(
        path for path in DESTINATION.rglob("*")
        if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS"}
    )
    files = {
        str(path.relative_to(DESTINATION)): {"size": path.stat().st_size, "sha256": sha256(path)}
        for path in payload_files
    }
    manifest = {
        "freeze_id": FREEZE_ID,
        "source_run": str(SOURCE_RUN),
        "counts": counts,
        "file_count": len(files),
        "total_payload_bytes": sum(item["size"] for item in files.values()),
        "files": files,
    }
    (DESTINATION / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (DESTINATION / "SHA256SUMS").write_text(
        "".join(f"{item['sha256']}  {name}\n" for name, item in files.items())
    )

    for path in sorted(DESTINATION.rglob("*"), reverse=True):
        mode = stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH
        if path.is_dir():
            mode |= stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
        os.chmod(path, mode)
    os.chmod(DESTINATION, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH |
             stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return DESTINATION


if __name__ == "__main__":
    print(freeze())
