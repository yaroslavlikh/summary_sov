"""One-command launcher for frozen, resumable paper benchmark runs."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from research.paper_benchmark_common import ROOT, freeze_run


MODULES = {
    "group": "research.groupmembench_episode_run",
    "ever": "research.evermembench_episode_run",
}
RUN_DIRS = {
    "research.groupmembench_episode_run": ROOT / ".research_runs" / "groupmembench_temporal_episodes_official_v1",
    "research.evermembench_episode_run": ROOT / ".research_runs" / "evermembench_temporal_episodes_official_v1",
}


def _run_resumable(module: str) -> None:
    while True:
        code = subprocess.run([sys.executable, "-m", module, "--all"], cwd=ROOT).returncode
        if code == 0:
            return
        if code != 75:
            raise SystemExit(code)
        print("API rate limit: весь прогресс сохранён; повтор через 65 секунд", flush=True)
        time.sleep(65)


def _freeze_external(module: str) -> None:
    marker = RUN_DIRS[module] / "paper_freeze_path.txt"
    if marker.exists():
        print(f"already frozen: {marker.read_text().strip()}")
        return
    result = subprocess.run(
        [sys.executable, "-m", module, "--freeze"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    )
    destination = result.stdout.strip().splitlines()[-1]
    marker.write_text(destination + "\n")
    print(f"frozen: {destination}")


def freeze_social() -> Path:
    run = ROOT / ".research_runs" / "socialmembench_temporal_episodes_official_harness_v1"
    marker = run / "paper_freeze_path.txt"
    if marker.exists():
        return Path(marker.read_text().strip())
    destination = freeze_run(
        run,
        [ROOT / "research" / name for name in (
            "socialmembench_full_run.py", "socialmembench_episode_full_run.py",
            "socialmembench_episode_official_harness.py", "temporal_episode_prototype.py",
            "query_independent_episode_pipeline.py", "paper_benchmark_common.py",
            "PAPER_BENCHMARK_PROTOCOL.md",
        )],
        related_dirs=[ROOT / ".research_runs" / "socialmembench_temporal_episodes_full_v1"],
    )
    marker.write_text(str(destination) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--group", action="store_true")
    modes.add_argument("--ever", action="store_true")
    modes.add_argument("--external-all", action="store_true")
    modes.add_argument("--freeze-social", action="store_true")
    args = parser.parse_args()
    if args.freeze_social:
        print(freeze_social()); return
    selected = ["group", "ever"] if args.external_all else ["group" if args.group else "ever"]
    for name in selected:
        _run_resumable(MODULES[name])
        _freeze_external(MODULES[name])


if __name__ == "__main__":
    main()
