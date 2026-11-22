"""Prioritize the VGG-16 RG robustness sweeps.

This worker reads the shared task queue and atomically claims only the
VGG-16 RG robustness tasks.  Validation tasks run first; test shards follow.
It uses the same task directories as the ordinary workers, so work cannot be
duplicated even when the queues overlap.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def is_vgg_rg_robustness(task: dict) -> bool:
    if task.get("kind") != "robustness" or task.get("model") != "vgg16":
        return False
    argv = task.get("argv", [])
    if "--row" not in argv:
        return False
    return argv[argv.index("--row") + 1].startswith("rg__")


def split_priority(task: dict) -> tuple:
    argv = task["argv"]
    split = argv[argv.index("--split") + 1]
    return (0 if split == "val" else 1, task["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.queue.read_text())
    root = args.queue.parent
    tasks = sorted(
        (task for task in manifest["tasks"] if is_vgg_rg_robustness(task)),
        key=split_priority,
    )
    for task in tasks:
        state = root / "tasks" / task["id"]
        state.parent.mkdir(parents=True, exist_ok=True)
        try:
            state.mkdir()
        except FileExistsError:
            continue
        status = {
            "task": task,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "job": os.environ.get("SLURM_JOB_ID"),
            "started": time.time(),
            "status": "running",
            "priority_worker": "vgg-rg-robustness-v1",
        }
        (state / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        with (state / "output.log").open("w") as log:
            try:
                code = subprocess.run(
                    [sys.executable, *task["argv"]],
                    cwd=manifest["snapshot"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ).returncode
            except Exception as exc:
                log.write(repr(exc) + "\n")
                code = -1
        status.update(
            status="complete" if code == 0 else "failed",
            exit_code=code,
            finished=time.time(),
        )
        (state / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(task["id"], status["status"], flush=True)


if __name__ == "__main__":
    main()
