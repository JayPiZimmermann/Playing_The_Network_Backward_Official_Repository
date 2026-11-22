"""Run the control tasks with a shared, atomic directory-claim queue.

Each task runs in a separate process from a frozen source snapshot. Failures
remain visible and are not retried silently; per-image runners are resumable.
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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("queue",type=Path)
    args=ap.parse_args()
    manifest=json.loads(args.queue.read_text())
    root=args.queue.parent
    for task in manifest["tasks"]:
        state=root/"tasks"/task["id"]
        state.parent.mkdir(parents=True,exist_ok=True)
        try: state.mkdir()
        except FileExistsError: continue
        status={"task":task,"host":socket.gethostname(),"pid":os.getpid(),
                "job":os.environ.get("SLURM_JOB_ID"),"started":time.time(),"status":"running"}
        (state/"status.json").write_text(json.dumps(status,indent=2)+"\n")
        with (state/"output.log").open("w") as log:
            try:
                code=subprocess.run([sys.executable,*task["argv"]],
                                    cwd=manifest["snapshot"],stdout=log,stderr=subprocess.STDOUT).returncode
            except Exception as exc:
                log.write(repr(exc)+"\n"); code=-1
        status.update(status="complete" if code==0 else "failed",exit_code=code,finished=time.time())
        (state/"status.json").write_text(json.dumps(status,indent=2)+"\n")
        print(task["id"],status["status"],flush=True)


if __name__=="__main__": main()
