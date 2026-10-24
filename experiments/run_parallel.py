"""Parallel fixed-set MASt3R sweep with shared baselines and process-safe records."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from filelock import FileLock
from metrics import condition_counts
from run import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / "controller.lock"), timeout=0):
        execute(args, out)


def execute(args, out):
    stop_file = out / "stop_requested"
    stop_file.unlink(missing_ok=True)
    state = {"pid": os.getpid(), "status": "running", "started_unix": time.time(),
             "device": args.device, "optimizer_device": args.optimizer_device or args.device,
             "threads_per_worker": args.threads, "worker_count": args.workers, "workers": []}
    processes, logs = [], []

    def request_stop(signum, frame):
        stop_file.touch()
        state["status"] = "stopping"

    signal.signal(signal.SIGTERM, request_stop)

    def refresh():
        rows = []
        backends = ["mast3r"]
        if (out / "vggt/utilities.json").exists():
            backends.append("vggt")
        for backend in backends:
            path = out / backend / "utilities.json"
            if path.exists():
                rows.extend(json.loads(path.read_text()))
        counts = condition_counts(rows, args.sizes)
        state.update(updated_unix=time.time(), completed_conditions=counts["ok"],
                     attempted_conditions=counts["attempted"], partial_conditions=counts["partial"],
                     failed_conditions=counts["failed"], planned_conditions=len(backends)*sum(args.sizes)*6)
        for info, process in zip(state["workers"], processes):
            info["returncode"] = process.poll()
            path = out / "mast3r" / f"current_worker_{info['index']}.json"
            if path.exists():
                info["current"] = json.loads(path.read_text())
        atomic_json(out / "batch_status.json", state)
        subprocess.run([sys.executable, str(ROOT / "experiments/summarize.py"), str(out)], check=True)

    try:
        for index in range(args.workers):
            command = [sys.executable, "-u", str(ROOT / "experiments/run.py"), "--backend", "mast3r",
                       "--device", args.device, "--threads", str(args.threads), "--repeat-clean",
                       "--shard-index", str(index), "--shard-count", str(args.workers),
                       "--stop-file", str(stop_file), "--output", str(out), "--sizes", *map(str, args.sizes)]
            if args.optimizer_device:
                command += ["--optimizer-device", args.optimizer_device]
            if args.pair_cache_source:
                command += ["--pair-cache-source", args.pair_cache_source]
            logpath = out / f"worker_{index}.log"
            logfile = logpath.open("a")
            logs.append(logfile)
            process = subprocess.Popen(command, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT,
                                       env={**os.environ, "TQDM_DISABLE": "1"})
            processes.append(process)
            state["workers"].append({"index": index, "pid": process.pid, "log": logpath.name})
        refresh()
        while any(process.poll() is None for process in processes):
            time.sleep(15)
            refresh()
        if stop_file.exists():
            state["status"] = "interrupted"
        elif all(process.returncode == 0 for process in processes) and state["attempted_conditions"] == state["planned_conditions"]:
            state["status"] = "complete" if not state["partial_conditions"] and not state["failed_conditions"] else "complete_with_failures"
        else:
            state["status"] = "partial"
        refresh()
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for logfile in logs:
            logfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/office_fast")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--optimizer-device", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--sizes", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--pair-cache-source", help="Output root with compatible pretrained neural pairs")
    run(parser.parse_args())
