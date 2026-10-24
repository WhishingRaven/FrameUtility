"""Run both full sweeps sequentially and refresh reviewable results during execution."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    child = None
    state = {"pid": os.getpid(), "device": args.device, "started_unix": time.time(),
             "status": "running", "phases": [], "requested_sizes": args.sizes,
             "planned_conditions": 2*sum(args.sizes)*6}

    def save():
        state["updated_unix"] = time.time()
        state["completed_conditions"] = 0
        for backend in ("vggt", "mast3r"):
            path = out / backend / "utilities.json"
            if path.exists():
                rows = json.loads(path.read_text())
                state["completed_conditions"] += len({(r["size"], r["target"], r["blur_deg"])
                                                      for r in rows if r["status"] == "ok" and r["size"] in args.sizes})
        temp = out / "batch_status.tmp.json"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(out / "batch_status.json")

    def refresh():
        subprocess.run([sys.executable, str(ROOT / "experiments/summarize.py"), str(out)], check=True)
        save()

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        for backend in ("vggt", "mast3r"):
            for size in args.sizes:
                logpath = out / backend / f"sweep_m{size}.log"
                logpath.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(ROOT / "experiments/run.py"), "--backend", backend,
                           "--device", args.device, "--sizes", str(size), "--repeat-clean", "--output", str(out)]
                print(f"Starting {backend}, {size} views", flush=True)
                with logpath.open("a") as logfile:
                    child = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT,
                                             env={**os.environ, "TQDM_DISABLE": "1"}, cwd=ROOT)
                    state["current"] = {"backend": backend, "size": size, "pid": child.pid,
                                        "log": str(logpath.relative_to(out))}
                    save()
                    while True:
                        try:
                            returncode = child.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            refresh()
                state["phases"].append({"backend": backend, "size": size, "returncode": returncode})
                child = None
                refresh()
        state.pop("current", None)
        state["status"] = "complete" if state["completed_conditions"] == 2*sum(args.sizes)*6 else "partial"
        save()
    except BaseException:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        state["status"] = "interrupted"
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--output", default="results/office_sweep")
    parser.add_argument("--sizes", nargs="+", type=int, default=[5, 10, 20])
    run(parser.parse_args())
