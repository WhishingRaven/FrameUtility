"""Approved fixed-protocol execution: VGGT once, then parallel GPU MASt3R."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from filelock import FileLock
from metrics import condition_counts
from run import atomic_json, load_official_ground_truth, resolve_device, sha256

ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="experiments/study_v1.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cohort", required=True, help="Results root containing executions/<scene>/ and runtime records")
    parser.add_argument("--device", required=True, choices=("mps", "cuda"),
                        help="Choose one GPU platform explicitly for the entire cohort")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def require_approval(protocol, *, require_implementation=True):
    if protocol.get("status") != "approved":
        raise RuntimeError("Experimental plan awaits user approval; no experiment was started")
    if require_implementation and not protocol.get("implementation_ready", False):
        raise RuntimeError("Approved procedure is not yet fully implemented and verified; no experiment was started")
    if protocol.get("approved_plan_sha256") != sha256(ROOT/protocol["plan"]):
        raise ValueError("Plan text differs from the approved document")


def validate_protocol(protocol, manifest, seed):
    if protocol.get("status") != "approved":
        raise RuntimeError("Experimental plan awaits user approval; no experiment was started")
    if not protocol.get("implementation_ready", False):
        raise RuntimeError("Approved procedure is not yet fully implemented and verified; no experiment was started")
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("Manifest does not belong to the approved protocol")
    if manifest["severities_deg"] != protocol["blur_degrees"]:
        raise ValueError("Blur ladder differs from the approved protocol")
    if len(protocol["seeds"]) != 1:
        raise ValueError("The per-scene result layout requires one configured seed")
    if seed not in protocol["seeds"]:
        raise ValueError("Seed is outside the approved single-run plan")
    if manifest["scene"] not in protocol["primary_scenes"]:
        raise ValueError("Scene is outside the approved cohort")
    for size in protocol["sizes"]:
        if str(size) not in manifest["sets"]:
            raise ValueError("Manifest is missing an approved context size")
        ids = manifest["sets"][str(size)]
        if len(ids) != size or len(set(ids)) != size:
            raise ValueError("Manifest subset size/order is invalid")
    if protocol["execution"]["workers"] < 1 or protocol["execution"]["threads_per_worker"] < 1:
        raise ValueError("Workers and threads must be positive")


def runtime_identifier(device):
    host = hashlib.sha256(platform.node().encode()).hexdigest()[:8]
    return f"{device}-{host}"


def inference_source_hashes():
    sources = sorted(p for p in (ROOT/"experiments").glob("*.py")
                     if not p.name.startswith(("analyze_", "test_")) and p.name != "summarize.py")
    sources.append(ROOT/"experiments/weights.json")
    return {str(p.relative_to(ROOT)): sha256(p) for p in sources}


def cohort_configuration(protocol_path, device, protocol):
    packages = dict(sorted((dist.metadata["Name"], dist.version)
                           for dist in importlib.metadata.distributions() if dist.metadata["Name"]))
    vendors = {}
    for name, path in {"vggt": "vggt", "mast3r": "mast3r",
                       "dust3r": "mast3r/dust3r", "croco": "mast3r/dust3r/croco"}.items():
        base = ROOT/"vendor"/path
        commit = subprocess.check_output(["git", "-C", str(base), "rev-parse", "HEAD"], text=True).strip()
        changes = subprocess.check_output(["git", "-C", str(base), "diff", "--binary", "HEAD"])
        vendors[name] = {"commit": commit, "tracked_diff_sha256": hashlib.sha256(changes).hexdigest()}
    return {"protocol_sha256": sha256(protocol_path), "plan_sha256": sha256(ROOT/protocol["plan"]),
            "device": device, "optimizer_device": device,
            "precision": "float32", "workers": protocol["execution"]["workers"],
            "threads_per_worker": protocol["execution"]["threads_per_worker"],
            "seeds": protocol["seeds"], "packages": packages,
            "vendor_versions": vendors,
            "mast3r_model_config_sha256": sha256(ROOT/"checkpoints/mast3r-config.json"),
            "platform": platform.platform(), "host": platform.node(), "python": platform.python_version(),
            "runtime_environment": {name: os.environ.get(name) for name in
                                    ("CUDA_VISIBLE_DEVICES", "PYTORCH_MPS_FAST_MATH",
                                     "PYTORCH_ENABLE_MPS_FALLBACK", "CUBLAS_WORKSPACE_CONFIG")},
            "source_sha256": inference_source_hashes()}


def worker_command(manifest_path, out, backend, device, seed, threads, sizes, stop_file, index, workers):
    return [sys.executable, "-u", str(ROOT/"experiments/run.py"), "--backend", backend,
            "--manifest", str(manifest_path), "--output", str(out), "--device", device,
            "--optimizer-device", device, "--threads", str(threads), "--seed", str(seed),
            "--shard-index", str(index), "--shard-count", str(workers),
            "--stop-file", str(stop_file), "--sizes", *map(str, sizes)]


def run(args):
    protocol_path, manifest_path = ROOT/args.protocol, ROOT/args.manifest
    protocol = json.loads(protocol_path.read_text())
    # Approval/readiness is checked before touching inputs, GPU or output folders.
    require_approval(protocol)
    manifest = json.loads(manifest_path.read_text())
    validate_protocol(protocol, manifest, args.seed)
    # Verify the official depth/mask snapshot before resolving a device or creating outputs.
    load_official_ground_truth(manifest_path, manifest)
    device = resolve_device(args.device)
    cohort = ROOT/args.cohort
    cohort.mkdir(parents=True, exist_ok=True)
    frozen = cohort_configuration(protocol_path, device, protocol)
    runtime_id = runtime_identifier(device)
    lock_path = cohort/"runtime_configurations.json"
    with FileLock(str(lock_path)+".lock"):
        configurations = json.loads(lock_path.read_text()) if lock_path.exists() else {}
        if runtime_id in configurations and configurations[runtime_id] != frozen:
            raise ValueError("Cohort hardware, code, packages or settings changed; use a new approved cohort")
        configurations[runtime_id] = frozen
        atomic_json(lock_path, configurations)
    with FileLock(str(cohort/f".active_{runtime_id}.lock"), timeout=0):
        out = cohort/"executions"/manifest["scene"]
        out.mkdir(parents=True, exist_ok=True)
        with FileLock(str(out/"controller.lock"), timeout=0):
            scene_configuration = out/"cohort_configuration.json"
            if scene_configuration.exists() and json.loads(scene_configuration.read_text()) != frozen:
                raise ValueError("Scene execution environment changed; results cannot be overwritten or mixed")
            atomic_json(scene_configuration, frozen)
            execute(protocol, protocol_path, manifest, manifest_path, out, device, args.seed, frozen)


def execute(protocol, protocol_path, manifest, manifest_path, out, device, seed, frozen):
    stop_file = out/"stop_requested"
    stop_file.unlink(missing_ok=True)
    sizes, backends = protocol["sizes"], protocol["backends"]
    threads, workers = protocol["execution"]["threads_per_worker"], protocol["execution"]["workers"]
    state = {"pid": os.getpid(), "status": "running", "started_unix": time.time(),
             "scene": manifest["scene"], "seed": seed, "device": device, "optimizer_device": device,
             "runtime_group": runtime_identifier(device), "host": frozen.get("host", platform.node()),
             "threads_per_worker": threads, "worker_count": workers, "backends": backends,
             "planned_conditions": len(backends)*sum(sizes)*len(manifest["severities_deg"]), "workers": []}
    processes, logs = [], []

    def request_stop(signum, frame):
        stop_file.touch()
        state["status"] = "stopping"

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}

    def refresh():
        rows = []
        for backend in backends:
            path = out/backend/"utilities.json"
            if path.exists():
                rows.extend(json.loads(path.read_text()))
        counts = condition_counts(rows, sizes)
        state.update(updated_unix=time.time(), completed_conditions=counts["ok"],
                     attempted_conditions=counts["attempted"], partial_conditions=counts["partial"],
                     failed_conditions=counts["failed"])
        for info, process in zip(state["workers"], processes):
            info["returncode"] = process.poll()
            path = out/info["backend"]/f"current_worker_{info['index']}.json"
            if path.exists():
                info["current"] = json.loads(path.read_text())
        atomic_json(out/"batch_status.json", state)

    try:
        for backend in backends:
            if stop_file.exists():
                break
            if cohort_configuration(protocol_path, device, protocol) != frozen:
                raise ValueError("Code/environment changed during the cohort")
            group = []
            count = workers if backend == "mast3r" else 1
            for index in range(count):
                command = worker_command(manifest_path, out, backend, device, seed, threads,
                                         sizes, stop_file, index, count)
                logfile = (out/f"{backend}_worker_{index}.log").open("a")
                logs.append(logfile)
                process = subprocess.Popen(command, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT,
                                           env={**os.environ, "TQDM_DISABLE": "1"})
                group.append(process)
                processes.append(process)
                state["workers"].append({"backend": backend, "index": index, "pid": process.pid,
                                         "log": Path(logfile.name).name})
            refresh()
            while any(p.poll() is None for p in group):
                time.sleep(15)
                refresh()
            refresh()
            if any(p.returncode != 0 for p in group):
                raise RuntimeError(f"{backend} worker exited unsuccessfully; later phases were not launched")
        if cohort_configuration(protocol_path, device, protocol) != frozen:
            raise ValueError("Code/environment changed during execution")
        if stop_file.exists():
            state["status"] = "interrupted"
        elif state["attempted_conditions"] == state["planned_conditions"]:
            state["status"] = "complete" if not state["partial_conditions"] and not state["failed_conditions"] else "complete_with_failures"
        else:
            state["status"] = "partial"
        refresh()
    except BaseException:
        state["status"] = "interrupted" if stop_file.exists() else "failed"
        refresh()
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for logfile in logs:
            logfile.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    run(build_parser().parse_args())
