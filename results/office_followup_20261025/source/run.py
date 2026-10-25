"""Run a resumable, fixed-set leave-one-out sweep. Failures remain explicit outcomes."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

# Upstream MASt3R caches contain numpy objects generated locally by this experiment.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import numpy as np
import torch
from filelock import FileLock

from backends import ROOT, load_backend, reference_depth
from metrics import METRICS, evaluate_available, utility


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def utility_records(backend, scene, size, target, severity, common, drop, add, pixels):
    drop_values, drop_errors = drop
    add_values, add_errors = add
    rows = []
    for metric in METRICS:
        family = "pose" if metric.startswith("pose_") else "geometry"
        row = {"backend": backend, "scene": scene, "size": size, "n_context": len(common),
               "target": target, "blur_deg": severity, "metric": metric,
               "E_drop": drop_values.get(metric), "E_add": add_values.get(metric),
               "evaluation_pixels": pixels}
        errors = [f"{name} {family}: {error}" for name, error in
                  [("DROP", drop_errors.get(family)), ("ADD", add_errors.get(family))] if error]
        if errors:
            row.update(status="failed", error="; ".join(errors))
        else:
            raw, relative = utility(drop_values[metric], add_values[metric])
            row.update(status="ok", utility_raw=raw, utility_relative=relative,
                       utility=relative if family == "pose" else raw)
        rows.append(row)
    return rows


def replace_condition(records, rows):
    key = (rows[0]["size"], rows[0]["target"], rows[0]["blur_deg"])
    return [r for r in records if (r["size"], r["target"], r["blur_deg"]) != key] + rows


def save_condition(path, rows):
    with FileLock(str(path) + ".lock"):
        records = json.loads(path.read_text()) if path.exists() else []
        records = replace_condition(records, rows)
        atomic_json(path, records)
    return records


def pair_cache_signature(fingerprint):
    """Only neural inference is reusable across different alignment devices."""
    return {key: fingerprint[key] for key in
            ("manifest_sha256", "weight_sha256", "model_config_sha256", "source_commits", "packages",
             "device", "precision", "seed", "float32_matmul_precision", "cudnn_allow_tf32", "torch_threads")}


def environment(args, manifest_path):
    repos = {"vggt": "vggt", "mast3r": "mast3r", "dust3r": "mast3r/dust3r", "croco": "mast3r/dust3r/croco"}
    commits = {name: subprocess.check_output(["git", "-C", str(ROOT / "vendor" / path), "rev-parse", "HEAD"], text=True).strip()
               for name, path in repos.items()}
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "numpy", "scipy", "pillow", "opencv-python", "huggingface-hub")}
    weights = json.loads((ROOT / "experiments/weights.json").read_text())[args.backend]
    model_path = ROOT / weights["file"]
    if model_path.stat().st_size != weights["bytes"]:
        raise ValueError(f"Incomplete model download: {model_path}")
    weight_sha = sha256(model_path)
    if weight_sha != weights["sha256"]:
        raise ValueError(f"Model SHA-256 differs from the official pinned checkpoint: {model_path}")
    return {"manifest_sha256": sha256(manifest_path), "backend": args.backend,
            "device": args.device, "precision": "float32", "seed": 2026,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "source_commits": commits, "weight_sha256": weight_sha, "weight_revision": weights["revision"],
            "model_config_sha256": sha256(ROOT / "checkpoints/mast3r-config.json") if args.backend == "mast3r" else None,
            "packages": packages,
            "code_sha256": {name: sha256(Path(__file__).parent / name) for name in
                            ("prepare.py", "backends.py", "metrics.py", "run.py")},
            "platform": platform.platform(), "python": platform.python_version(),
            "mast3r": {"graph": "complete symmetrized", "optimizer_device": getattr(args, "optimizer_device", None) or args.device,
                       "niter1": 300, "niter2": 300},
            "evaluation": "common context only; Sim(3) poses; one median depth scale per common context",
            "torch_threads": torch.get_num_threads()}


def run(args):
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text())
    frames = {f["id"]: f for f in manifest["frames"]}
    for f in frames.values():
        if sha256(ROOT / f["depth"]) != f["depth_sha256"]:
            raise ValueError(f"Geometry reference changed: {f['depth']}")
        for variant in f["variants"].values():
            if sha256(ROOT / variant["path"]) != variant["sha256"]:
                raise ValueError(f"Input image changed: {variant['path']}")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process; check the PyTorch build and GPU driver")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Metal GPU is unavailable in this process; check GPU access or select --device cpu")
    args.optimizer_device = getattr(args, "optimizer_device", None) or args.device
    args.shard_count = getattr(args, "shard_count", 1)
    args.shard_index = getattr(args, "shard_index", 0)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid target shard")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(getattr(args, "threads", None) or min(8, os.cpu_count() or 1))
    np.random.seed(2026)
    torch.manual_seed(2026)
    out = ROOT / args.output / args.backend
    predictions = out / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    fingerprint = environment(args, manifest_path)
    pair_cache = None
    if args.backend == "mast3r" and getattr(args, "pair_cache_source", None):
        source = ROOT / args.pair_cache_source / "mast3r"
        source_environment = json.loads((source / "environment.json").read_text())
        if pair_cache_signature(source_environment) != pair_cache_signature(fingerprint):
            raise ValueError("Shared pair-cache inference settings or inputs differ")
        pair_cache = source / "mast3r_cache/pairs"
        atomic_json(out / "pair_cache_source.json", {"path": str(source), "environment": source_environment})
    inputs_path = out.parent / "inputs.json"
    if inputs_path.exists() and json.loads(inputs_path.read_text()) != manifest:
        raise ValueError("The two backends must use identical frozen inputs; use a new --output directory")
    atomic_json(inputs_path, manifest)
    envpath = out / "environment.json"
    with FileLock(str(envpath) + ".lock"):
        if envpath.exists() and json.loads(envpath.read_text()) != fingerprint:
            raise ValueError("Environment/code/inputs changed; use a new --output directory")
        atomic_json(envpath, fingerprint)
    refs = {i: reference_depth(ROOT / f["depth"], args.backend) for i, f in frames.items()}
    poses = {i: f["c2w"] for i, f in frames.items()}
    model = None
    executions = 0
    started = time.time()
    runlog = out / "runs.jsonl"

    def predict(ids, severity, target, key):
        path = predictions / f"{key}.npz"
        with FileLock(str(path) + ".lock"):
            return compute_prediction(ids, severity, target, key, path)

    def compute_prediction(ids, severity, target, key, path):
        nonlocal model, executions
        if path.exists():
            with np.load(path, allow_pickle=False) as loaded:
                return {k: loaded[k] for k in loaded.files}
        if getattr(args, "stop_file", None) and (ROOT / args.stop_file).exists():
            raise StopIteration
        if args.max_new_runs is not None and executions >= args.max_new_runs:
            raise StopIteration
        if model is None:
            print(f"Loading {args.backend} on {args.device}", flush=True)
            model = load_backend(args.backend, args.device, out / "mast3r_cache",
                                 optimizer_device=args.optimizer_device, pair_cache=pair_cache)
        np.random.seed(2026)
        torch.manual_seed(2026)
        paths = [str(ROOT / frames[i]["variants"][f"{severity if i == target else 0:g}"]["path"]) for i in ids]
        before = time.time()
        print(f"START {key}: {len(ids)} views", flush=True)
        record = {"key": key, "ids": ids, "target": target, "severity_deg": severity,
                  "start_unix": before, "status": "running", "pid": os.getpid(), "shard_index": args.shard_index}
        atomic_json(out / "current.json", record)
        atomic_json(out / f"current_worker_{args.shard_index}.json", record)
        executions += 1
        try:
            result = model.predict(ids, paths, key)
            if list(result["ids"]) != ids:
                raise ValueError("Backend changed the requested image set/order")
            temp = path.with_suffix(".tmp.npz")
            np.savez_compressed(temp, **result)
            temp.replace(path)
            record["status"] = "ok"
            print(f"DONE {key}: {time.time()-before:.1f}s", flush=True)
            return result
        except Exception as e:
            record.update({"status": "failed", "error": str(e), "traceback": traceback.format_exc()})
            print(f"FAILED {key}: {e}", flush=True)
            if isinstance(e, MemoryError) or "out of memory" in str(e).lower():
                raise
            return None
        finally:
            record["seconds"] = time.time()-before
            record["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
            with runlog.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            atomic_json(out / "current.json", record)
            atomic_json(out / f"current_worker_{args.shard_index}.json", record)
            if args.device == "mps":
                torch.mps.empty_cache()

    records_path = out / "utilities.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    seen = {(r["size"], r["target"], r["blur_deg"], r["metric"]) for r in records if r["status"] == "ok"}
    variance_path = out / "repeatability.json"
    variance = json.loads(variance_path.read_text()) if variance_path.exists() else []
    try:
        for size in args.sizes:
            ids = manifest["sets"][str(size)]
            clean = predict(ids, 0, None, f"m{size}_clean")
            repeated = predict(ids, 0, None, f"m{size}_clean_repeat") if args.repeat_clean else None
            targets = ids[:args.targets] if args.targets else ids
            # Use the global frame index so the same blurred target belongs to one
            # worker across every context size; shared pair-cache writes stay disjoint.
            targets = [i for i in targets if list(frames).index(i) % args.shard_count == args.shard_index]
            for target in targets:
                common = [i for i in ids if i != target]
                pixels = sum(int(np.count_nonzero((refs[i] > 0) & np.isfinite(refs[i]))) for i in common)
                if clean is not None and repeated is not None:
                    first_metrics, first_errors = evaluate_available(clean, common, poses, refs)
                    repeat_metrics, repeat_errors = evaluate_available(repeated, common, poses, refs)
                    variance_row = {"size": size, "target": target,
                                     "absolute_metric_delta": {m: abs(first_metrics[m]-repeat_metrics[m]) for m in METRICS
                                                               if m in first_metrics and m in repeat_metrics},
                                     "errors": {"clean": first_errors, "repeat": repeat_errors}}
                    with FileLock(str(variance_path) + ".lock"):
                        variance = json.loads(variance_path.read_text()) if variance_path.exists() else []
                        variance = [r for r in variance if (r["size"], r["target"]) != (size, target)] + [variance_row]
                        atomic_json(variance_path, variance)
                drop = predict(common, 0, None, f"m{size}_drop_{target}")
                drop_metrics = evaluate_available(drop, common, poses, refs)
                for severity in manifest["severities_deg"]:
                    if all((size, target, severity, metric) in seen for metric in METRICS):
                        continue
                    add = clean if severity == 0 else predict(ids, severity, target, f"m{size}_add_{target}_b{severity:g}")
                    add_metrics = evaluate_available(add, common, poses, refs)
                    rows = utility_records(args.backend, manifest["scene"], size, target, severity,
                                           common, drop_metrics, add_metrics, pixels)
                    records = save_condition(records_path, rows)
    except StopIteration:
        print("Reached requested run limit; all completed results are saved for resume.", flush=True)
    print(f"Saved {sum(r['status']=='ok' for r in records)} metric rows; new reconstructions {executions}; elapsed {time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/experiments/office/manifest.json")
    parser.add_argument("--output", default="results/office_sweep")
    parser.add_argument("--backend", required=True, choices=["vggt", "mast3r"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--optimizer-device", choices=["cpu", "mps", "cuda"], help="Default: same device as inference")
    parser.add_argument("--threads", type=int, help="PyTorch CPU threads per worker; default 8 or CPU count")
    parser.add_argument("--pair-cache-source", help="Reuse neural pairs from a verified compatible output root")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--stop-file", help="Finish the current reconstruction, then stop before new inference")
    parser.add_argument("--sizes", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--targets", type=int, help="Limit target count per size for a pilot; default all targets")
    parser.add_argument("--max-new-runs", type=int, help="Bound new reconstructions; completed predictions are resumable")
    parser.add_argument("--repeat-clean", action="store_true", help="Repeat clean inference to measure representative numerical variation")
    run(parser.parse_args())
