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

from backends import ROOT, load_backend, project_depth_to_common_grid, reference_depth
from metrics import METRICS, evaluate_available, utility


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_official_ground_truth(manifest_path, manifest):
    """Load and verify the scene's fixed official depth/mask snapshot."""
    gt_manifest_path = manifest_path.parent / "official_ground_truth_manifest.json"
    if not gt_manifest_path.is_file():
        raise FileNotFoundError(f"Official ground-truth manifest is required: {gt_manifest_path}")
    gt_manifest = json.loads(gt_manifest_path.read_text())
    protocol = json.loads((ROOT / "experiments/study_v1.json").read_text())
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("Input manifest does not identify the approved study protocol")
    if gt_manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("Official ground truth belongs to a different protocol")
    if gt_manifest.get("scene") != manifest.get("scene"):
        raise ValueError("Official ground-truth scene does not match the input manifest")
    selection_path = manifest_path.parent.parent / "official_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Official selection snapshot is required: {selection_path}")
    selection = json.loads(selection_path.read_text())
    selected_scene = selection["scenes"].get(manifest["scene"], {})
    if selected_scene.get("status") != "pass":
        raise ValueError("Scene has no passing official ETH3D selection")
    if gt_manifest.get("official_selection_sha256") != sha256(selection_path):
        raise ValueError("Official GT was generated from a different selection snapshot")
    if gt_manifest.get("source_archive_sha256") != selected_scene.get("depth_archive_sha256"):
        raise ValueError("Official GT depth archive differs from the selection record")
    if gt_manifest.get("selected_sets") != selected_scene.get("sets"):
        raise ValueError("Official GT and selection snapshot have different view sets")
    for size in protocol["sizes"]:
        if manifest["sets"][str(size)] != gt_manifest["selected_sets"][str(size)]:
            raise ValueError(f"Input and official GT selected-view lists differ for size {size}")

    rows = {row["image_id"]: row for row in gt_manifest["images"]}
    selected_ids = set().union(*(set(manifest["sets"][str(size)]) for size in protocol["sizes"]))
    frame_ids = {frame["id"] for frame in manifest["frames"]}
    if frame_ids != selected_ids:
        raise ValueError("Input manifest frames must exactly match the official 15-view set")
    if len(rows) != len(gt_manifest["images"]) or rows.keys() != selected_ids:
        raise ValueError("Official GT records must exactly match the selected 15-view set")
    official_validity = {row["image_id"]: row for row in selected_scene["photo_validity"]}
    depths, masks, file_records = {}, {}, []
    grid_width, grid_height = gt_manifest["grid_size"]
    for image_id in selected_ids:
        row = rows[image_id]
        paths = {key: manifest_path.parent / row[key]
                 for key in ("depth_file", "mask_file", "metadata_file")}
        hashes = {"depth_file": "depth_sha256", "mask_file": "mask_sha256",
                  "metadata_file": "metadata_sha256"}
        for key, path in paths.items():
            if not path.resolve().is_relative_to(manifest_path.parent.resolve()):
                raise ValueError(f"Official GT path escapes its scene directory: {path}")
            if not path.is_file() or sha256(path) != row[hashes[key]]:
                raise ValueError(f"Official ground-truth input missing or changed: {path}")
        depth = np.load(paths["depth_file"], allow_pickle=False)
        mask = np.load(paths["mask_file"], allow_pickle=False)
        metadata = json.loads(paths["metadata_file"].read_text())
        if depth.shape != (grid_height, grid_width) or depth.dtype.kind != "f" or depth.dtype.itemsize != 4:
            raise ValueError(f"Invalid official depth array shape or dtype: {paths['depth_file']}")
        if mask.shape != depth.shape or mask.dtype != np.bool_:
            raise ValueError(f"Invalid official validity mask shape or dtype: {paths['mask_file']}")
        if not np.array_equal(np.isfinite(depth) & (depth > 0), mask):
            raise ValueError(f"Official depth and validity mask disagree: {image_id}")
        depth_hash = hashlib.sha256(np.ascontiguousarray(depth).tobytes()).hexdigest()
        mask_hash = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
        if depth_hash != row["depth_array_sha256"] or mask_hash != row["valid_mask_sha256"]:
            raise ValueError(f"Official depth or mask array hash mismatch: {image_id}")
        if int(mask.sum()) != row["valid_pixel_count"] or int(mask.sum()) < protocol["preparation"]["minimum_gt_pixels_per_image"]:
            raise ValueError(f"Official GT has an invalid pixel count: {image_id}")
        if (metadata.get("scene") != manifest["scene"] or metadata.get("image_id") != image_id
                or metadata.get("source_archive_sha256") != gt_manifest["source_archive_sha256"]
                or metadata.get("official_selection_sha256") != gt_manifest["official_selection_sha256"]):
            raise ValueError(f"Per-image official GT metadata disagrees with its manifest: {image_id}")
        selection_row = official_validity.get(image_id)
        if (not selection_row or selection_row["valid_gt_samples"] != row["valid_pixel_count"]
                or selection_row["valid_grid_mask_sha256"] != row["valid_mask_sha256"]):
            raise ValueError(f"Official GT mask differs from the selection record: {image_id}")
        depths[image_id], masks[image_id] = depth, mask
        file_records.append({"image_id": image_id, **{
            key: {"path": row[key], "sha256": row[hashes[key]]} for key in paths
        }})

    manifest["evaluation_ground_truth"] = "official_eth3d_occlusion_aware_depth"
    manifest["official_ground_truth"] = {
        "manifest": gt_manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(gt_manifest_path),
        "images": sorted(file_records, key=lambda row: row["image_id"]),
    }
    return depths, masks, gt_manifest, gt_manifest_path


def resolve_device(device):
    """Resolve once; accelerator requests never silently fall back to CPU."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("No GPU available; select --device cpu explicitly for a diagnostic run")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Metal GPU is unavailable in this process")
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError(f"Unknown device: {device}")
    return device


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
            "device": args.device, "precision": "float32", "seed": getattr(args, "seed", 2026),
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
            "evaluation": "official ETH3D depth, fixed common-grid mask, Sim(3) poses, one positive median depth scale per common context",
            "torch_threads": torch.get_num_threads()}


def run(args):
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text())
    frames = {f["id"]: f for f in manifest["frames"]}
    study_protocol = json.loads((ROOT / "experiments/study_v1.json").read_text())
    official_study = manifest.get("protocol_id") == study_protocol["protocol_id"]
    gt_masks = None
    gt_manifest_path = None
    if official_study:
        refs, gt_masks, gt_manifest, gt_manifest_path = load_official_ground_truth(manifest_path, manifest)
    else:
        refs = {}
    for f in frames.values():
        if not official_study and sha256(ROOT / f["depth"]) != f["depth_sha256"]:
            raise ValueError(f"Legacy geometry reference changed: {f['depth']}")
        for variant in f["variants"].values():
            if sha256(ROOT / variant["path"]) != variant["sha256"]:
                raise ValueError(f"Input image changed: {variant['path']}")
    args.device = resolve_device(args.device)
    seed = getattr(args, "seed", 2026)
    args.optimizer_device = getattr(args, "optimizer_device", None) or args.device
    args.shard_count = getattr(args, "shard_count", 1)
    args.shard_index = getattr(args, "shard_index", 0)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid target shard")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(getattr(args, "threads", None) or min(8, os.cpu_count() or 1))
    np.random.seed(seed)
    torch.manual_seed(seed)
    out = ROOT / args.output / args.backend
    predictions = out / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    fingerprint = environment(args, manifest_path)
    if official_study:
        fingerprint["official_ground_truth_manifest_sha256"] = sha256(gt_manifest_path)
        fingerprint["official_ground_truth_files"] = manifest["official_ground_truth"]["images"]
        fingerprint["study_protocol_sha256"] = sha256(ROOT / "experiments/study_v1.json")
        fingerprint["official_gt_preparation_code_sha256"] = sha256(ROOT / "experiments/prepare_official_gt.py")
        fingerprint["official_selection_code_sha256"] = sha256(ROOT / "experiments/select_official_scenes.py")
        fingerprint["evaluation_coordinate_map"] = {
            "source_image_size": [study_protocol["preparation"]["crop_width"],
                                  study_protocol["preparation"]["crop_height"]],
            "roi": study_protocol["preparation"]["evaluation_roi"],
            "grid": study_protocol["preparation"]["evaluation_grid"],
            "sampling": "nearest model-output pixel after exact backend resize/crop mapping",
        }
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
    if not official_study:
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
                result = {k: loaded[k] for k in loaded.files}
            if official_study and "depth_common_grid" not in result:
                raise ValueError("Cached prediction lacks official-grid depths; start with a fresh results directory")
            return result
        if getattr(args, "stop_file", None) and (ROOT / args.stop_file).exists():
            raise StopIteration
        if args.max_new_runs is not None and executions >= args.max_new_runs:
            raise StopIteration
        if model is None:
            print(f"Loading {args.backend} on {args.device}", flush=True)
            model = load_backend(args.backend, args.device, out / "mast3r_cache",
                                 optimizer_device=args.optimizer_device, pair_cache=pair_cache)
        paths = [str(ROOT / frames[i]["variants"][f"{severity if i == target else 0:g}"]["path"]) for i in ids]
        max_attempts = (study_protocol["execution"]["maximum_attempts_per_condition_before_diagnosis"]
                        if official_study else 1)
        last_error = None
        for attempt in range(1, max_attempts + 1):
            before = time.time()
            np.random.seed(seed)
            torch.manual_seed(seed)
            print(f"START {key}: {len(ids)} views (attempt {attempt}/{max_attempts})", flush=True)
            record = {"key": key, "ids": ids, "target": target, "severity_deg": severity,
                      "attempt": attempt, "max_attempts": max_attempts, "start_unix": before,
                      "status": "running", "pid": os.getpid(), "shard_index": args.shard_index}
            atomic_json(out / "current.json", record)
            atomic_json(out / f"current_worker_{args.shard_index}.json", record)
            executions += 1
            try:
                result = model.predict(ids, paths, key)
                if list(result["ids"]) != ids:
                    raise ValueError("Backend changed the requested image set/order")
                if official_study:
                    result["depth_common_grid"] = project_depth_to_common_grid(
                        result, args.backend,
                        [study_protocol["preparation"]["crop_width"],
                         study_protocol["preparation"]["crop_height"]],
                        study_protocol["preparation"]["evaluation_roi"],
                        study_protocol["preparation"]["evaluation_grid"],
                    )
                temp = path.with_suffix(".tmp.npz")
                np.savez_compressed(temp, **result)
                temp.replace(path)
                record["status"] = "ok"
                record["prediction_sha256"] = sha256(path)
                record["seconds"] = time.time() - before
                print(f"DONE {key}: {record['seconds']:.1f}s", flush=True)
                return result
            except Exception as error:
                last_error = error
                record.update({"status": "failed", "error": str(error),
                               "traceback": traceback.format_exc(), "seconds": time.time() - before})
                path.with_suffix(".tmp.npz").unlink(missing_ok=True)
                print(f"{'STOP' if attempt == max_attempts else 'RETRY'} {key}: {error}", flush=True)
            finally:
                record["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
                with runlog.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                atomic_json(out / "current.json", record)
                atomic_json(out / f"current_worker_{args.shard_index}.json", record)
                if args.device == "mps":
                    torch.mps.empty_cache()
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Condition {key} failed after {attempt} attempt(s); preserve the failure record, "
                    "diagnose and repair the cause before recomputing."
                ) from last_error

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
                pixels = sum(int(gt_masks[i].sum()) if official_study else
                             int(np.count_nonzero((refs[i] > 0) & np.isfinite(refs[i]))) for i in common)
                if clean is not None and repeated is not None:
                    first_metrics, first_errors = evaluate_available(
                        clean, common, poses, refs, gt_masks,
                        study_protocol["evaluation"]["minimum_positive_predictions_for_scale"])
                    repeat_metrics, repeat_errors = evaluate_available(
                        repeated, common, poses, refs, gt_masks,
                        study_protocol["evaluation"]["minimum_positive_predictions_for_scale"])
                    variance_row = {"size": size, "target": target,
                                     "absolute_metric_delta": {m: abs(first_metrics[m]-repeat_metrics[m]) for m in METRICS
                                                               if m in first_metrics and m in repeat_metrics},
                                     "errors": {"clean": first_errors, "repeat": repeat_errors}}
                    with FileLock(str(variance_path) + ".lock"):
                        variance = json.loads(variance_path.read_text()) if variance_path.exists() else []
                        variance = [r for r in variance if (r["size"], r["target"]) != (size, target)] + [variance_row]
                        atomic_json(variance_path, variance)
                drop = predict(common, 0, None, f"m{size}_drop_{target}")
                drop_metrics = evaluate_available(
                    drop, common, poses, refs, gt_masks,
                    study_protocol["evaluation"]["minimum_positive_predictions_for_scale"])
                for severity in manifest["severities_deg"]:
                    if all((size, target, severity, metric) in seen for metric in METRICS):
                        continue
                    add = clean if severity == 0 else predict(ids, severity, target, f"m{size}_add_{target}_b{severity:g}")
                    add_metrics = evaluate_available(
                        add, common, poses, refs, gt_masks,
                        study_protocol["evaluation"]["minimum_positive_predictions_for_scale"])
                    rows = utility_records(args.backend, manifest["scene"], size, target, severity,
                                           common, drop_metrics, add_metrics, pixels)
                    records = save_condition(records_path, rows)
    except StopIteration:
        print("Reached requested run limit; all completed results are saved for resume.", flush=True)
    print(f"Saved {sum(r['status']=='ok' for r in records)} metric rows; new reconstructions {executions}; elapsed {time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", required=True, choices=["vggt", "mast3r"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--optimizer-device", choices=["cpu", "mps", "cuda"], help="Default: same device as inference")
    parser.add_argument("--threads", type=int, help="PyTorch CPU threads per worker; default 8 or CPU count")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pair-cache-source", help="Reuse neural pairs from a verified compatible output root")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--stop-file", help="Finish the current reconstruction, then stop before new inference")
    parser.add_argument("--sizes", nargs="+", type=int, default=[5, 10, 15])
    parser.add_argument("--targets", type=int, help="Limit target count per size for a pilot; default all targets")
    parser.add_argument("--max-new-runs", type=int, help="Bound new reconstructions; completed predictions are resumable")
    parser.add_argument("--repeat-clean", action="store_true", help="Repeat clean inference to measure representative numerical variation")
    run(parser.parse_args())
