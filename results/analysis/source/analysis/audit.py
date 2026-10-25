"""Read-only result verification and CPU rescoring of saved predictions."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from metrics import METRICS, evaluate_available
from .core import PRIMARY, normalize_rows


class Sources:
    def __init__(self, root):
        self.root = Path(root)
        self.records = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        if path in self.records:
            digest = self.records[path]["sha256"]
        else:
            before = path.stat()
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Input changed while hashing: {path}")
            digest = h.hexdigest()
            self.records[path] = {"sha256": digest, "bytes": after.st_size, "mtime_ns": after.st_mtime_ns}
        if expected is not None and digest != expected:
            raise ValueError(f"Input SHA-256 differs from inference snapshot: {path}")
        return digest

    def json(self, path, expected=None):
        self.check(path, expected)
        return json.loads(Path(path).read_text())

    def assert_unchanged(self):
        for path, value in self.records.items():
            now = path.stat()
            if (now.st_size, now.st_mtime_ns) != (value["bytes"], value["mtime_ns"]):
                raise ValueError(f"Input changed during analysis: {path}")

    def manifest(self):
        def name(path):
            return str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)
        return {name(path): value for path, value in sorted(self.records.items())}


def expected_predictions(manifest, protocol):
    output = {}
    for size in protocol["sizes"]:
        ids = manifest["sets"][str(size)]
        output[f"m{size}_clean"] = {"size": size, "ids": ids, "role": "clean", "degree": 0, "targets": ids}
        for target in ids:
            common = [i for i in ids if i != target]
            output[f"m{size}_drop_{target}"] = {"size": size, "ids": common, "role": "drop", "degree": 0, "targets": [target]}
            for degree in protocol["blur_degrees"]:
                if degree:
                    output[f"m{size}_add_{target}_b{degree:g}"] = {"size": size, "ids": ids,
                        "role": "add", "degree": degree, "targets": [target]}
    return output


def diagnostic(prediction, common, masks):
    ids = list(prediction["ids"])
    arrays = [prediction["depth_common_grid"][ids.index(i)][masks[i]] for i in common]
    pixels = np.concatenate(arrays)
    finite = np.isfinite(pixels)
    nonpositive = int(np.count_nonzero(finite & (pixels <= 0)))
    nonfinite = int(np.count_nonzero(~finite))
    return {"evaluation_pixels": len(pixels), "nonpositive_pixels": nonpositive,
            "nonfinite_pixels": nonfinite, "nonpositive_fraction": nonpositive / len(pixels),
            "nonfinite_fraction": nonfinite / len(pixels),
            "strict_positive_depth_valid": nonpositive == nonfinite == 0}


def runtime_summary(runs):
    intervals = sorted((r["start_unix"], r["start_unix"] + r["seconds"]) for r in runs)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return {"attempts": len(runs), "successful_attempts": sum(r["status"] == "ok" for r in runs),
            "failed_attempts": sum(r["status"] != "ok" for r in runs),
            "retries": sum(r.get("attempt", 1) > 1 for r in runs),
            "summed_worker_inference_seconds": sum(r["seconds"] for r in runs),
            "inference_interval_union_seconds": sum(b - a for a, b in merged),
            "first_to_last_prediction_seconds": max(b for a, b in intervals) - intervals[0][0] if intervals else None,
            "first_prediction_unix": intervals[0][0] if intervals else None,
            "last_prediction_unix": max(b for a, b in intervals) if intervals else None,
            "max_single_worker_process_rss_bytes": max((r.get("process_peak_rss_bytes", 0) for r in runs), default=0)}


def same_number(a, b, description):
    if a is None or b is None:
        if a != b:
            raise ValueError(f"Stored/recomputed availability differs: {description}")
    elif not math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError(f"Stored/recomputed score differs: {description}: {a} vs {b}")


def load_scene(root, results_root, scene, protocol, study, sources):
    folder = results_root / "executions" / scene
    manifest = sources.json(folder / "inputs.json")
    batch = sources.json(folder / "batch_status.json")
    frozen = sources.json(folder / "cohort_configuration.json")
    if manifest.get("scene") != scene or manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError(f"Scene/protocol identity differs: {scene}")
    if manifest["severities_deg"] != protocol["blur_degrees"] or batch["seed"] not in protocol["seeds"]:
        raise ValueError(f"Blur ladder/seed differs: {scene}")
    for key, expected in study.items():
        if frozen.get(key) != expected:
            raise ValueError(f"Scientific fingerprint differs: {scene}/{key}")
    if batch["device"] != frozen["device"] or batch["optimizer_device"] != frozen["optimizer_device"]:
        raise ValueError(f"Scene device differs between execution records: {scene}")
    assignments = sources.json(results_root / "scene_assignments.json")
    if assignments.get(scene) != batch["runtime_group"]:
        raise ValueError(f"Scene runtime assignment differs: {scene}")
    runtime = sources.json(results_root / "runtime_configurations.json")
    if runtime.get(batch["runtime_group"]) != frozen:
        raise ValueError(f"Runtime registry differs from scene fingerprint: {scene}")
    frames = {f["id"]: f for f in manifest["frames"]}
    if len(frames) != len(manifest["frames"]):
        raise ValueError(f"Duplicate image IDs in snapshot: {scene}")
    for size in protocol["sizes"]:
        ids = manifest["sets"][str(size)]
        if len(ids) != size or len(set(ids)) != size or not set(ids) <= frames.keys():
            raise ValueError(f"Invalid selected set: {scene}/{size}")
    sizes = sorted(protocol["sizes"])
    if any(not set(manifest["sets"][str(a)]) <= set(manifest["sets"][str(b)]) for a, b in zip(sizes, sizes[1:])):
        raise ValueError(f"Non-nested selected sets: {scene}")
    selection = sources.json(root / protocol["selection"]["result_file"], protocol["selection"]["result_sha256"])
    selected = selection["scenes"][scene]
    if selected["status"] != "pass" or selected["sets"] != manifest["sets"]:
        raise ValueError(f"Official selection differs: {scene}")
    gt_path = root / manifest["official_ground_truth_manifest"]
    gt_manifest = sources.json(gt_path, manifest["official_ground_truth_manifest_sha256"])
    if gt_manifest["selected_sets"] != manifest["sets"] or gt_manifest["protocol_id"] != protocol["protocol_id"]:
        raise ValueError(f"Official GT selected sets differ: {scene}")
    original_manifest_path = gt_path.parent / "manifest.json"
    if original_manifest_path.exists():
        prepared = sources.json(original_manifest_path)
        original_fields = {k: v for k, v in manifest.items() if k not in ("evaluation_ground_truth", "official_ground_truth")}
        if prepared != original_fields:
            raise ValueError(f"Prepared manifest differs from saved inference inputs: {scene}")
    depths, masks = {}, {}
    photo_validity = {r["image_id"]: r for r in selected["photo_validity"]}
    shape = tuple(reversed(protocol["preparation"]["evaluation_grid"]))
    for image_id, frame in frames.items():
        for p, h in (("depth", "depth_sha256"), ("official_valid_mask", "official_valid_mask_sha256"),
                     ("official_gt_metadata", "official_gt_metadata_sha256")):
            sources.check(root / frame[p], frame[h])
        depths[image_id] = np.load(root / frame["depth"], allow_pickle=False)
        masks[image_id] = np.load(root / frame["official_valid_mask"], allow_pickle=False)
        metadata = sources.json(root / frame["official_gt_metadata"])
        if metadata["scene"] != scene or metadata["image_id"] != image_id or metadata["protocol_id"] != protocol["protocol_id"]:
            raise ValueError(f"GT metadata identity differs: {scene}/{image_id}")
        gt, mask = depths[image_id], masks[image_id]
        if mask.dtype != bool or gt.shape != mask.shape or gt.shape != shape:
            raise ValueError(f"GT shape/mask dtype differs: {scene}/{image_id}")
        if not np.all(np.isfinite(gt[mask]) & (gt[mask] > 0)) or int(mask.sum()) != frame["official_gt_valid_pixels"]:
            raise ValueError(f"Invalid GT mask/depth values: {scene}/{image_id}")
        mask_digest = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
        if (mask_digest != metadata["valid_mask_sha256"]
                or mask_digest != photo_validity[image_id]["valid_grid_mask_sha256"]
                or int(mask.sum()) != photo_validity[image_id]["valid_gt_samples"]):
            raise ValueError(f"GT array mask hash differs: {scene}/{image_id}")
        if hashlib.sha256(np.ascontiguousarray(gt).tobytes()).hexdigest() != metadata["depth_array_sha256"]:
            raise ValueError(f"GT array depth hash differs: {scene}/{image_id}")
        for variant in frame["variants"].values():
            sources.check(root / variant["path"], variant["sha256"])
    poses = {i: f["c2w"] for i, f in frames.items()}
    expected = expected_predictions(manifest, protocol)
    normalized, diagnostics, audits, costs = [], [], [], []
    environments = {}
    weights = sources.json(root / "experiments/weights.json")
    for backend in protocol["backends"]:
        backend_folder = folder / backend
        utility_path = backend_folder / "utilities.json"
        raw = sources.json(utility_path) if utility_path.exists() else []
        rows = normalize_rows(raw, scene, backend, manifest, protocol)
        normalized.extend(rows)
        if not (backend_folder / "environment.json").exists():
            if raw or batch["status"] == "complete":
                raise ValueError(f"Missing environment for scored/complete backend: {scene}/{backend}")
            continue
        env = sources.json(backend_folder / "environment.json")
        environments[backend] = env
        if original_manifest_path.exists():
            sources.check(original_manifest_path, env["manifest_sha256"])
        if env["device"] != frozen["device"] or env["seed"] != batch["seed"] or env["backend"] != backend:
            raise ValueError(f"Worker environment differs: {scene}/{backend}")
        if (env["precision"] != frozen["precision"] or env["weight_sha256"] != weights[backend]["sha256"]
                or env["weight_revision"] != weights[backend]["revision"]):
            raise ValueError(f"Worker precision/weight identity differs: {scene}/{backend}")
        if any(env["source_commits"][name] != value["commit"] for name, value in study["vendor_versions"].items()):
            raise ValueError(f"Worker vendor source commits differ: {scene}/{backend}")
        if env["study_protocol_sha256"] != study["protocol_sha256"]:
            raise ValueError(f"Worker protocol fingerprint differs: {scene}/{backend}")
        for key, digest in env["code_sha256"].items():
            if study["source_sha256"].get("experiments/" + key) != digest:
                raise ValueError(f"Worker/scientific source fingerprints differ: {scene}/{backend}/{key}")
        if env["official_ground_truth_manifest_sha256"] != manifest["official_ground_truth_manifest_sha256"]:
            raise ValueError(f"Worker GT fingerprint differs: {scene}/{backend}")
        if env["official_ground_truth_files"] != manifest["official_ground_truth"]["images"]:
            raise ValueError(f"Worker GT file/mask records differ: {scene}/{backend}")
        log_path = backend_folder / "runs.jsonl"
        sources.check(log_path)
        runs = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        successes = defaultdict(list)
        for record in runs:
            if record["key"] not in expected:
                raise ValueError(f"Unexpected prediction attempt: {scene}/{backend}/{record['key']}")
            if record["status"] == "ok":
                successes[record["key"]].append(record)
        scores, depth_diags, count = {}, {}, 0
        pred_paths = {p.stem: p for p in (backend_folder / "predictions").glob("*.npz")}
        if pred_paths.keys() - expected.keys():
            raise ValueError(f"Unexpected prediction files: {scene}/{backend}")
        for key, info in expected.items():
            if key not in pred_paths:
                continue
            accepted = successes.get(key, [])
            if not accepted:
                raise ValueError(f"Prediction lacks successful provenance: {scene}/{backend}/{key}")
            sources.check(pred_paths[key], accepted[0]["prediction_sha256"])
            if any(r["prediction_sha256"] != accepted[0]["prediction_sha256"] for r in accepted[1:]):
                raise ValueError(f"Multiple different successful predictions for one condition: {key}")
            with np.load(pred_paths[key], allow_pickle=False) as data:
                prediction = {name: data[name] for name in ("ids", "c2w", "depth_common_grid")}
            if list(prediction["ids"]) != info["ids"] or accepted[0]["ids"] != info["ids"]:
                raise ValueError(f"Prediction image order/context differs: {scene}/{backend}/{key}")
            if prediction["depth_common_grid"].shape != (len(info["ids"]), *shape):
                raise ValueError(f"Prediction evaluation grid differs: {key}")
            if prediction["c2w"].shape != (len(info["ids"]), 4, 4):
                raise ValueError(f"Prediction pose dimensions differ: {key}")
            for target in info["targets"]:
                common = [i for i in manifest["sets"][str(info["size"]) ] if i != target]
                score = evaluate_available(prediction, common, poses, depths, masks,
                                          protocol["evaluation"]["minimum_positive_predictions_for_scale"])
                role = "add" if info["role"] == "clean" else info["role"]
                score_key = (info["size"], target, role, info["degree"])
                scores[score_key] = score
                diag = diagnostic(prediction, common, masks)
                diag["strict_absrel_diagnostic"] = score[0].get(PRIMARY) if diag["strict_positive_depth_valid"] else None
                depth_diags[score_key] = diag
            count += 1
        for row in rows:
            size, target, degree = row["size"], row["target"], row["blur_deg"]
            common = [i for i in manifest["sets"][str(size)] if i != target]
            pixels = sum(int(masks[i].sum()) for i in common)
            if row["evaluation_pixels"] is not None and row["evaluation_pixels"] != pixels:
                raise ValueError(f"GT pixel count differs: {scene}/{backend}/{size}/{target}")
            if row["status"] == "missing":
                continue
            recomputed = []
            for role, blur in (("drop", 0), ("add", degree)):
                score = scores.get((size, target, role, blur))
                if score is None:
                    raise ValueError(f"Recorded score lacks its source prediction: {scene}/{backend}/{size}/{target}/{degree}")
                recomputed.append(score[0].get(row["metric"]))
            for value, field in zip(recomputed, ("E_drop", "E_add")):
                same_number(row.get(field), value, f"{scene}/{backend}/{size}/{target}/{degree}/{row['metric']}/{field}")
            if (row["status"] == "ok") != all(v is not None for v in recomputed):
                raise ValueError(f"Stored/recomputed score validity differs: {row}")
            if row["metric"] == PRIMARY:
                record = {k: row[k] for k in ("scene", "backend", "size", "target", "blur_deg", "status")}
                for role, blur in (("drop", 0), ("add", degree)):
                    for name, value in depth_diags[(size, target, role, blur)].items():
                        record[role + "_" + name] = value
                diagnostics.append(record)
        if batch["status"] == "complete" and (count != len(expected) or any(r["status"] != "ok" for r in rows)):
            raise ValueError(f"Complete batch has missing/invalid predictions or scores: {scene}/{backend}")
        audits.append({"scene": scene, "backend": backend, "expected_predictions": len(expected),
                       "verified_prediction_hashes": count, "stored_metric_rows": len(raw),
                       "rescored_metric_rows": sum(r["status"] != "missing" for r in rows),
                       "all_stored_scores_match_cpu_rescoring": True})
        costs.append({"scene": scene, "backend": backend, **runtime_summary(runs)})
    if len(environments) == len(protocol["backends"]):
        values = list(environments.values())
        for key in ("manifest_sha256", "official_ground_truth_manifest_sha256", "official_ground_truth_files", "seed", "device", "precision"):
            if any(v[key] != values[0][key] for v in values[1:]):
                raise ValueError(f"Two backends used different inputs/GT/device: {scene}/{key}")
    info = {"scene": scene, "device": batch["device"], "runtime_group": batch["runtime_group"],
            "seed": batch["seed"], "batch_status": batch["status"], "host": batch["host"],
            "complete": batch["status"] == "complete" and all(r["status"] == "ok" for r in normalized)}
    return {"info": info, "manifest": manifest, "selection": selected, "rows": normalized,
            "diagnostics": diagnostics, "audit": audits, "costs": costs, "gt_depths": depths, "gt_masks": masks}
