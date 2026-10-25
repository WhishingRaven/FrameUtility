"""Freeze, prepare and execute the bounded local repeatability follow-up."""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))

import numpy as np
from PIL import Image
import torch

from backends import ROOT, load_backend, reference_depth
from followup_metrics import POLICY, evaluate_followup, fixed_mask_geometry
from prepare import exposure, read_calibration
from run import atomic_json, environment, sha256

DEFAULT_OUT = "results/office_followup_20261025"
PILOT = ROOT / "results/office_fast"
DENSE = [0, .025, .05, .075, .1, .15, .2, .25]
CASES = [
    {"backend": "vggt", "size": 5, "target": "DSC_0250", "degrees": DENSE,
     "selection": "Pilot depth transition with clean effect above the fixed floor; 5-view context"},
    {"backend": "vggt", "size": 10, "target": "DSC_0242", "degrees": DENSE,
     "selection": "Pilot depth transition with clean effect above the fixed floor; 10-view context"},
    {"backend": "vggt", "size": 20, "target": "DSC_0252", "degrees": DENSE,
     "selection": "Smallest pilot-positive VGGT clean depth effect; stress test of tiny sign changes"},
    {"backend": "mast3r", "size": 5, "target": "DSC_0251", "degrees": [0, .5, 1],
     "selection": "Small pilot clean depth benefit and first crossing in [0.5,1]"},
    {"backend": "mast3r", "size": 20, "target": "DSC_0238", "degrees": [0, 2, 4],
     "selection": "Previously illustrated strong five-metric transition in [2,4]"},
]
CODE = ("followup.py", "followup_metrics.py", "prepare.py", "backends.py", "metrics.py", "run.py")


def case_id(case):
    return f"m{case['size']}_{case['target']}"


def read_json(path):
    return json.loads(path.read_text())


def load_prediction(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def frozen_protocol(out):
    protocol = read_json(out / "protocol.json")
    if protocol["policy"] != POLICY:
        raise ValueError("Frozen evaluation policy changed; use a new output root")
    for name, expected in protocol["code_sha256"].items():
        if sha256(ROOT / "experiments" / name) != expected:
            raise ValueError(f"Frozen source changed: {name}; use a new output root")
    for name, expected in protocol["pilot_sources_sha256"].items():
        if sha256(PILOT / name) != expected:
            raise ValueError(f"Pilot source changed: {name}")
    return protocol


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        frozen_protocol(out)
        print("Existing frozen protocol retained", flush=True)
        return
    manifest = read_json(PILOT / "inputs.json")
    scene = ROOT / "data/prepared/eth3d/office"
    K, (ow, oh), _ = read_calibration(scene)
    sw, sh = 1120, round(oh * 1120 / ow)
    Ksource = np.diag([sw / ow, sh / oh, 1.]) @ K
    left, top, w, h = manifest["blur"]["fixed_overscan_crop"]
    images = out / "images"
    images.mkdir(exist_ok=True)
    controls, variants = [], {}
    needed = {c["target"] for c in CASES if c["backend"] == "vggt"}
    for frame in manifest["frames"]:
        source = ROOT / frame["source"]
        if sha256(source) != frame["source_sha256"]:
            raise ValueError(f"Source image changed: {source}")
        rgb = np.asarray(Image.open(source).convert("RGB").resize((sw, sh), Image.Resampling.LANCZOS))
        control, support = exposure(rgb, Ksource, 0, force_pipeline=True)
        control = control[top:top+h, left:left+w]
        clean_path = ROOT / frame["variants"]["0"]["path"]
        clean = np.asarray(Image.open(clean_path))
        path = images / f"{frame['id']}_zero_pipeline.png"
        Image.fromarray(control).save(path)
        controls.append({"frame": frame["id"], "path": str(path.relative_to(ROOT)),
                         "sha256": sha256(path), "clean_sha256": sha256(clean_path),
                         "changed_pixels": int(np.count_nonzero(np.any(control != clean, axis=2))),
                         "maximum_channel_difference": int(np.abs(control.astype(int)-clean).max()),
                         "minimum_support": float(support[top:top+h, left:left+w].min())})
        if not np.array_equal(control, clean):
            raise ValueError("Zero-motion pipeline is not byte-identical; revise the protocol before inference")
        if frame["id"] in needed:
            variants[frame["id"]] = {}
            for degree in DENSE:
                if degree == 0:
                    variant_path = clean_path
                else:
                    blurred, mask = exposure(rgb, Ksource, degree)
                    if mask[top:top+h, left:left+w].min() < .999:
                        raise ValueError("Invalid synthetic border in dense ladder")
                    pixels = blurred[top:top+h, left:left+w]
                    variant_path = images / f"{frame['id']}_yaw_{degree:g}.png"
                    Image.fromarray(pixels).save(variant_path)
                    if degree == .25:
                        original = np.asarray(Image.open(ROOT / frame["variants"]["0.25"]["path"]))
                        if not np.array_equal(original, pixels):
                            raise ValueError("Regenerated 0.25-degree anchor differs from pilot")
                        variant_path = ROOT / frame["variants"]["0.25"]["path"]
                variants[frame["id"]][f"{degree:g}"] = {"path": str(variant_path.relative_to(ROOT)),
                                                          "sha256": sha256(variant_path)}
        print("PREPARED", frame["id"], "zero control identical", flush=True)
    protocol = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Pilot-selected control, dense first-crossing and numerical-repeatability follow-up; one scene, no population inference",
        "policy": POLICY, "cases": CASES,
        "selection_note": "All five cases chosen from the exploratory pilot before new inference; not random or independent scene samples",
        "dense_degrees": DENSE, "new_reconstructions": 105,
        "zero_control": "All 20 images traverse 33 identity homographies, linear-radiance integration, transfer/quantization and the same crop; decoded pixels must equal clean exactly. Identical controls alias clean for inference.",
        "repeat_design": "Three fresh same-seed reconstructions per condition. MASt3R neural pair caches are fresh and separate for each replicate block; only pairs within a block are reused. All alignment is recomputed. Pilot predictions are never counted as new repeats.",
        "execution": "VGGT one process; MASt3R up to three simultaneous replicate processes; 8 CPU threads/process; float32; native model preprocessing; 300+300 MASt3R alignment iterations",
        "analysis": "Report all five cases, all repeats and failures. First crossing is a grid bracket only; no monotonicity assumption or interpolation. Use all DROP/ADD repeat combinations for sign/effect stability. Primary AbsRel and secondary metrics kept separate.",
        "pilot_sources_sha256": {f"{b}/utilities.json": sha256(PILOT/b/"utilities.json") for b in ("vggt", "mast3r")},
        "pilot_manifest_sha256": sha256(PILOT / "inputs.json"),
        "code_sha256": {name: sha256(ROOT / "experiments" / name) for name in CODE},
        "controls": controls, "variants": variants,
    }
    atomic_json(out / "inputs.json", manifest)
    atomic_json(out / "protocol.json", protocol)
    print("FROZEN", out / "protocol.json", "new reconstructions", protocol["new_reconstructions"], flush=True)


def verify_inputs(out, protocol):
    if sha256(out / "inputs.json") != protocol["pilot_manifest_sha256"]:
        raise ValueError("Frozen manifest differs from pilot")
    manifest = read_json(out / "inputs.json")
    for frame in manifest["frames"]:
        if sha256(ROOT / frame["depth"]) != frame["depth_sha256"]:
            raise ValueError("GT depth changed")
        for variant in frame["variants"].values():
            if sha256(ROOT / variant["path"]) != variant["sha256"]:
                raise ValueError("Pilot image changed")
    for variants in protocol["variants"].values():
        for variant in variants.values():
            if sha256(ROOT / variant["path"]) != variant["sha256"]:
                raise ValueError("Follow-up image changed")
    return manifest


def audit_policy(out):
    protocol = frozen_protocol(out)
    manifest = verify_inputs(out, protocol)
    frames = {f["id"]: f for f in manifest["frames"]}
    rows, identical, previous_failures = [], 0, 0
    for backend in ("vggt", "mast3r"):
        refs = {i: reference_depth(ROOT/f["depth"], backend) for i, f in frames.items()}
        original = {(r["size"], r["target"], r["blur_deg"], r["metric"]): r
                    for r in read_json(PILOT/backend/"utilities.json")}

        @lru_cache(maxsize=3)
        def prediction(key):
            return load_prediction(PILOT/backend/"predictions"/(key+".npz"))

        for size in (5, 10, 20):
            for target in manifest["sets"][str(size)]:
                common = [i for i in manifest["sets"][str(size)] if i != target]
                drop, drop_diag, drop_error = fixed_mask_geometry(prediction(f"m{size}_drop_{target}"), common, refs)
                for degree in manifest["severities_deg"]:
                    key = f"m{size}_clean" if degree == 0 else f"m{size}_add_{target}_b{degree:g}"
                    add, add_diag, add_error = fixed_mask_geometry(prediction(key), common, refs)
                    row = {"backend": backend, "size": size, "target": target, "blur_deg": degree,
                           "drop_diagnostics": drop_diag, "add_diagnostics": add_diag,
                           "drop_error": drop_error, "add_error": add_error, "metrics": {}}
                    for metric in ("geometry_depth_absrel", "geometry_depth_rmse_m"):
                        before = original[(size, target, degree, metric)]
                        if before["status"] == "ok":
                            assert drop[metric] == before["E_drop"] and add[metric] == before["E_add"]
                            identical += 1
                        else:
                            previous_failures += 1
                        row["metrics"][metric] = {"E_drop": drop.get(metric), "E_add": add.get(metric),
                                                  "utility_raw": drop[metric]-add[metric] if not drop_error and not add_error else None,
                                                  "legacy_status": before["status"]}
                    rows.append(row)
            print("AUDITED", backend, size, flush=True)
    assert identical == 712 and previous_failures == 128 and len(rows) == 420
    atomic_json(out / "policy_audit.json", {"policy_version": POLICY["version"], "pilot_reanalysis": True,
                                           "legacy_results_modified": False, "identical_valid_legacy_metric_rows": identical,
                                           "previously_invalid_metric_rows": previous_failures,
                                           "conditions": len(rows), "rows": rows})
    print("POLICY AUDIT PASS", identical, "legacy-valid metric rows exactly unchanged", flush=True)


def run(out, backend, device, replicate):
    protocol = frozen_protocol(out)
    manifest = verify_inputs(out, protocol)
    frames = {f["id"]: f for f in manifest["frames"]}
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; run with GPU access or explicitly choose another device/output root")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    args = argparse.Namespace(backend=backend, device=device, optimizer_device=device)
    fingerprint = environment(args, out/"inputs.json")
    fingerprint["followup_protocol_sha256"] = sha256(out / "protocol.json")
    folder = out / backend
    folder.mkdir(exist_ok=True)
    from filelock import FileLock
    with FileLock(str(folder/"environment.json")+".lock"):
        if (folder/"environment.json").exists() and read_json(folder/"environment.json") != fingerprint:
            raise ValueError("Hardware/software fingerprint changed; use a new output root")
        atomic_json(folder / "environment.json", fingerprint)
    refs = {i: reference_depth(ROOT/f["depth"], backend) for i, f in frames.items()}
    poses = {i: f["c2w"] for i, f in frames.items()}
    replicates = range(POLICY["replicates"]) if replicate == "all" else [int(replicate)]
    model = None
    for rep in replicates:
        block = folder / f"replicate_{rep}"
        block.mkdir(exist_ok=True)
        predictions = block / "predictions"
        predictions.mkdir(exist_ok=True)
        if model is not None and backend == "mast3r":
            model.pair_cache = block / "pair_cache"
        for case in protocol["cases"]:
            if case["backend"] != backend:
                continue
            cid = case_id(case)
            ids = manifest["sets"][str(case["size"])]
            common = [i for i in ids if i != case["target"]]
            # Clean first fills the within-block neural cache; DROP and blur align independently.
            conditions = [("add_b0", 0)] + [("drop", None)] + [(f"add_b{b:g}", b) for b in case["degrees"] if b]
            for condition, degree in conditions:
                key = cid + "_" + condition
                record_path = block / (key+".json")
                pred_path = predictions / (key+".npz")
                if record_path.exists():
                    record = read_json(record_path)
                    if record["status"] == "ok":
                        if not pred_path.exists() or sha256(pred_path) != record["prediction_sha256"]:
                            raise ValueError("Saved prediction missing or changed")
                        print("REUSED", backend, rep, key, flush=True)
                        continue
                selected = common if degree is None else ids
                paths = []
                for frame in selected:
                    variant = frames[frame]["variants"]["0"]
                    if frame == case["target"] and degree:
                        variant = (protocol["variants"][frame][f"{degree:g}"] if backend == "vggt"
                                   else frames[frame]["variants"][f"{degree:g}"])
                    paths.append(str(ROOT / variant["path"]))
                record = {"backend": backend, "case": cid, "size": case["size"], "target": case["target"],
                          "replicate": rep, "condition": condition, "blur_deg": degree, "ids": selected,
                          "input_sha256": [sha256(Path(p)) for p in paths], "common_ids": common,
                          "start_unix": time.time(), "status": "running", "pid": os.getpid()}
                atomic_json(block / "status.json", record)
                print("START", backend, rep, key, flush=True)
                if model is None:
                    model = load_backend(backend, device, block/"cache", optimizer_device=device,
                                         pair_cache=block/"pair_cache" if backend == "mast3r" else None)
                np.random.seed(POLICY["seed_per_reconstruction"])
                torch.manual_seed(POLICY["seed_per_reconstruction"])
                start = time.time()
                try:
                    prediction = model.predict(selected, paths, key)
                    if list(prediction["ids"]) != selected:
                        raise ValueError("Backend changed input image order")
                    temp = pred_path.with_suffix(".tmp.npz")
                    np.savez_compressed(temp, **prediction)
                    temp.replace(pred_path)
                    record.update(evaluate_followup(prediction, common, poses, refs))
                    record["prediction_sha256"] = sha256(pred_path)
                    record["status"] = "ok" if not record["errors"] else "partial"
                except Exception as error:
                    record.update(status="failed", error=repr(error))
                    if "out of memory" in str(error).lower():
                        raise
                finally:
                    record["seconds"] = time.time()-start
                    record["finished_unix"] = time.time()
                    atomic_json(record_path, record)
                    atomic_json(block / "status.json", record)
                print("DONE", backend, rep, key, record["status"], round(record["seconds"], 2), flush=True)
                if device == "mps":
                    torch.mps.empty_cache()
        atomic_json(block/"status.json", {"backend": backend, "replicate": rep, "status": "complete",
                                         "finished_unix": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "audit", "run"))
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--backend", choices=("vggt", "mast3r"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--replicate", choices=("all", "0", "1", "2"), default="all")
    args = parser.parse_args()
    out = ROOT / args.output
    if args.phase == "prepare":
        prepare(out)
    elif args.phase == "audit":
        audit_policy(out)
    else:
        if args.backend is None:
            parser.error("run requires --backend")
        run(out, args.backend, args.device, args.replicate)
