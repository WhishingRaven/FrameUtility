"""Repeat only the selected DROP and severe ADD to audit numerical stability."""
import json
import os
from pathlib import Path
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import numpy as np
import torch

from backends import MASt3RBackend, ROOT, reference_depth
from metrics import METRICS, evaluate_available


def main():
    folder = ROOT / "results/office_fast"
    out = folder / "analysis/case_repeats"
    out.mkdir(exist_ok=True)
    manifest = json.loads((folder / "inputs.json").read_text())
    frames = {f["id"]: f for f in manifest["frames"]}
    ids = manifest["sets"]["20"]
    target = "DSC_0238"
    common = [i for i in ids if i != target]
    poses = {i: frames[i]["c2w"] for i in common}
    refs = {i: reference_depth(ROOT / frames[i]["depth"], "mast3r") for i in common}
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    model = MASt3RBackend("mps", out / "cache", optimizer_device="mps",
                         pair_cache=ROOT / "results/office_sweep/mast3r/mast3r_cache/pairs")
    report = {"backend": "mast3r", "size": 20, "target": target,
              "primary_results_modified": False, "repeats": {}}
    for kind in ("drop", "add4"):
        selected = common if kind == "drop" else ids
        paths = [str(ROOT / frames[i]["variants"]["4" if i == target else "0"]["path"]) for i in selected]
        torch.manual_seed(2026)
        np.random.seed(2026)
        start = time.time()
        print("START repeat", kind, flush=True)
        result = model.predict(selected, paths, "verify_" + kind)
        np.savez_compressed(out / (kind + ".npz"), **result)
        values, errors = evaluate_available(result, common, poses, refs)
        key = f"m20_drop_{target}" if kind == "drop" else f"m20_add_{target}_b4"
        with np.load(folder / "mast3r/predictions" / (key + ".npz"), allow_pickle=False) as original:
            original_values, original_errors = evaluate_available(original, common, poses, refs)
        report["repeats"][kind] = {"seconds": time.time()-start, "metrics": values, "errors": errors,
                                   "original_metrics": original_values, "original_errors": original_errors,
                                   "absolute_delta": {m: abs(values[m]-original_values[m]) for m in METRICS
                                                      if m in values and m in original_values}}
        (out / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("DONE", kind, report["repeats"][kind], flush=True)
    with np.load(folder / "mast3r/predictions/m20_clean.npz", allow_pickle=False) as original:
        clean, clean_errors = evaluate_available(original, common, poses, refs)
    with np.load(folder / "mast3r/predictions/m20_clean_repeat.npz", allow_pickle=False) as original:
        clean_repeat, clean_repeat_errors = evaluate_available(original, common, poses, refs)
    report["sign_stability"] = {}
    for metric in METRICS:
        drop = report["repeats"]["drop"]
        add = report["repeats"]["add4"]
        if not all(metric in data for data in (drop["metrics"], drop["original_metrics"],
                                               add["metrics"], add["original_metrics"], clean, clean_repeat)):
            continue
        drops = [drop["metrics"][metric], drop["original_metrics"][metric]]
        cleans = [clean[metric], clean_repeat[metric]]
        bads = [add["metrics"][metric], add["original_metrics"][metric]]
        report["sign_stability"][metric] = {
            "clean_raw_u_range": [min(d-c for d in drops for c in cleans), max(d-c for d in drops for c in cleans)],
            "blur4_raw_u_range": [min(d-b for d in drops for b in bads), max(d-b for d in drops for b in bads)],
            "all_clean_combinations_positive": all(d > c for d in drops for c in cleans),
            "all_blur4_combinations_negative": all(d < b for d in drops for b in bads)}
    (out / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("SIGN STABILITY", report["sign_stability"], flush=True)


if __name__ == "__main__":
    raise SystemExit("Retired one-off/legacy entry point. Review EXPERIMENT_PLAN.md and use the approved experiments/run_all.py workflow.")
    main()
