"""Post-hoc sensitivity only: finite nonpositive depth receives error on the fixed GT mask.

Primary strict scores remain untouched. Fit a positive median scale from positive
predictions, then score every finite prediction (including zero/negative values).
"""
from collections import defaultdict
from functools import lru_cache
import json

import numpy as np

from analyze_results import summarize_curve
from backends import ROOT, reference_depth

FOLDER = ROOT / "results/office_fast"
METRICS = ("geometry_depth_absrel", "geometry_depth_rmse_m")


@lru_cache(maxsize=8)
def prediction(backend, key):
    with np.load(FOLDER / backend / "predictions" / (key + ".npz"), allow_pickle=False) as data:
        return {str(frame): depth for frame, depth in zip(data["ids"], data["depth"])}


def score(depths, ids, refs):
    pd, gd = [], []
    for frame in ids:
        mask = (refs[frame] > 0) & np.isfinite(refs[frame])
        pd.append(depths[frame][mask].astype(np.float64))
        gd.append(refs[frame][mask])
    pd, gd = np.concatenate(pd), np.concatenate(gd)
    if not np.isfinite(pd).all():
        raise ValueError("Nonfinite depth still cannot be scored")
    positive = pd > 0
    if positive.sum() < 100:
        raise ValueError("Insufficient positive depths to estimate a positive scale")
    scale = np.median(gd[positive]/pd[positive])
    assert np.isfinite(scale) and scale > 0
    difference = scale*pd-gd
    return {"geometry_depth_absrel": float(np.mean(np.abs(difference)/gd)),
            "geometry_depth_rmse_m": float(np.sqrt(np.mean(difference**2))),
            "nonpositive_fraction": float(np.mean(~positive)), "scale": float(scale)}


def main():
    manifest = json.loads((FOLDER / "inputs.json").read_text())
    frames = {f["id"]: f for f in manifest["frames"]}
    summaries, rows, curve_stats = [], [], []
    identical = 0
    for backend in ("vggt", "mast3r"):
        refs = {i: reference_depth(ROOT/f["depth"], backend) for i, f in frames.items()}
        primary = json.loads((FOLDER / backend / "utilities.json").read_text())
        primary = {(r["size"], r["target"], r["blur_deg"], r["metric"]): r for r in primary}
        for size in (5, 10, 20):
            for target in manifest["sets"][str(size)]:
                ids = [i for i in manifest["sets"][str(size)] if i != target]
                drop = score(prediction(backend, f"m{size}_drop_{target}"), ids, refs)
                clean = score(prediction(backend, f"m{size}_clean"), ids, refs)
                repeated = score(prediction(backend, f"m{size}_clean_repeat"), ids, refs)
                group = defaultdict(list)
                for b in manifest["severities_deg"]:
                    add = clean if b == 0 else score(prediction(backend, f"m{size}_add_{target}_b{b:g}"), ids, refs)
                    for metric in METRICS:
                        original = primary[(size, target, b, metric)]
                        if original["status"] == "ok":
                            assert drop[metric] == original["E_drop"] and add[metric] == original["E_add"]
                            identical += 1
                        row = {"backend": backend, "size": size, "target": target, "blur_deg": b, "metric": metric,
                               "status": "ok", "E_drop": drop[metric], "E_add": add[metric],
                               "utility": drop[metric]-add[metric], "utility_raw": drop[metric]-add[metric],
                               "drop_nonpositive_fraction": drop["nonpositive_fraction"],
                               "add_nonpositive_fraction": add["nonpositive_fraction"]}
                        rows.append(row)
                        group[metric].append(row)
                for metric in METRICS:
                    summary = summarize_curve(group[metric], abs(clean[metric]-repeated[metric]))
                    summary.update(backend=backend, size=size, target=target, metric=metric)
                    curve_stats.append(summary)
        print(backend, "post-hoc sensitivity scored", flush=True)
    for backend in ("vggt", "mast3r"):
        for size in (5, 10, 20, "all"):
            for metric in METRICS:
                selected = [c for c in curve_stats if c["backend"] == backend and c["metric"] == metric
                            and (size == "all" or c["size"] == size)]
                summary = {"backend": backend, "size": size, "metric": metric, "n": len(selected),
                           "clean_positive": sum(c["clean_positive"] for c in selected),
                           "any_transition": sum(c["any_clean_to_negative"] for c in selected),
                           "endpoint_transition": sum(c["endpoint_clean_to_negative"] for c in selected)}
                available = [c["noise_screens"]["3"] for c in selected]
                summary["noise_screen3"] = {"clean_positive": sum(c["clean_clearly_positive"] for c in available),
                                            "any_transition": sum(c["any_transition"] for c in available),
                                            "endpoint_transition": sum(c["endpoint_transition"] for c in available)}
                summaries.append(summary)
    result = {"post_hoc": True,
              "definition": "Scale = median(GT/pred) over positive predictions; errors evaluated on the entire fixed GT mask, including finite zero/negative predictions; nonfinite remains a failure",
              "primary_results_modified": False, "identical_primary_valid_metric_rows": identical,
              "summaries": summaries, "curves": curve_stats, "rows": rows}
    (FOLDER / "analysis/depth_sensitivity.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Unchanged valid primary geometry rows:", identical)
    for r in summaries:
        if r["size"] == "all":
            print(r)


if __name__ == "__main__":
    main()
