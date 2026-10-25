"""Descriptive analysis of frozen FrameUtility results, with missingness and noise audits."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from backends import ROOT, reference_depth
from metrics import METRICS, condition_counts

PRIMARY = ["geometry_depth_absrel", "pose_ate_normalized", "pose_pair_rotation_deg"]
NAMES = {"geometry_depth_absrel": "Depth AbsRel", "geometry_depth_rmse_m": "Depth RMSE",
         "pose_ate_normalized": "Pose ATE", "pose_pair_translation_relative": "Pair translation",
         "pose_pair_rotation_deg": "Pair rotation"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_curve(group, noise):
    valid = {r["blur_deg"]: r for r in group if r["status"] == "ok" and r["utility"] is not None}
    levels = sorted(valid)
    complete = len(valid) == 6
    raw = {b: valid[b]["utility_raw"] for b in levels}
    clean = raw.get(0)
    output = {"complete": complete, "n_valid": len(valid), "clean_raw": clean,
              "last_raw": raw.get(4), "clean_delta": noise,
              "clean_positive": clean is not None and clean > 0,
              "any_clean_to_negative": complete and clean > 0 and any(raw[b] < 0 for b in levels[1:]),
              "endpoint_clean_to_negative": complete and clean > 0 and raw[4] < 0,
              "first_bracket": None, "noise_screens": {}}
    if complete:
        first = next((i for i, b in enumerate(levels[1:], 1) if clean > 0 and raw[b] < 0), None)
        if first is not None:
            output["first_bracket"] = [levels[first-1], levels[first]]
        output["end_worse_than_clean"] = raw[4] < raw[0]
        output["end_harmful_to_beneficial"] = raw[0] < 0 < raw[4]
        output["spearman_blur_utility"] = float(spearmanr(levels, [raw[b] for b in levels]).statistic)
        output["strictly_nonincreasing"] = all(raw[b] <= raw[a] for a, b in zip(levels, levels[1:]))
        output["returns_positive_after_negative"] = first is not None and any(raw[b] > 0 for b in levels[first+1:])
    if noise is not None and complete:
        for multiplier in (1, 3):
            threshold = multiplier*noise
            positive = raw[0] > threshold
            output["noise_screens"][str(multiplier)] = {
                "threshold_raw": threshold, "clean_clearly_positive": positive,
                "any_transition": positive and any(raw[b] < -threshold for b in levels[1:]),
                "endpoint_transition": positive and raw[4] < -threshold}
    return output


def audit(folder):
    out = folder / "analysis"
    out.mkdir(exist_ok=True)
    manifest = json.loads((folder / "inputs.json").read_text())
    levels = manifest["severities_deg"]
    rows, noise, sources = [], {}, {}
    for backend in ("vggt", "mast3r"):
        path = folder / backend / "utilities.json"
        sources[str(path.relative_to(folder))] = digest(path)
        current = json.loads(path.read_text())
        expected = {(size, target, b, m) for size in (5, 10, 20)
                    for target in manifest["sets"][str(size)] for b in levels for m in METRICS}
        keys = [(r["size"], r["target"], r["blur_deg"], r["metric"]) for r in current]
        assert len(keys) == len(set(keys)) == 1050 and set(keys) == expected
        assert all(r["n_context"] == r["size"]-1 for r in current)
        for r in current:
            if r["status"] == "ok":
                assert all(np.isfinite(r[k]) for k in ("E_drop", "E_add", "utility_raw"))
        rows.extend(current)
        for r in json.loads((folder / backend / "repeatability.json").read_text()):
            for metric, delta in r["absolute_metric_delta"].items():
                noise[(backend, r["size"], r["target"], metric)] = delta
    curves = defaultdict(list)
    for r in rows:
        curves[(r["backend"], r["size"], r["target"], r["metric"])].append(r)
    curve_stats = []
    for key, group in sorted(curves.items()):
        baseline = {r["E_drop"] for r in group if r.get("E_drop") is not None}
        assert len(baseline) <= 1
        assert len({r["evaluation_pixels"] for r in group if r.get("evaluation_pixels") is not None}) == 1
        result = summarize_curve(group, noise.get(key))
        result.update(dict(zip(("backend", "size", "target", "metric"), key)))
        curve_stats.append(result)
    aggregates = []
    for backend in ("vggt", "mast3r"):
        for size in (5, 10, 20, "all"):
            for metric in METRICS:
                selected = [c for c in curve_stats if c["backend"] == backend and c["metric"] == metric
                            and (size == "all" or c["size"] == size)]
                complete = [c for c in selected if c["complete"]]
                record = {"backend": backend, "size": size, "metric": metric,
                          "n_curves": len(selected), "n_complete": len(complete),
                          "clean_positive_complete": sum(c["clean_positive"] for c in complete),
                          "any_transition": sum(c["any_clean_to_negative"] for c in complete),
                          "endpoint_transition": sum(c["endpoint_clean_to_negative"] for c in complete),
                          "end_worse_than_clean": sum(c["end_worse_than_clean"] for c in complete),
                          "reverse_endpoint": sum(c["end_harmful_to_beneficial"] for c in complete),
                          "monotone_decreasing": sum(c["strictly_nonincreasing"] for c in complete),
                          "recovers_after_crossing": sum(c["returns_positive_after_negative"] for c in complete),
                          "median_spearman": float(np.median([c["spearman_blur_utility"] for c in complete])) if complete else None,
                          "brackets": dict(Counter(str(c["first_bracket"]) for c in complete if c["first_bracket"])),
                          "noise_screens": {}}
                for multiplier in (1, 3):
                    available = [c["noise_screens"][str(multiplier)] for c in complete if str(multiplier) in c["noise_screens"]]
                    record["noise_screens"][str(multiplier)] = {"n": len(available),
                        "clean_positive": sum(c["clean_clearly_positive"] for c in available),
                        "any_transition": sum(c["any_transition"] for c in available),
                        "endpoint_transition": sum(c["endpoint_transition"] for c in available)}
                record["severity"] = []
                for b in levels:
                    available = [r for r in rows if r["backend"] == backend and r["metric"] == metric
                                 and r["blur_deg"] == b and (size == "all" or r["size"] == size)]
                    valid = [r for r in available if r["status"] == "ok" and r["utility"] is not None]
                    balanced = [r for r in valid if any(c["size"] == r["size"] and c["target"] == r["target"] for c in complete)]
                    record["severity"].append({"blur": b, "planned": len(available), "valid": len(valid),
                        "failed": sum(r["status"] != "ok" for r in available),
                        "negative": sum(r["utility"] < 0 for r in valid),
                        "median_u_available": float(np.median([r["utility"] for r in valid])) if valid else None,
                        "median_u_complete": float(np.median([r["utility"] for r in balanced])) if balanced else None,
                        "median_drop_error": float(np.median([r["E_drop"] for r in valid])) if valid else None})
                aggregates.append(record)
    failures = [r for r in rows if r["metric"] == "geometry_depth_absrel" and r["status"] != "ok"]
    failure_summary = {"by_backend_size": {f"{b}_{s}": sum(r["backend"] == b and r["size"] == s for r in failures)
                       for b in ("vggt", "mast3r") for s in (5, 10, 20)},
                       "drop_involved": sum("DROP geometry:" in r["error"] for r in failures),
                       "add_only": sum("DROP geometry:" not in r["error"] for r in failures),
                       "failed_drop_contexts": len({(r["backend"], r["size"], r["target"]) for r in failures if "DROP geometry:" in r["error"]}),
                       "errors": dict(Counter(r["error"] for r in failures))}
    paired = []
    for metric in METRICS:
        eligible = {(c["size"], c["target"]) for c in curve_stats if c["backend"] == "mast3r" and c["metric"] == metric and c["complete"]}
        for size in (5, 10, 20, "all"):
            targets = {(s, t) for s, t in eligible if size == "all" or s == size}
            comparison = {"metric": metric, "size": size, "n_matched_complete": len(targets), "severity": []}
            for b in levels:
                model_rows = {model: [r for r in rows if r["backend"] == model and r["metric"] == metric
                              and r["blur_deg"] == b and (r["size"], r["target"]) in targets] for model in ("vggt", "mast3r")}
                comparison["severity"].append({"blur": b, **{model: {
                    "median_utility": float(np.median([r["utility"] for r in rs])) if rs else None,
                    "negative": sum(r["utility"] < 0 for r in rs)} for model, rs in model_rows.items()}})
            paired.append(comparison)
    result = {"integrity": {"rows": len(rows), "conditions": condition_counts(rows), "expected_rows": 2100,
                           "unique_targets": len(manifest["frames"]), "scenes": [manifest["scene"]]},
              "aggregates": aggregates, "curves": curve_stats, "failures": failure_summary,
              "matched_comparisons": paired, "sources_sha256": sources}
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot_curves(rows, curve_stats, levels, out)
    plot_negative_rates(aggregates, levels, out)
    plot_examples(rows, manifest, levels, out)
    audit_invalid_depths(folder, manifest, failures, out)
    print(json.dumps(result["integrity"]))
    print(json.dumps(failure_summary))
    for r in aggregates:
        if r["size"] == "all":
            print(r["backend"], r["metric"], "complete", r["n_complete"], "clean+", r["clean_positive_complete"],
                  "cross", r["any_transition"], "endcross", r["endpoint_transition"], "screen3", r["noise_screens"]["3"])


def plot_curves(rows, curves, levels, out):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    for row_index, size in enumerate((5, 10, 20)):
        for col, metric in enumerate(PRIMARY):
            ax = axes[row_index, col]
            for model, color in (("vggt", "#2868b2"), ("mast3r", "#c65d23")):
                targets = {c["target"] for c in curves if c["backend"] == model and c["size"] == size
                           and c["metric"] == metric and c["complete"]}
                data = np.array([[next(r["utility"] for r in rows if r["backend"] == model and r["size"] == size
                                 and r["metric"] == metric and r["target"] == target and r["blur_deg"] == b)
                                  for b in levels] for target in sorted(targets)])
                if not len(data):
                    continue
                data *= 100
                ax.fill_between(levels, np.percentile(data, 25, axis=0), np.percentile(data, 75, axis=0), color=color, alpha=.13)
                ax.plot(levels, np.median(data, axis=0), "o-", color=color, markersize=4, label=f"{model.upper()} n={len(targets)}")
            ax.axhline(0, color="#555555", linewidth=.8)
            ax.set_title(f"{size} views · {NAMES[metric]}")
            ax.set_ylabel("U (AbsRel percentage points)" if metric.startswith("geometry") else "U (relative error reduction, %)")
            ax.grid(alpha=.15)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            if row_index == 2:
                ax.set_xlabel("Exposure yaw path (degrees)")
    fig.suptitle("Marginal utility · medians and target IQR · complete curves only", fontsize=15)
    fig.tight_layout(rect=(0, .02, 1, .96))
    fig.text(.5, .006, "IQR describes this scene's target/context spread; it is not a confidence interval. Curve counts differ when geometry is invalid.", ha="center", fontsize=9)
    fig.savefig(out / "utility_median_iqr.png", dpi=170)
    plt.close(fig)


def plot_negative_rates(aggregates, levels, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    groups = [(model, size) for model in ("vggt", "mast3r") for size in (5, 10, 20)]
    for ax, metric in zip(axes, PRIMARY):
        stats = [next(r for r in aggregates if r["backend"] == model and r["size"] == size and r["metric"] == metric) for model, size in groups]
        values = np.array([[r["negative"]/r["valid"] if r["valid"] else np.nan for r in row["severity"]] for row in stats])
        cmap = plt.get_cmap("YlOrRd").copy()
        cmap.set_bad("#dddddd")
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i, row in enumerate(stats):
            for j, cell in enumerate(row["severity"]):
                text = f"{cell['negative']}/{cell['valid']}" if cell["valid"] else "NA"
                ax.text(j, i, text, ha="center", va="center", color="white" if values[i, j] > .65 else "black", fontsize=10)
        ax.set_xticks(range(6), [str(b) for b in levels])
        ax.set_yticks(range(6), [f"{m.upper()} · {s}" for m, s in groups])
        ax.set_xlabel("Exposure yaw path (degrees)")
        ax.set_title(NAMES[metric])
    fig.colorbar(im, ax=axes.tolist(), label="Fraction U < 0 among valid conditions", fraction=.025, pad=.025)
    fig.suptitle("Negative utility · cells show harmful / valid at each blur level", y=.98, fontsize=14)
    fig.subplots_adjust(left=.12, right=.87, top=.87, bottom=.15, wspace=.27)
    fig.savefig(out / "negative_utility_rates.png", dpi=170)
    plt.close(fig)


def audit_invalid_depths(folder, manifest, failures, out):
    frames = {f["id"]: f for f in manifest["frames"]}
    refs = {i: reference_depth(ROOT / f["depth"], "mast3r") for i, f in frames.items()}

    @lru_cache(maxsize=4)
    def load_depths(key):
        with np.load(folder / "mast3r/predictions" / (key + ".npz"), allow_pickle=False) as z:
            return {str(i): d for i, d in zip(z["ids"], z["depth"])}

    details = []
    for r in failures:
        common = [i for i in manifest["sets"][str(r["size"]) ] if i != r["target"]]
        drop = f"m{r['size']}_drop_{r['target']}"
        add = f"m{r['size']}_clean" if r["blur_deg"] == 0 else f"m{r['size']}_add_{r['target']}_b{r['blur_deg']:g}"
        for side, key in (("DROP", drop), ("ADD", add)):
            if side + " geometry:" not in r["error"]:
                continue
            depths = load_depths(key)
            invalid = pixels = nonfinite = 0
            bad_frames = []
            for frame in common:
                mask = (refs[frame] > 0) & np.isfinite(refs[frame])
                values = depths[frame][mask]
                bad = (~np.isfinite(values)) | (values <= 0)
                count = int(bad.sum())
                pixels += len(values)
                invalid += count
                nonfinite += int(np.sum(~np.isfinite(values)))
                if count:
                    bad_frames.append({"frame": frame, "invalid": count, "min_depth": float(np.nanmin(values))})
            assert invalid > 0
            details.append({"size": r["size"], "target": r["target"], "blur": r["blur_deg"],
                            "side": side, "prediction": key, "invalid_pixels": invalid, "evaluation_pixels": pixels,
                            "invalid_fraction": invalid/pixels, "nonfinite": nonfinite, "bad_frames": bad_frames})
    (out / "invalid_depth_audit.json").write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")


def plot_examples(rows, manifest, levels, out):
    cases = [("vggt", 5, "DSC_0250", "geometry_depth_absrel"),
             ("mast3r", 20, "DSC_0238", "geometry_depth_absrel"),
             ("mast3r", 20, "DSC_0238", "pose_pair_rotation_deg")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    examples = []
    for ax, (backend, size, target, metric) in zip(axes, cases):
        selected = sorted([r for r in rows if r["backend"] == backend and r["size"] == size
                           and r["target"] == target and r["metric"] == metric], key=lambda r: r["blur_deg"])
        assert len(selected) == 6 and all(r["status"] == "ok" for r in selected)
        multiplier = 100 if metric.startswith("geometry") else 1
        added = np.array([r["E_add"] for r in selected])*multiplier
        dropped = selected[0]["E_drop"]*multiplier
        ax.axhline(dropped, color="#555555", linestyle="--", label="DROP")
        ax.plot(levels, added, "o-", color="#2868b2", label="ADD")
        ax.fill_between(levels, added, dropped, where=added > dropped, interpolate=True, color="#d45a57", alpha=.22)
        ax.fill_between(levels, added, dropped, where=added < dropped, interpolate=True, color="#60a274", alpha=.14)
        ax.set_title(f"{backend.upper()} · {size} views\n{target}")
        ax.set_ylabel("Depth AbsRel (%)" if metric.startswith("geometry") else "Mean pair rotation error (degrees)")
        ax.set_xlabel("Exposure yaw path (degrees)")
        ax.grid(alpha=.15)
        ax.legend(fontsize=9)
        examples.append({"backend": backend, "size": size, "target": target, "metric": metric,
                         "selection": "Illustrative observed clean-to-harmful endpoint; not a random sample",
                         "conditions": selected})
    fig.suptitle("Concrete add-vs-drop examples · lower error is better", fontsize=15)
    fig.tight_layout(rect=(0, .07, 1, .94))
    fig.text(.5, .015, "Red shading: ADD error exceeds DROP, so U < 0. Lines guide the eye; no interpolated threshold is estimated.", ha="center", fontsize=9)
    fig.savefig(out / "counterfactual_examples.png", dpi=180)
    fig.savefig(out / "counterfactual_examples.svg")
    plt.close(fig)
    (out / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    frames = {f["id"]: f for f in manifest["frames"]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, b in zip(axes, (0, 4)):
        im = plt.imread(ROOT / frames["DSC_0238"]["variants"][str(b)]["path"])
        ax.imshow(im)
        ax.set_title(f"{'Clean' if b == 0 else 'Blur'} · full yaw exposure path {b}°")
        ax.axis("off")
    fig.suptitle("Same target and nominal viewpoint · DSC_0238", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "target_intervention.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit("Retired one-off/legacy entry point. Review EXPERIMENT_PLAN.md and use the approved experiments/run_all.py workflow.")
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default="results/office_fast")
    audit(ROOT / parser.parse_args().folder)
