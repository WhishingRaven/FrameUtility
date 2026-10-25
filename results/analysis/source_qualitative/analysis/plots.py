"""Scientific figures from audited table values; no model inference."""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import PRIMARY, LABELS, CATEGORIES, representative_ids

COLORS = {"vggt": "#2366ad", "mast3r": "#d06b20"}
CLASS_COLORS = ("#3b936f", "#c34747", "#c4c8cb", "#353c49")


def save(fig, folder, name):
    fig.savefig(folder / (name + ".png"), dpi=170, bbox_inches="tight")
    fig.savefig(folder / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def axis(ax, levels, ylabel=None):
    ax.set_xticks(range(len(levels)), [f"{b:g}" for b in levels], rotation=50, fontsize=8)
    # Shared x axes otherwise hide the numbers on all but the bottom row.
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel("Yaw span (deg)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=.18)


def make_scene_figures(data, stages, contexts, protocol, folder):
    folder.mkdir(parents=True, exist_ok=True)
    scene = data["info"]["scene"]
    levels, sizes, backends = protocol["blur_degrees"], protocol["sizes"], protocol["backends"]
    x = np.arange(len(levels))
    index = {(r["backend"], r["size"], r["metric"], r["blur_deg"]): r for r in stages}
    rows = data["rows"]
    fig, axes = plt.subplots(len(sizes), len(backends), figsize=(15, 10), squeeze=False, sharex=True)
    for ri, size in enumerate(sizes):
        panel_values = []
        for ci, backend in enumerate(backends):
            ax = axes[ri, ci]
            for target in data["manifest"]["sets"][str(size)]:
                values = [next(r["utility_raw"] for r in rows if r["backend"] == backend and r["size"] == size
                               and r["target"] == target and r["metric"] == PRIMARY and r["blur_deg"] == b) for b in levels]
                ax.plot(x, [np.nan if v is None else v * 100 for v in values], color=COLORS[backend], alpha=.26, lw=.8)
            group = [index[(backend, size, PRIMARY, b)] for b in levels]
            med = [r["utility_raw"]["median"] for r in group]
            convert = lambda values: [np.nan if v is None else v * 100 for v in values]
            ax.fill_between(x, convert([r["utility_raw"]["q25"] for r in group]), convert([r["utility_raw"]["q75"] for r in group]),
                            color=COLORS[backend], alpha=.2, label="Target IQR (not a confidence interval)")
            ax.plot(x, convert(med), "o-", color=COLORS[backend], lw=2, ms=3, label="Median across targets")
            eps = protocol["evaluation"]["primary_absrel_effect_floor"] * 100
            ax.axhspan(-eps, eps, color="gray", alpha=.12)
            ax.axhline(0, color="#333333", lw=.8)
            counts = [r["valid"] for r in group]
            ax.set_title(f"{scene} | {backend.upper()} | {size} views | planned {size}, valid {min(counts)}–{max(counts)}")
            axis(ax, levels, "Depth U (pp; + helpful)")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7)
            panel_values.extend(v * 100 for r in rows if r["backend"] == backend and r["size"] == size and r["metric"] == PRIMARY
                                for v in [r["utility_raw"]] if v is not None)
        if panel_values:
            lo, hi = min(panel_values), max(panel_values)
            padding = max((hi - lo) * .08, .005)
            for ax in axes[ri]:
                ax.set_ylim(lo - padding, hi + padding)
    fig.suptitle("Positive = helpful; negative = harmful | Thin lines: every target | Shading: target IQR\n"
                 "Discrete observed stages; connecting lines do not locate a continuous threshold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, folder, "depth_utility")

    fig, axes = plt.subplots(len(sizes), len(backends), figsize=(15, 10), squeeze=False, sharex=True)
    for ri, size in enumerate(sizes):
        for ci, backend in enumerate(backends):
            ax = axes[ri, ci]
            group = [index[(backend, size, PRIMARY, b)] for b in levels]
            bottom = np.zeros(len(levels))
            for category, color in zip(CATEGORIES, CLASS_COLORS):
                values = np.array([r[category + "_fraction_planned"] * 100 for r in group])
                ax.bar(x, values, bottom=bottom, color=color, width=.85, label=category)
                bottom += values
            ax.set_ylim(0, 100)
            ax.set_title(f"{backend.upper()} | {size} views | planned denominator N={size}")
            axis(ax, levels, "% of all planned targets")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .96), ncol=4, fontsize=9)
    fig.suptitle(f"{scene}: primary depth classification | epsilon = {protocol['evaluation']['primary_absrel_effect_floor'] * 100:g} percentage points")
    fig.tight_layout(rect=(0, 0, 1, .92))
    save(fig, folder, "depth_classification")

    fig, axes = plt.subplots(len(sizes), len(backends), figsize=(15, 10), squeeze=False, sharex=True)
    for ri, size in enumerate(sizes):
        for ci, backend in enumerate(backends):
            ax = axes[ri, ci]
            group = [index[(backend, size, PRIMARY, b)] for b in levels]
            for field, color, label in (("E_add", COLORS[backend], "ADD"), ("E_drop", "#575e68", "DROP")):
                convert = lambda stat: [np.nan if r[field][stat] is None else r[field][stat] * 100 for r in group]
                ax.fill_between(x, convert("q25"), convert("q75"), color=color, alpha=.15)
                ax.plot(x, convert("median"), "o-", color=color, ms=3, label=label)
            ax.set_title(f"{backend.upper()} | {size} views | N={size}")
            axis(ax, levels, "AbsRel (%; lower is better)")
            ax.legend(fontsize=8)
    fig.suptitle(f"{scene}: absolute depth error | Medians and IQR across targets (not confidence intervals)")
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(fig, folder, "depth_errors")

    fig, axes = plt.subplots(len(sizes), len(backends), figsize=(15, 10), squeeze=False, sharex=True)
    for ri, size in enumerate(sizes):
        for ci, backend in enumerate(backends):
            ax = axes[ri, ci]
            for target in representative_ids(data["manifest"]["sets"][str(size)]):
                values = [next(r["utility_raw"] for r in rows if r["backend"] == backend and r["size"] == size and
                               r["target"] == target and r["metric"] == PRIMARY and r["blur_deg"] == b) for b in levels]
                ax.plot(x, [np.nan if v is None else 100 * v for v in values], "o-", ms=3, label=target)
            ax.axhline(0, color="black", lw=.8)
            ax.set_title(f"{backend.upper()} | {size} views")
            axis(ax, levels, "Depth U (pp; + helpful)")
            ax.legend(fontsize=7)
    fig.suptitle(f"{scene}: fixed representative targets = lexicographic first / middle / last")
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(fig, folder, "representative_targets")

    secondary = protocol["evaluation"]["secondary_metrics"]
    fig, axes = plt.subplots(len(sizes), len(secondary), figsize=(18, 10), squeeze=False, sharex=True)
    for ri, size in enumerate(sizes):
        for ci, metric in enumerate(secondary):
            ax = axes[ri, ci]
            for backend in backends:
                group = [index[(backend, size, metric, b)] for b in levels]
                cv = lambda k: [np.nan if r["utility_raw"][k] is None else r["utility_raw"][k] for r in group]
                ax.plot(x, cv("median"), "o-", ms=2, color=COLORS[backend], label=backend.upper())
                ax.fill_between(x, cv("q25"), cv("q75"), alpha=.15, color=COLORS[backend])
            ax.axhline(0, color="black", lw=.7)
            ax.set_title(f"{size} views | {LABELS[metric][0]}", fontsize=10)
            axis(ax, levels, f"U raw ({LABELS[metric][2]})")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{scene}: secondary metrics | Positive = helpful | Target median / IQR, no significance tests")
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(fig, folder, "secondary_metrics")

    descriptors = (("target_mean_overlap_to_context", "Target overlap with context"),
                   ("context_mean_baseline_m", "Context mean camera baseline (m)"),
                   ("context_mean_rotation_deg", "Context mean relative rotation (deg)"),
                   ("context_gt_depth_median_m", "Context median GT depth (m)"))
    fig, axes = plt.subplots(2, len(descriptors), figsize=(18, 8), squeeze=False)
    for ri, field in enumerate(("clean_utility_raw", "last_utility_raw")):
        for ci, (key, label) in enumerate(descriptors):
            ax = axes[ri, ci]
            for backend in backends:
                group = [r for r in contexts if r["backend"] == backend and r[field] is not None]
                for size, marker in zip(sizes, ("o", "s", "^")):
                    subset = [r for r in group if r["size"] == size]
                    ax.scatter([r[key] for r in subset], [r[field] * 100 for r in subset], marker=marker,
                               color=COLORS[backend], alpha=.7, s=28, label=f"{backend} / {size}")
            ax.axhline(0, color="gray", lw=.6)
            ax.set_xlabel(label)
            ax.set_ylabel(("Clean" if ri == 0 else f"{levels[-1]:g} deg") + " U (percentage points)")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"{scene}: exploratory GT context associations | Overlapping contexts are not independent samples")
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(fig, folder, "context_exploration")


def make_cohort_figure(cohort, protocol, folder):
    main = [r for r in cohort if r["stratum"] == "all" and r["metric"] == PRIMARY]
    if not main:
        return
    folder.mkdir(parents=True, exist_ok=True)
    sizes, backends, levels = protocol["sizes"], protocol["backends"], protocol["blur_degrees"]
    x = np.arange(len(levels))
    fig, axes = plt.subplots(len(sizes), len(backends), figsize=(15, 10), squeeze=False, sharex=True)
    index = {(r["backend"], r["size"], r["blur_deg"]): r for r in main}
    for ri, size in enumerate(sizes):
        for ci, backend in enumerate(backends):
            ax = axes[ri, ci]
            group = [index[(backend, size, degree)] for degree in levels]
            harm = lambda field: [r["harmful_scene_fraction"][field] * 100 for r in group]
            ax.fill_between(x, harm("min"), harm("max"), color=COLORS[backend], alpha=.2, label="Observed scene min–max")
            ax.plot(x, harm("mean"), "o-", ms=3, color=COLORS[backend], label="Equal scene mean")
            ax.plot(x, [r["unavailable_scene_fraction"]["mean"] * 100 for r in group], "--", color="gray", label="Unavailable scene mean")
            ax.set_ylim(-2, 102)
            ax.set_title(f"{backend.upper()} | {size} views | scenes {group[0]['n_scenes_used']}/{group[0]['n_scenes_requested']}")
            axis(ax, levels, "% of planned targets")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Confirmed harmful depth comparisons | Equal scene weights | Range across scenes, not confidence intervals\n"
                 "Discrete observed stages; pending scenes are not imputed as zero", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, folder, "cohort_harmful_fraction")
