"""Versioned follow-up scoring; legacy strict metrics remain unchanged."""
import numpy as np

from metrics import evaluate_available

POLICY = {
    "version": "fixed_mask_signed_depth_v1",
    "primary_metric": "geometry_depth_absrel",
    "secondary_metrics": ["geometry_depth_rmse_m", "pose_ate_normalized",
                          "pose_pair_translation_relative", "pose_pair_rotation_deg"],
    "mask": "Identical finite positive GT depth pixels in the common context; target excluded",
    "minimum_reference_pixels_per_frame": 100,
    "scale": "One median(GT/prediction) over finite positive predictions in the entire common context",
    "minimum_positive_predictions_for_scale": 100,
    "scoring": "Score every fixed-mask pixel, including finite zero and negative predictions; no clipping or removal",
    "nonfinite": "Any nonfinite prediction on the fixed mask fails geometry only",
    "diagnostics": "Report nonpositive and nonfinite counts/fractions separately, plus legacy strict scores/failures",
    "absrel_effect_floor": 0.0005,
    "effect_floor_note": "Prospectively fixed for this pilot-selected follow-up, informed by the exploratory run; not a significance threshold",
    "replicates": 3,
    "seed_per_reconstruction": 2026,
    "stable_raw_transition": "Every DROP-minus-clean combination is positive and every DROP-minus-ADD combination is negative",
    "stable_effect_transition": "Every DROP-minus-clean combination exceeds +0.0005 and every DROP-minus-ADD combination is below -0.0005",
    "uncertainty": "Observed repeat ranges, no scene-level confidence intervals or p-values; same-seed numerical repeatability",
}


def fixed_mask_geometry(prediction, common_ids, gt_depths):
    """Return scores, diagnostics and an explicit error, including failed inputs."""
    diagnostics = {"evaluation_pixels": 0, "nonpositive_pixels": 0,
                   "nonfinite_pixels": 0, "frames": []}
    pd, gd = [], []
    try:
        ids = list(prediction["ids"])
        for frame in common_ids:
            depth = prediction["depth"][ids.index(frame)].astype(np.float64)
            ref = gt_depths[frame]
            if depth.shape != ref.shape:
                raise ValueError("Prediction/reference depth shapes differ")
            mask = (ref > 0) & np.isfinite(ref)
            pixels = int(mask.sum())
            if pixels < POLICY["minimum_reference_pixels_per_frame"]:
                raise ValueError("Insufficient laser reference coverage")
            values = depth[mask]
            finite = np.isfinite(values)
            nonpositive = int(np.sum(finite & (values <= 0)))
            nonfinite = int(np.sum(~finite))
            diagnostics["evaluation_pixels"] += pixels
            diagnostics["nonpositive_pixels"] += nonpositive
            diagnostics["nonfinite_pixels"] += nonfinite
            diagnostics["frames"].append({"frame": frame, "evaluation_pixels": pixels,
                                           "nonpositive_pixels": nonpositive, "nonfinite_pixels": nonfinite})
            pd.append(values)
            gd.append(ref[mask])
        count = diagnostics["evaluation_pixels"]
        if not count:
            raise ValueError("Empty evaluation context")
        for name in ("nonpositive", "nonfinite"):
            diagnostics[name + "_fraction"] = diagnostics[name + "_pixels"] / count
        if diagnostics["nonfinite_pixels"]:
            raise ValueError("Nonfinite depth on fixed evaluation mask")
        pd, gd = np.concatenate(pd), np.concatenate(gd)
        positive = pd > 0
        if positive.sum() < POLICY["minimum_positive_predictions_for_scale"]:
            raise ValueError("Insufficient positive depths for scale estimation")
        scale = np.median(gd[positive] / pd[positive])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid estimated depth scale")
        difference = scale * pd - gd
        values = {"geometry_depth_absrel": float(np.mean(np.abs(difference) / gd)),
                  "geometry_depth_rmse_m": float(np.sqrt(np.mean(difference ** 2))),
                  "alignment_depth_scale": float(scale), "evaluation_pixels": int(len(gd))}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Nonfinite geometry score")
        return values, diagnostics, None
    except (ValueError, KeyError) as error:
        return {}, diagnostics, str(error)


def evaluate_followup(prediction, common_ids, gt_poses, gt_depths):
    strict_values, strict_errors = evaluate_available(prediction, common_ids, gt_poses, gt_depths)
    values = {k: v for k, v in strict_values.items() if k.startswith(("pose_", "alignment_pose"))}
    errors = {k: v for k, v in strict_errors.items() if k == "pose"}
    if prediction is None:
        geometry, diagnostics, error = {}, {}, "Reconstruction failed"
    else:
        geometry, diagnostics, error = fixed_mask_geometry(prediction, common_ids, gt_depths)
    values.update(geometry)
    if error:
        errors["geometry"] = error
    return {"metrics": values, "errors": errors, "depth_diagnostics": diagnostics,
            "strict_metrics": strict_values, "strict_errors": strict_errors}
