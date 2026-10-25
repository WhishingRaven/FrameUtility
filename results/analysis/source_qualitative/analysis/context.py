"""Exploratory descriptors from the frozen GT/input snapshot, never predictions."""
from itertools import combinations
import numpy as np
from scipy.spatial.transform import Rotation

from .core import PRIMARY


def context_descriptors(scene_data, protocol):
    manifest = scene_data["manifest"]
    frames = {f["id"]: f for f in manifest["frames"]}
    overlaps = {frozenset((p["a"], p["b"])): p["mean_overlap"] for p in scene_data["selection"]["pair_overlaps"]}
    scores = {(r["backend"], r["size"], r["target"], r["blur_deg"]): r for r in scene_data["rows"] if r["metric"] == PRIMARY}
    output = []
    for size in protocol["sizes"]:
        ids = manifest["sets"][str(size)]
        for target in ids:
            common = [i for i in ids if i != target]
            poses = {i: np.asarray(frames[i]["c2w"]) for i in ids}
            pairs = list(combinations(common, 2))
            depths = np.concatenate([scene_data["gt_depths"][i][scene_data["gt_masks"][i]] for i in common])
            record = {"scene": manifest["scene"], "size": size, "target": target,
                      "context_mean_pair_overlap": float(np.mean([overlaps[frozenset(p)] for p in pairs])),
                      "target_mean_overlap_to_context": float(np.mean([overlaps[frozenset((target, i))] for i in common])),
                      "context_mean_baseline_m": float(np.mean([np.linalg.norm(poses[a][:3, 3] - poses[b][:3, 3]) for a, b in pairs])),
                      "context_mean_rotation_deg": float(np.mean([np.rad2deg(Rotation.from_matrix(poses[a][:3, :3].T @ poses[b][:3, :3]).magnitude()) for a, b in pairs])),
                      "context_gt_depth_p10_m": float(np.quantile(depths, .1)),
                      "context_gt_depth_median_m": float(np.median(depths)),
                      "context_gt_depth_p90_m": float(np.quantile(depths, .9)),
                      "evaluation_pixels": len(depths),
                      "target_4deg_center_motion_prepared_px": float(2 * frames[target]["intrinsics"][0][0] * np.tan(np.deg2rad(4) / 2)),
                      "scope": "exploratory_GT_only_descriptors_from_pre_inference_snapshot_derived_during_analysis"}
            for backend in protocol["backends"]:
                output.append({**record, "backend": backend,
                               "clean_utility_raw": scores[(backend, size, target, 0)]["utility_raw"],
                               "last_utility_raw": scores[(backend, size, target, protocol["blur_degrees"][-1])]["utility_raw"]})
    return output
