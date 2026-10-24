"""Evaluate ADD and DROP on exactly the same clean context cameras/pixels."""
import numpy as np
from scipy.spatial.transform import Rotation


def sim3(source, target):
    """Umeyama alignment: target = scale * source @ rotation.T + translation."""
    source, target = np.asarray(source, np.float64), np.asarray(target, np.float64)
    x, y = source-source.mean(0), target-target.mean(0)
    if np.linalg.matrix_rank(x, tol=1e-8) < 2 or np.linalg.matrix_rank(y, tol=1e-8) < 2:
        raise ValueError("Degenerate camera centers for Sim(3) alignment")
    U, D, Vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(U @ Vt)
    R = (U * sign) @ Vt
    scale = np.sum(D * sign) / np.mean(np.sum(x*x, axis=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid estimated scale")
    return scale, R, target.mean(0) - scale * R @ source.mean(0)


def evaluate_pose(prediction, common_ids, gt_poses):
    ids = list(prediction["ids"])
    ix = [ids.index(i) for i in common_ids]
    pred = prediction["c2w"][ix].astype(np.float64)
    gt = np.array([gt_poses[i] for i in common_ids], np.float64)
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite pose prediction")
    pc, gc = pred[:, :3, 3], gt[:, :3, 3]
    scale, rot, shift = sim3(pc, gc)
    aligned_centers = scale * pc @ rot.T + shift
    ate = np.sqrt(np.mean(np.sum((aligned_centers-gc)**2, axis=1)))
    scene_radius = np.sqrt(np.mean(np.sum((gc-gc.mean(0))**2, axis=1)))
    rpe_t, rpe_r = [], []
    for i in range(len(ix)):
        for j in range(i+1, len(ix)):
            pr = pred[i, :3, :3].T @ pred[j, :3, :3]
            gr = gt[i, :3, :3].T @ gt[j, :3, :3]
            rpe_r.append(np.rad2deg(Rotation.from_matrix(pr @ gr.T).magnitude()))
            pt = pred[i, :3, :3].T @ (pc[j]-pc[i]) * scale
            gt_t = gt[i, :3, :3].T @ (gc[j]-gc[i])
            baseline = np.linalg.norm(gt_t)
            if baseline > 1e-8:
                rpe_t.append(np.linalg.norm(pt-gt_t) / baseline)
    return {"pose_ate_m": float(ate), "pose_ate_normalized": float(ate/scene_radius),
            "pose_pair_translation_relative": float(np.mean(rpe_t)),
            "pose_pair_rotation_deg": float(np.mean(rpe_r)), "alignment_pose_scale": float(scale)}


def evaluate_geometry(prediction, common_ids, gt_depths):
    ids = list(prediction["ids"])
    ix = [ids.index(i) for i in common_ids]
    # One scale for the entire common context, never a separate fit per image.
    pd, gd = [], []
    for idx, frame in zip(ix, common_ids):
        depth = prediction["depth"][idx].astype(np.float64)
        ref = gt_depths[frame]
        assert depth.shape == ref.shape
        mask = (ref > 0) & np.isfinite(ref)
        if np.count_nonzero(mask) < 100:
            raise ValueError("Insufficient laser reference coverage")
        if not np.all(np.isfinite(depth[mask]) & (depth[mask] > 0)):
            raise ValueError("Invalid predicted depth on fixed evaluation mask")
        pd.append(depth[mask])
        gd.append(ref[mask])
    pd, gd = np.concatenate(pd), np.concatenate(gd)
    depth_scale = np.median(gd/pd)
    difference = pd*depth_scale-gd
    return {"geometry_depth_absrel": float(np.mean(np.abs(difference)/gd)),
            "geometry_depth_rmse_m": float(np.sqrt(np.mean(difference**2))),
            "alignment_depth_scale": float(depth_scale),
            "evaluation_pixels": int(len(gd))}


def evaluate(prediction, common_ids, gt_poses, gt_depths):
    """Strict combined evaluation; any invalid family raises as before."""
    return {**evaluate_pose(prediction, common_ids, gt_poses),
            **evaluate_geometry(prediction, common_ids, gt_depths)}


def evaluate_available(prediction, common_ids, gt_poses, gt_depths):
    """Keep valid metrics while recording invalid pose/geometry as separate outcomes."""
    values, errors = {}, {}
    if prediction is None:
        return values, {"pose": "Reconstruction failed", "geometry": "Reconstruction failed"}
    for family, evaluator, reference in [("pose", evaluate_pose, gt_poses),
                                          ("geometry", evaluate_geometry, gt_depths)]:
        try:
            values.update(evaluator(prediction, common_ids, reference))
        except ValueError as error:
            errors[family] = str(error)
    return values, errors


METRICS = ["pose_ate_normalized", "pose_pair_translation_relative", "pose_pair_rotation_deg",
           "geometry_depth_absrel", "geometry_depth_rmse_m"]


def condition_counts(rows, sizes=None):
    cases = {}
    for row in rows:
        if sizes is None or row["size"] in sizes:
            cases.setdefault((row["backend"], row["size"], row["target"], row["blur_deg"]), {})[row["metric"]] = row["status"]
    counts = {"attempted": 0, "ok": 0, "partial": 0, "failed": 0}
    for statuses in cases.values():
        if not all(metric in statuses for metric in METRICS):
            continue
        counts["attempted"] += 1
        successes = sum(statuses[metric] == "ok" for metric in METRICS)
        counts["ok" if successes == len(METRICS) else "partial" if successes else "failed"] += 1
    return counts


def utility(drop, add, epsilon=1e-10):
    raw = drop-add
    return raw, (raw/drop if abs(drop) > epsilon else None)
