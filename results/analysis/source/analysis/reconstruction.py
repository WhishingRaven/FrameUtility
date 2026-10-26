"""RGB point-cloud illustrations from saved depths, intrinsics and camera poses.

No model loading, shape fitting, GT surface replacement, or metric modification.
The virtual camera is chosen from GT/input data, independently of either model.
"""
import numpy as np
from PIL import Image, ImageOps

from metrics import sim3
from .plots import plt

RENDER_POLICY = {
    "clean_selection": "largest available common clean context; lexicographic middle reference photo",
    "contrast_selection": "same context and middle target; largest configured nonzero blur",
    "alignment": "one positive Sim(3) from predicted to GT camera centers; no depth/surface fitting",
    "contrast_alignment_ids": "identical common context cameras, excluding target, in all six panels",
    "contrast_display_ids": "identical common context views only; added target affects inference but contributes no displayed points",
    "sampling": "protocol common prepared-image grid, no GT validity mask filtering of model points",
    "color": "RGB of actual input variant after exact backend PIL resize/crop, sampled at native depth pixels",
    "point_diameter_px": 2,
    "max_points_per_panel": 400000,
    "render_width": 900,
    "render_height": 600,
    "horizontal_fov_deg": 70,
    "background_rgb": [18, 24, 32],
    "view_offset_right_up_back_in_reference_median_depth": [.18, .10, .12],
    "occlusion": "perspective projection and nearest-depth z-buffer; no lighting or image synthesis",
}


def backend_image(path, backend, expected_shape):
    """Reproduce only the upstream PIL operations, without importing torch."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB") if backend == "mast3r" else opened.convert("RGB")
        width, height = image.size
        if backend == "vggt":
            resized = (518, round(height * 518 / width / 14) * 14)
            image = image.resize(resized, Image.Resampling.BICUBIC)
            crop_left, crop_top = 0, max(0, (resized[1] - 518) // 2)
            image = image.crop((0, crop_top, resized[0], crop_top + min(resized[1], 518)))
        elif backend == "mast3r":
            resized = tuple(round(v * 512 / max(width, height)) for v in (width, height))
            interpolation = Image.Resampling.LANCZOS if max(width, height) > 512 else Image.Resampling.BICUBIC
            image = image.resize(resized, interpolation)
            cx, cy = resized[0] // 2, resized[1] // 2
            half_width, half_height = (2 * cx) // 16 * 8, (2 * cy) // 16 * 8
            if resized[0] == resized[1]:
                half_height = 3 * half_width / 4
            crop_left, crop_top = cx - half_width, cy - half_height
            image = image.crop((crop_left, crop_top, cx + half_width, cy + half_height))
        else:
            raise ValueError(f"Unsupported reconstruction backend: {backend}")
        pixels = np.asarray(image)
    if pixels.shape[:2] != tuple(expected_shape):
        raise ValueError(f"RGB/native-depth preprocessing shape differs: {backend}/{path}")
    return pixels, {"source_size": [width, height], "resized_size": list(resized),
                    "crop_offset": [int(crop_left), int(crop_top)], "depth_shape": list(expected_shape)}


def sample_native_grid(transform, preparation):
    roi_x, roi_y, roi_width, roi_height = preparation["evaluation_roi"]
    grid_width, grid_height = preparation["evaluation_grid"]
    x = roi_x + (np.arange(grid_width) + .5) * roi_width / grid_width
    y = roi_y + (np.arange(grid_height) + .5) * roi_height / grid_height
    source_w, source_h = transform["source_size"]
    resized_w, resized_h = transform["resized_size"]
    left, top = transform["crop_offset"]
    u = np.floor(x * resized_w / source_w).astype(int) - left
    v = np.floor(y * resized_h / source_h).astype(int) - top
    height, width = transform["depth_shape"]
    if u.min() < 0 or v.min() < 0 or u.max() >= width or v.max() >= height:
        raise ValueError("Shared qualitative grid extends outside model crop")
    return np.meshgrid(u, v)


def unproject(depth, intrinsics, c2w, u, v):
    """Camera optical-axis depth at integer native pixels, matching upstream geometry."""
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(c2w, dtype=np.float64)
    if (intrinsic.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(intrinsic).all()
            or not np.isfinite(pose).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0):
        raise ValueError("Invalid saved intrinsics/camera pose for point-cloud backprojection")
    rays = np.column_stack((u.ravel(), v.ravel(), np.ones(u.size))) @ np.linalg.inv(intrinsic).T
    camera_points = rays * np.asarray(depth, dtype=np.float64).ravel()[:, None]
    return camera_points @ pose[:3, :3].T + pose[:3, 3]


def align_cloud(points, predicted_poses, gt_poses, ids):
    predicted = np.array([predicted_poses[i][:3, 3] for i in ids])
    target = np.array([gt_poses[i][:3, 3] for i in ids])
    scale, rotation, translation = sim3(predicted, target)
    transformed = scale * points @ rotation.T + translation
    residual = scale * predicted @ rotation.T + translation - target
    return transformed, {"camera_ids": list(ids), "scale": float(scale), "rotation": rotation.tolist(),
                         "translation": translation.tolist(),
                         "aligned_camera_center_rmse_m": float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))}


def choose_view(data, reference, policy=RENDER_POLICY):
    frame = next(f for f in data["manifest"]["frames"] if f["id"] == reference)
    pose = np.asarray(frame["c2w"], dtype=np.float64)
    depth, mask = data["gt_depths"][reference], data["gt_masks"][reference]
    prep = data["manifest"]["preparation"]
    roi_x, roi_y, roi_w, roi_h = prep["evaluation_roi"]
    grid_w, grid_h = prep["evaluation_grid"]
    u, v = np.meshgrid(roi_x + (np.arange(grid_w) + .5) * roi_w / grid_w,
                       roi_y + (np.arange(grid_h) + .5) * roi_h / grid_h)
    world = unproject(depth[mask], frame["intrinsics"], pose, u[mask], v[mask])
    focus = np.median(world, axis=0)
    median_depth = float(np.median(depth[mask]))
    right, up, back = policy["view_offset_right_up_back_in_reference_median_depth"]
    eye = pose[:3, 3] + median_depth * (right * pose[:3, 0] - up * pose[:3, 1] - back * pose[:3, 2])
    forward = focus - eye
    forward /= np.linalg.norm(forward)
    view_right = np.cross(forward, -pose[:3, 1])
    view_right /= np.linalg.norm(view_right)
    view_down = np.cross(forward, view_right)
    virtual_pose = np.eye(4)
    virtual_pose[:3, :3] = np.column_stack((view_right, view_down, forward))
    virtual_pose[:3, 3] = eye
    width, height = policy["render_width"], policy["render_height"]
    focal = width / (2 * np.tan(np.deg2rad(policy["horizontal_fov_deg"]) / 2))
    intrinsic = np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1.]])
    return {"reference_image_id": reference, "c2w": virtual_pose.tolist(), "intrinsics": intrinsic.tolist(),
            "width": width, "height": height, "horizontal_fov_deg": policy["horizontal_fov_deg"],
            "vertical_fov_deg": float(np.rad2deg(2 * np.arctan(height / (2 * focal)))),
            "near_m": .01 * median_depth, "far_m": 10 * median_depth,
            "point_diameter_px": policy["point_diameter_px"], "focus_gt_world_m": focus.tolist(),
            "selection_uses_model_outputs": False}


def render_cloud(points, colors, view, background=(18, 24, 32)):
    """Deterministic perspective point splats with a common nearest-depth z-buffer."""
    width, height = view["width"], view["height"]
    pose, intrinsic = np.asarray(view["c2w"]), np.asarray(view["intrinsics"])
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    front = np.isfinite(camera).all(axis=1) & (camera[:, 2] > view["near_m"]) & (camera[:, 2] < view["far_m"])
    camera, colors = camera[front], colors[front]
    projected = camera @ intrinsic.T
    pixels = np.rint(projected[:, :2] / projected[:, 2:]).astype(int)
    diameter = view["point_diameter_px"]
    flat_ids, z_values, color_values = [], [], []
    for dy in range(-(diameter // 2), diameter - diameter // 2):
        for dx in range(-(diameter // 2), diameter - diameter // 2):
            x, y = pixels[:, 0] + dx, pixels[:, 1] + dy
            keep = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            flat_ids.append(y[keep] * width + x[keep])
            z_values.append(camera[keep, 2])
            color_values.append(colors[keep])
    canvas = np.full((height * width, 3), background, dtype=np.uint8)
    ids, depths = np.concatenate(flat_ids), np.concatenate(z_values)
    rgb = np.concatenate(color_values)
    if len(ids):
        order = np.lexsort((np.arange(len(ids)), depths, ids))
        ordered = ids[order]
        nearest = order[np.r_[True, ordered[1:] != ordered[:-1]]]
        canvas[ids[nearest]] = rgb[nearest]
        occupied = len(nearest)
    else:
        occupied = 0
    return canvas.reshape(height, width, 3), {"rendered_pixels": occupied, "rendered_pixel_fraction": occupied / (width * height)}


def load_cloud(root, prediction_path, backend, data, protocol, sources, display_ids, alignment_ids,
               target=None, blur=0):
    sources.check(prediction_path)
    with np.load(prediction_path, allow_pickle=False) as stored:
        prediction = {k: stored[k] for k in ("ids", "c2w", "intrinsics", "depth", "depth_common_grid")}
    frames = {f["id"]: f for f in data["manifest"]["frames"]}
    predicted_poses = {str(i): np.asarray(p) for i, p in zip(prediction["ids"], prediction["c2w"])}
    gt_poses = {i: np.asarray(frames[i]["c2w"]) for i in alignment_ids}
    if not set(display_ids) <= predicted_poses.keys() or not set(alignment_ids) <= predicted_poses.keys():
        raise ValueError(f"Missing common cameras in qualitative prediction: {prediction_path}")
    chunks, colors, invalid = [], [], 0
    for image_id in display_ids:
        index = list(prediction["ids"]).index(image_id)
        degree = blur if image_id == target else 0
        variant = frames[image_id]["variants"][f"{degree:g}"]
        image_path = root / variant["path"]
        sources.check(image_path, variant["sha256"])
        depth = prediction["depth"][index]
        rgb, transform = backend_image(image_path, backend, depth.shape)
        u, v = sample_native_grid(transform, protocol["preparation"])
        selected_depth = depth[v, u]
        if not np.array_equal(selected_depth, prediction["depth_common_grid"][index], equal_nan=True):
            raise ValueError(f"Qualitative native sampling disagrees with saved common grid: {prediction_path}/{image_id}")
        valid = np.isfinite(selected_depth) & (selected_depth > 0)
        invalid += int((~valid).sum())
        world = unproject(selected_depth[valid], prediction["intrinsics"][index], prediction["c2w"][index], u[valid], v[valid])
        chunks.append(world)
        colors.append(rgb[v[valid], u[valid]])
    points, colors = np.concatenate(chunks), np.concatenate(colors)
    points, alignment = align_cloud(points, predicted_poses, gt_poses, alignment_ids)
    generated = len(points)
    limit = RENDER_POLICY["max_points_per_panel"]
    if generated > limit:
        indices = np.linspace(0, generated - 1, limit, dtype=np.int64)
        points, colors = points[indices], colors[indices]
    return points, colors, {"prediction_file": str(prediction_path.relative_to(root)),
                           "prediction_sha256": sources.check(prediction_path), "display_image_ids": list(display_ids),
                           "valid_generated_points": generated, "displayed_points": len(points),
                           "excluded_nonfinite_or_nonpositive_depth_samples": invalid,
                           "native_image_transform": transform, "alignment": alignment}


def save_comparison(panels, titles, path, caption):
    rows, cols = len(panels), len(panels[0])
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.6 * rows), squeeze=False)
    for row in range(rows):
        for col in range(cols):
            axes[row, col].imshow(panels[row][col])
            axes[row, col].set_title(titles[row][col], fontsize=13)
            axes[row, col].axis("off")
    fig.text(.5, .015, caption, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def make_reconstruction_figures(root, execution_folder, data, protocol, folder, sources):
    """Apply the same deterministic illustration rule to every analyzed scene."""
    folder.mkdir(parents=True, exist_ok=True)
    backends = ("vggt", "mast3r")
    size = next((m for m in sorted(protocol["sizes"], reverse=True) if all(
        (execution_folder / b / "predictions" / f"m{m}_clean.npz").exists() for b in backends)), None)
    if size is None:
        return {"status": "unavailable", "reason": "Both saved clean reconstructions are required; no inference was run"}
    ids = data["manifest"]["sets"][str(size)]
    reference = sorted(ids)[len(ids) // 2]
    view = choose_view(data, reference)
    info = {"status": "generated", "scene": data["info"]["scene"], "size": size,
            "reference_image_id": reference, "policy": RENDER_POLICY, "shared_virtual_camera": view,
            "clean_panels": {}, "contrast": {"status": "unavailable"}, "inference_executed": False}
    reference_frame = next(f for f in data["manifest"]["frames"] if f["id"] == reference)
    rgb_path = root / reference_frame["variants"]["0"]["path"]
    sources.check(rgb_path, reference_frame["variants"]["0"]["sha256"])
    info["original_rgb"] = {"path": str(rgb_path.relative_to(root)), "sha256": sources.check(rgb_path)}
    with Image.open(rgb_path) as image:
        original = np.asarray(image.convert("RGB").resize((view["width"], view["height"]), Image.Resampling.LANCZOS))
    panels = [original]
    for backend in backends:
        path = execution_folder / backend / "predictions" / f"m{size}_clean.npz"
        points, colors, details = load_cloud(root, path, backend, data, protocol, sources, ids, ids)
        rendered, coverage = render_cloud(points, colors, view, RENDER_POLICY["background_rgb"])
        panels.append(rendered)
        info["clean_panels"][backend] = {**details, **coverage, "virtual_camera": "shared_virtual_camera"}
    save_comparison([panels], [[f"Original RGB | {reference}", f"VGGT reconstruction | {size} clean views",
                               f"MASt3R reconstruction | {size} clean views"]], folder / "reconstruction_comparison",
                    f"RGB-colored 3D points | Both models: same virtual view, {view['horizontal_fov_deg']:g} deg horizontal FOV, "
                    f"{view['point_diameter_px']} px points | Camera-center Sim(3) alignment")
    target, blur = reference, max(protocol["blur_degrees"])
    common = [i for i in ids if i != target]
    keys = (f"m{size}_drop_{target}", f"m{size}_clean", f"m{size}_add_{target}_b{blur:g}")
    if blur > 0 and all((execution_folder / b / "predictions" / (key + ".npz")).exists() for b in backends for key in keys):
        condition_panels, titles = [], []
        info["contrast"] = {"status": "generated", "target": target, "blur_deg": blur,
                            "size": size, "common_image_ids": common, "panels": {}}
        for backend in backends:
            row = []
            for role, key in zip(("drop", "clean", "blurred"), keys):
                points, colors, details = load_cloud(root, execution_folder / backend / "predictions" / (key + ".npz"),
                    backend, data, protocol, sources, common, common, target, blur if role == "blurred" else 0)
                rendered, coverage = render_cloud(points, colors, view, RENDER_POLICY["background_rgb"])
                row.append(rendered)
                info["contrast"]["panels"][f"{backend}_{role}"] = {**details, **coverage, "virtual_camera": "shared_virtual_camera"}
            condition_panels.append(row)
            titles.append([f"{backend.upper()} | DROP", f"{backend.upper()} | ADD clean", f"{backend.upper()} | ADD blurred ({blur:g} deg)"])
        save_comparison(condition_panels, titles, folder / "reconstruction_conditions",
                        f"Target {target} | Only the same {len(common)} context views supply displayed points | Same camera, FOV and point size in all six panels")
    return info
