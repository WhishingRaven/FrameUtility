"""Select ETH3D study views from the full catalog using official depth GT."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/eth3d/high_res_multi_view"
ARCHIVE_ROOT = DATA_ROOT / "multi_view_training_dslr_depth_archives"
OUTPUT_PATH = ROOT / "data/experiments/eth3d_multi_view/official_selection.json"
SCENES = (
    "courtyard", "delivery_area", "electro", "facade", "kicker", "meadow",
    "office", "playground", "relief", "relief_2", "terrace", "terrains",
)
TARGET_SIZES = (15, 10, 5)
OVERLAP_THRESHOLD = 0.10
MIN_VALID_GT_SAMPLES = 500
OUTPUT_WIDTH, OUTPUT_HEIGHT = 1008, 672
OVERSCAN_WIDTH = 1120


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cameras(path: Path) -> dict[int, dict]:
    cameras = {}
    rows = [line.split() for line in path.read_text().splitlines()
            if line and not line.startswith("#")]
    for row in rows:
        camera_id, model_name = int(row[0]), row[1]
        width, height = map(int, row[2:4])
        params = list(map(float, row[4:]))
        camera = pycolmap.Camera.create_from_model_name(
            camera_id, model_name, params[0], width, height
        )
        if not camera.set_params_from_string(",".join(format(value, ".17g") for value in params)):
            raise ValueError(f"Could not load COLMAP camera parameters from {path}: {camera_id}")
        cameras[camera_id] = {
            "camera": camera,
            "model": model_name,
            "size": (width, height),
            "params": params,
        }
    return cameras


def read_images(path: Path) -> dict[str, dict]:
    rows = path.read_text().splitlines()
    images = {}
    index = 0
    while index < len(rows):
        line = rows[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"Malformed COLMAP image row in {path}: {line}")
        q = np.asarray(list(map(float, fields[1:5])), dtype=np.float64)
        rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        translation = np.asarray(list(map(float, fields[5:8])), dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rotation.T
        c2w[:3, 3] = -rotation.T @ translation
        filename = Path(fields[9]).name
        if filename in images:
            raise ValueError(f"Duplicate calibrated filename in {path}: {filename}")
        images[filename] = {
            "image_id": int(fields[0]),
            "camera_id": int(fields[8]),
            "c2w": c2w,
        }
        index += 1  # the following COLMAP 2D-observations row
    return images


def read_scene_calibrations(scene: str) -> tuple[list[dict], dict]:
    undist = DATA_ROOT / "multi_view_training_dslr_undistorted" / scene / "dslr_calibration_undistorted"
    distorted = DATA_ROOT / "multi_view_training_dslr_jpg" / scene / "dslr_calibration_jpg"
    undist_images_dir = (DATA_ROOT / "multi_view_training_dslr_undistorted" / scene /
                         "images/dslr_images_undistorted")
    distorted_images_dir = (DATA_ROOT / "multi_view_training_dslr_jpg" / scene /
                            "images/dslr_images")
    undist_cameras = read_cameras(undist / "cameras.txt")
    raw_cameras = read_cameras(distorted / "cameras.txt")
    undist_images = read_images(undist / "images.txt")
    raw_images = read_images(distorted / "images.txt")
    if set(undist_images) != set(raw_images):
        raise ValueError(f"{scene}: distorted and undistorted calibration image lists differ")

    frames = []
    for filename in sorted(undist_images):
        for image_dir in (undist_images_dir, distorted_images_dir):
            if not (image_dir / filename).is_file():
                raise FileNotFoundError(f"{scene}: calibrated image is missing: {image_dir / filename}")
        un = undist_images[filename]
        raw = raw_images[filename]
        if un["camera_id"] not in undist_cameras or raw["camera_id"] not in raw_cameras:
            raise ValueError(f"{scene}/{filename}: unknown CAMERA_ID")
        rotation_error = Rotation.from_matrix(
            un["c2w"][:3, :3].T @ raw["c2w"][:3, :3]
        ).magnitude()
        center_error = np.linalg.norm(un["c2w"][:3, 3] - raw["c2w"][:3, 3])
        if rotation_error > 1e-4 or center_error > 1e-4:
            raise ValueError(f"{scene}/{filename}: distorted and undistorted poses disagree")

        camera = undist_cameras[un["camera_id"]]
        width, height = camera["size"]
        scaled_height = round(height * OVERSCAN_WIDTH / width)
        sx, sy = OVERSCAN_WIDTH / width, scaled_height / height
        K = np.array([
            [camera["params"][0] * sx, 0, camera["params"][2] * sx],
            [0, camera["params"][1] * sy, camera["params"][3] * sy],
            [0, 0, 1],
        ], dtype=np.float64)
        left = (OVERSCAN_WIDTH - OUTPUT_WIDTH) // 2
        top = (scaled_height - OUTPUT_HEIGHT) // 2
        K[0, 2] -= left
        K[1, 2] -= top
        frames.append({
            "id": Path(filename).stem,
            "filename": filename,
            "undistorted_camera_id": un["camera_id"],
            "distorted_camera_id": raw["camera_id"],
            "c2w": un["c2w"],
            "Kcrop": K,
            "distorted_camera": raw_cameras[raw["camera_id"]]["camera"],
            "distorted_size": raw_cameras[raw["camera_id"]]["size"],
        })
    metadata = {
        "undistorted_cameras_sha256": sha256(undist / "cameras.txt"),
        "undistorted_images_sha256": sha256(undist / "images.txt"),
        "distorted_cameras_sha256": sha256(distorted / "cameras.txt"),
        "distorted_images_sha256": sha256(distorted / "images.txt"),
    }
    return frames, metadata


def evaluation_grid() -> tuple[np.ndarray, np.ndarray]:
    x = np.tile(4 * np.arange(252, dtype=np.float64) + 2, 160)
    y = np.repeat(16 + 4 * np.arange(160, dtype=np.float64) + 2, 252)
    return x, y


def get_depth_map(path: Path, size: tuple[int, int]) -> np.memmap:
    width, height = size
    expected_bytes = width * height * 4
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Unexpected depth size for {path}: {path.stat().st_size} != {expected_bytes}")
    return np.memmap(path, dtype="<f4", mode="r", shape=(height, width))


def sample_distorted_depth(frame: dict, depth: np.ndarray,
                           normalized_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_points = np.column_stack((normalized_xy, np.ones(len(normalized_xy), dtype=np.float64)))
    projected = np.asarray(frame["distorted_camera"].img_from_cam(camera_points), dtype=np.float64)
    width, height = frame["distorted_size"]
    finite = np.isfinite(projected).all(axis=1)
    uv = np.zeros((len(projected), 2), dtype=np.int64)
    uv[finite] = np.floor(projected[finite] + 0.5).astype(np.int64)
    inside = finite & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    values = np.full(len(projected), np.inf, dtype=np.float32)
    values[inside] = depth[uv[inside, 1], uv[inside, 0]]
    valid = inside & np.isfinite(values) & (values > 0)
    return values, valid


def prepare_official_depth_samples(frames: list[dict], extraction_root: Path) -> list[dict]:
    grid_x, grid_y = evaluation_grid()
    prepared = []
    for frame in frames:
        width, height = frame["distorted_size"]
        depth_path = extraction_root / "ground_truth_depth/dslr_images" / frame["filename"]
        depth = get_depth_map(depth_path, (width, height))
        K = frame["Kcrop"]
        normalized_xy = np.column_stack(((grid_x - K[0, 2]) / K[0, 0],
                                         (grid_y - K[1, 2]) / K[1, 1]))
        values, valid = sample_distorted_depth(frame, depth, normalized_xy)
        item = dict(frame)
        item.update({
            "grid_x": grid_x,
            "grid_y": grid_y,
            "normalized_xy": normalized_xy,
            "depth_samples": values,
            "valid_samples": valid,
            "valid_sample_count": int(valid.sum()),
            "valid_grid_mask_sha256": hashlib.sha256(np.packbits(valid).tobytes()).hexdigest(),
            "gt_depth_file": depth_path.name,
        })
        prepared.append(item)
        del depth
    return prepared


def directed_overlap(a: dict, b: dict) -> float:
    valid = a["valid_samples"]
    denominator = int(valid.sum())
    if denominator == 0:
        return 0.0
    z = a["depth_samples"][valid].astype(np.float64)
    xy = a["normalized_xy"][valid]
    points_a = np.column_stack((xy[:, 0] * z, xy[:, 1] * z, z))
    Ta, Tb = a["c2w"], b["c2w"]
    world = points_a @ Ta[:3, :3].T + Ta[:3, 3]
    points_b = (world - Tb[:3, 3]) @ Tb[:3, :3]
    front = points_b[:, 2] > 0
    points_b = points_b[front]
    if not len(points_b):
        return 0.0

    K = b["Kcrop"]
    u_eval = K[0, 0] * points_b[:, 0] / points_b[:, 2] + K[0, 2]
    v_eval = K[1, 1] * points_b[:, 1] / points_b[:, 2] + K[1, 2]
    in_eval = ((u_eval >= 0) & (u_eval < OUTPUT_WIDTH) &
               (v_eval >= 16) & (v_eval < OUTPUT_HEIGHT - 16))
    points_b = points_b[in_eval]
    if not len(points_b):
        return 0.0

    normalized_b = points_b[:, :2] / points_b[:, 2:3]
    ref, ref_valid = sample_distorted_depth(
        b, b["depth_map"], normalized_b
    )
    z_proj = points_b[:, 2].astype(np.float32)
    visible = ref_valid & (np.abs(ref - z_proj) <= np.maximum(.05, .02 * z_proj))
    return float(visible.sum() / denominator)


def connected(ids: list[str], graph: dict[str, set[str]]) -> bool:
    if len(ids) <= 1:
        return True
    allowed = set(ids)
    seen = set()
    frontier = [min(allowed)]
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(graph[current] & allowed - seen)
    return seen == allowed


def robustly_connected(ids: list[str], graph: dict[str, set[str]]) -> bool:
    return len(ids) >= 3 and all(connected([other for other in ids if other != removed], graph)
                                 for removed in ids)


def biconnected_components(graph: dict[str, set[str]]) -> list[list[str]]:
    discovery, low, parent = {}, {}, {}
    edge_stack = []
    components = []
    tick = 0

    def dfs(node: str) -> None:
        nonlocal tick
        discovery[node] = low[node] = tick
        tick += 1
        for neighbor in sorted(graph[node]):
            if neighbor not in discovery:
                parent[neighbor] = node
                edge_stack.append((node, neighbor))
                dfs(neighbor)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] >= discovery[node]:
                    block = set()
                    while edge_stack:
                        edge = edge_stack.pop()
                        block.update(edge)
                        if edge == (node, neighbor):
                            break
                    components.append(sorted(block))
            elif parent.get(node) != neighbor and discovery[neighbor] < discovery[node]:
                edge_stack.append((node, neighbor))
                low[node] = min(low[node], discovery[neighbor])

    for node in sorted(graph):
        if node not in discovery:
            dfs(node)
            if edge_stack:
                block = set()
                while edge_stack:
                    block.update(edge_stack.pop())
                components.append(sorted(block))
    return components


def rank2_for_drop_contexts(ids: list[str], frame_by_id: dict[str, dict]) -> list[dict]:
    failures = []
    for removed in ids:
        remaining = [frame_by_id[view_id]["c2w"][:3, 3] for view_id in ids if view_id != removed]
        centers = np.asarray(remaining, dtype=np.float64)
        rank = int(np.linalg.matrix_rank(centers - centers.mean(axis=0)))
        if rank < 2:
            failures.append({"dropped_view": removed, "rank": rank})
    return failures


def reduce_to_target(start: list[str], target: int, graph: dict[str, set[str]],
                     overlaps: dict[frozenset, float]) -> tuple[list[str] | None, list[dict]]:
    current = sorted(start)
    removals = []
    while len(current) > target:
        candidates = []
        for removed in current:
            kept = [view_id for view_id in current if view_id != removed]
            if robustly_connected(kept, graph):
                average = float(np.mean([overlaps[frozenset((removed, other))] for other in kept]))
                candidates.append((average, removed, kept))
        if not candidates:
            return None, removals
        average, removed, current = min(candidates, key=lambda item: (item[0], item[1]))
        removals.append({"removed": removed, "mean_overlap_to_remaining": average})
    return current, removals


def select_scene(scene: str, archive: Path, temp_root: Path) -> dict:
    frames, calibration_hashes = read_scene_calibrations(scene)
    archive_hash = sha256(archive)
    with tempfile.TemporaryDirectory(prefix=f"eth3d-{scene}-", dir=temp_root) as tmp:
        extraction_root = Path(tmp)
        subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(extraction_root)], check=True)
        scene_root = extraction_root / scene
        if not scene_root.is_dir():
            raise FileNotFoundError(f"Archive {archive} does not contain {scene}/")

        prepared = prepare_official_depth_samples(frames, scene_root)
        eligible = [frame for frame in prepared if frame["valid_sample_count"] >= MIN_VALID_GT_SAMPLES]
        excluded = [
            {"image_id": frame["id"], "valid_gt_samples": frame["valid_sample_count"],
             "reason": f"fewer_than_{MIN_VALID_GT_SAMPLES}_valid_common_grid_gt_samples"}
            for frame in prepared if frame["valid_sample_count"] < MIN_VALID_GT_SAMPLES
        ]
        if len(eligible) < 15:
            return {
                "scene": scene, "status": "fail",
                "reason": "fewer_than_15_photos_pass_official_gt_validity_check",
                "calibrated_photo_count": len(frames), "eligible_photo_count": len(eligible),
                "excluded_photos": excluded, "calibration_hashes": calibration_hashes,
                "depth_archive": archive.name, "depth_archive_sha256": archive_hash,
                "photo_validity": [
                    {"image_id": frame["id"], "valid_gt_samples": frame["valid_sample_count"],
                     "valid_grid_mask_sha256": frame["valid_grid_mask_sha256"]}
                    for frame in prepared
                ],
            }

        for frame in prepared:
            width, height = frame["distorted_size"]
            depth_path = scene_root / "ground_truth_depth/dslr_images" / frame["filename"]
            frame["depth_map"] = get_depth_map(depth_path, (width, height))

        eligible_ids = [frame["id"] for frame in eligible]
        graph = {view_id: set() for view_id in eligible_ids}
        overlaps = {}
        pair_rows = []
        by_id = {frame["id"]: frame for frame in eligible}
        for index, a in enumerate(eligible):
            for b in eligible[index + 1:]:
                forward = directed_overlap(a, b)
                reverse = directed_overlap(b, a)
                overlap = .5 * (forward + reverse)
                key = frozenset((a["id"], b["id"]))
                overlaps[key] = overlap
                if overlap >= OVERLAP_THRESHOLD:
                    graph[a["id"]].add(b["id"])
                    graph[b["id"]].add(a["id"])
                pair_rows.append({"a": a["id"], "b": b["id"],
                                  "overlap_a_to_b": forward, "overlap_b_to_a": reverse,
                                  "mean_overlap": overlap})

        components = biconnected_components(graph)
        components = sorted(components, key=lambda component: (-len(component), tuple(component)))
        largest_size = len(components[0]) if components else 0
        if largest_size < 15:
            return {
                "scene": scene, "status": "fail",
                "reason": "largest_official_gt_biconnected_component_has_fewer_than_15_photos",
                "calibrated_photo_count": len(frames), "eligible_photo_count": len(eligible),
                "excluded_photos": excluded,
                "graph_edge_count": sum(map(len, graph.values())) // 2,
                "biconnected_component_sizes": [len(component) for component in components],
                "calibration_hashes": calibration_hashes,
                "depth_archive": archive.name, "depth_archive_sha256": archive_hash,
                "photo_validity": [
                    {"image_id": frame["id"], "valid_gt_samples": frame["valid_sample_count"],
                     "valid_grid_mask_sha256": frame["valid_grid_mask_sha256"]}
                    for frame in prepared
                ],
            }

        chosen = components[0]
        sets = {}
        removal_orders = {}
        current = chosen
        errors = []
        for target in TARGET_SIZES:
            selected, removals = reduce_to_target(current, target, graph, overlaps)
            if selected is None:
                errors.append(f"could_not_reduce_robust_component_to_{target}_without_breaking_connectivity")
                break
            sets[str(target)] = selected
            removal_orders[str(target)] = removals
            current = selected

        if len(sets) == len(TARGET_SIZES):
            for target in TARGET_SIZES:
                rank_failures = rank2_for_drop_contexts(sets[str(target)], by_id)
                if rank_failures:
                    errors.append(f"rank_below_2_in_{target}_view_drop_contexts")
                    removal_orders.setdefault("rank_failures", {})[str(target)] = rank_failures

        status = "pass" if len(sets) == len(TARGET_SIZES) and not errors else "fail"
        result = {
            "scene": scene,
            "status": status,
            "failure_reasons": errors,
            "calibrated_photo_count": len(frames),
            "eligible_photo_count": len(eligible),
            "excluded_photos": excluded,
            "graph_edge_count": sum(map(len, graph.values())) // 2,
            "overlap_threshold": OVERLAP_THRESHOLD,
            "biconnected_component_sizes": [len(component) for component in components],
            "selected_max_component": chosen,
            "sets": sets,
            "removal_orders": removal_orders,
            "pair_overlaps": pair_rows,
            "photo_validity": [
                {"image_id": frame["id"], "valid_gt_samples": frame["valid_sample_count"],
                 "valid_grid_mask_sha256": frame["valid_grid_mask_sha256"]}
                for frame in prepared
            ],
            "calibration_hashes": calibration_hashes,
            "depth_archive": archive.name,
            "depth_archive_sha256": archive_hash,
            "depth_coordinate_space": "official distorted DSLR coordinates; pycolmap THIN_PRISM_FISHEYE projection",
        }
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--temp-root", type=Path, default=Path("/private/tmp"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if shutil.which("bsdtar") is None:
        raise RuntimeError("bsdtar is required to extract the official ETH3D archives")
    missing = [scene for scene in SCENES
               if not (args.archive_root / f"{scene}_dslr_depth.7z").is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing official depth archives: " + ", ".join(f"{scene}_dslr_depth.7z" for scene in missing)
        )
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing selection result: {args.output}")

    args.temp_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for scene in SCENES:
        archive = args.archive_root / f"{scene}_dslr_depth.7z"
        print(f"{scene}: mapping official depth to common evaluation grid", flush=True)
        results[scene] = select_scene(scene, archive, args.temp_root)
        print(f"{scene}: {results[scene]['status']} {results[scene].get('failure_reasons', results[scene].get('reason', []))}", flush=True)

    passed_new = [scene for scene, result in results.items()
                  if scene != "office" and result["status"] == "pass"]
    document = {
        "selection_status": "official_section_6_result",
        "method": {
            "start_set": "all photos passing official file, calibration, and GT validity checks",
            "gt_validity_min_samples": MIN_VALID_GT_SAMPLES,
            "pair_overlap_threshold": OVERLAP_THRESHOLD,
            "pair_overlap_validity": "both directed fractions averaged; connect if mean >= threshold",
            "depth_match_tolerance": "abs(z_ref - z_projected) <= max(0.05 m, 0.02 * z_projected)",
            "common_grid": {
                "size": [252, 160], "crop_size": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
                "sample_centers": "x=4*j+2; y=16+4*k+2; j=0..251; k=0..159",
                "distorted_gt_mapping": "undistorted pinhole ray projected into official distorted COLMAP camera",
            },
            "maximal_subset": "largest Tarjan vertex-biconnected component",
            "equal_component_tie_break": "lexicographically smallest sorted image ID list",
            "nested_reduction": "greedy removal of lowest mean overlap, preserving vertex-biconnectivity after each removal; ties by image ID",
            "target_sizes": list(TARGET_SIZES),
            "rank_check": "centered camera-center rank >= 2 in every single-view DROP context",
        },
        "excluded_from_candidate_list": {"pipes": "14 calibrated photos; below 15-view minimum"},
        "passing_new_scene_count": len(passed_new),
        "minimum_new_scenes_required": 5,
        "meets_minimum_new_scenes": len(passed_new) >= 5,
        "scenes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_name(args.output.name + ".tmp")
    temp_output.write_text(json.dumps(document, indent=2) + "\n")
    temp_output.replace(args.output)
    print(f"Wrote official selection result: {args.output}", flush=True)


if __name__ == "__main__":
    main()
