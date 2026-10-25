"""Build reviewable per-view source bundles for the ETH3D restart study.

This staging bundle follows the existing office layout. Its laser_depth.npy is
the legacy aligned-scan z-buffer reference; the approved study still requires
its official depth GT preparation before any model inference.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/eth3d/high_res_multi_view"
OUT_ROOT = ROOT / "data/experiments/eth3d_multi_view"
# Preview operations follow the active study so excluded scene folders stay absent.
SCENES = tuple(json.loads((ROOT / "experiments/study_v1.json").read_text())["primary_scenes"])
SEVERITIES = (0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1, 2, 4)
EXISTING_SEVERITIES = (0, 0.25, 0.5, 1, 2, 4)
MISSING_SEVERITIES = (0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
VIEW_COUNT = 15
OUTPUT_SIZE = (1008, 672)
OVERSCAN_WIDTH = 1120
OVERLAP_THRESHOLD = 0.1


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_scene(scene_name: str) -> tuple[Path, dict[int, dict], list[dict]]:
    scene = DATA_ROOT / "multi_view_training_dslr_undistorted" / scene_name
    camera_rows = [line.split() for line in
                   (scene / "dslr_calibration_undistorted/cameras.txt").read_text().splitlines()
                   if line and not line.startswith("#")]
    cameras = {}
    for row in camera_rows:
        camera_id, model = int(row[0]), row[1]
        if model != "PINHOLE":
            raise ValueError(f"{scene_name}: unsupported camera model {model}")
        width, height = map(int, row[2:4])
        fx, fy, cx, cy = map(float, row[4:8])
        cameras[camera_id] = {
            "size": (width, height),
            "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], dtype=np.float64),
        }

    lines = (scene / "dslr_calibration_undistorted/images.txt").read_text().splitlines()
    frames = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 10:
            raise ValueError(f"{scene_name}: malformed COLMAP image row: {line}")
        camera_id = int(f[8])
        q = np.array(list(map(float, f[1:5])))
        R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        t = np.array(list(map(float, f[5:8])))
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3], c2w[:3, 3] = R.T, -R.T @ t
        filename = Path(f[9]).name
        frames.append({
            "id": Path(filename).stem,
            "filename": filename,
            "camera_id": camera_id,
            "c2w": c2w,
        })
        i += 1  # COLMAP observations row

    frames.sort(key=lambda frame: frame["filename"])
    return scene, cameras, frames


def uniform_indices(count: int, selected: int = VIEW_COUNT) -> list[int]:
    if count < selected:
        raise ValueError(f"Need at least {selected} calibrated views; got {count}")
    indexes = np.rint(np.linspace(0, count - 1, selected)).astype(int).tolist()
    if len(set(indexes)) != selected:
        raise ValueError(f"Uniform selection produced duplicate indexes for count={count}")
    return indexes


def aligned_laser_points(scene_name: str) -> np.ndarray:
    folder = DATA_ROOT / "multi_view_training_dslr_scan_eval" / scene_name / "dslr_scan_eval"
    chunks = []
    for mesh in ET.parse(folder / "scan_alignment.mlp").iter("MLMesh"):
        transform = np.fromstring(mesh.find("MLMatrix44").text, sep=" ").reshape(4, 4)
        vertices = PlyData.read(folder / mesh.attrib["filename"])["vertex"]
        xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float32)
        chunks.append(xyz @ transform[:3, :3].T + transform[:3, 3])
    if not chunks:
        raise ValueError(f"{scene_name}: no aligned scan meshes")
    return np.concatenate(chunks).astype(np.float32)


def rasterize(points: np.ndarray, c2w: np.ndarray, K: np.ndarray,
              shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    zbuf = np.full(height * width, np.inf, np.float32)
    for xyz in np.array_split(points, max(1, len(points) // 500_000)):
        cam = (xyz - c2w[:3, 3]) @ c2w[:3, :3]
        cam = cam[cam[:, 2] > 0]
        uv = cam @ K.T
        uv = np.floor(uv[:, :2] / uv[:, 2:] + 0.5).astype(np.int64)
        valid = ((uv[:, 0] >= 0) & (uv[:, 0] < width) &
                 (uv[:, 1] >= 0) & (uv[:, 1] < height))
        np.minimum.at(zbuf, uv[valid, 1] * width + uv[valid, 0], cam[valid, 2])
    zbuf[~np.isfinite(zbuf)] = 0
    return zbuf.reshape(height, width)


def exposure(image: np.ndarray, K: np.ndarray, degrees: float,
             samples: int = 33, *, force_pipeline: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if degrees == 0 and not force_pipeline:
        return image.copy(), np.ones(image.shape[:2], dtype=np.float32)
    srgb = image.astype(np.float32) / 255
    linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
    integrated = np.zeros_like(linear)
    support = np.ones(image.shape[:2], dtype=np.float32)
    height, width = image.shape[:2]
    for angle in np.linspace(-degrees / 2, degrees / 2, samples):
        H = K @ Rotation.from_euler("y", angle, degrees=True).as_matrix() @ np.linalg.inv(K)
        integrated += cv2.warpPerspective(linear, H, (width, height), flags=cv2.INTER_LINEAR) / samples
        support = np.minimum(
            support,
            cv2.warpPerspective(np.ones((height, width), np.float32), H, (width, height)),
        )
    srgb_out = np.where(integrated <= .0031308, 12.92 * integrated,
                        1.055 * np.maximum(integrated, 0) ** (1 / 2.4) - .055)
    return np.clip(np.rint(srgb_out * 255), 0, 255).astype(np.uint8), support


def visible_overlap(a: dict, b: dict, depth_a: np.ndarray,
                   depth_b: np.ndarray) -> float:
    y, x = np.mgrid[0:depth_a.shape[0]:8, 0:depth_a.shape[1]:8]
    z = depth_a[y, x].ravel()
    pixels = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)])
    valid = z > 0
    if not valid.any():
        return 0.0
    cam = (pixels[valid] @ np.linalg.inv(a["Kcrop"]).T) * z[valid, None]
    Ta, Tb = a["c2w"], b["c2w"]
    world = cam @ Ta[:3, :3].T + Ta[:3, 3]
    other = (world - Tb[:3, 3]) @ Tb[:3, :3]
    front = other[:, 2] > 0
    other = other[front]
    if not len(other):
        return 0.0
    uv = other @ b["Kcrop"].T
    uv = np.floor(uv[:, :2] / uv[:, 2:] + 0.5).astype(int)
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < depth_b.shape[1]) &
              (uv[:, 1] >= 0) & (uv[:, 1] < depth_b.shape[0]))
    ref = depth_b[uv[inside, 1], uv[inside, 0]]
    visible = (ref > 0) & (np.abs(ref - other[inside, 2]) < np.maximum(.05, .02 * ref))
    return float(visible.sum() / max(1, valid.sum()))


def graph_connected(ids: list[str], overlaps: dict[frozenset, float]) -> bool:
    if len(ids) < 2:
        return True
    seen, frontier = set(), [ids[0]]
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(other for other in ids if other not in seen and
                        overlaps.get(frozenset((current, other)), 0) >= OVERLAP_THRESHOLD)
    return len(seen) == len(ids)


def robustly_connected(ids: list[str], overlaps: dict[frozenset, float]) -> bool:
    return all(graph_connected([other for other in ids if other != removed], overlaps)
               for removed in ids)


def nested_sets(frames: list[dict], overlaps: dict[frozenset, float]) -> tuple[dict, list[str]]:
    ids = [frame["id"] for frame in frames]
    sets = {"15": ids.copy()}
    errors = []
    current = ids.copy()
    if not robustly_connected(current, overlaps):
        errors.append("15-view overlap graph is not connected after every single-view DROP")
        return sets, errors
    for target in (10, 5):
        while len(current) > target:
            candidates = []
            for removed in current:
                kept = [view_id for view_id in current if view_id != removed]
                if robustly_connected(kept, overlaps):
                    mean_overlap = np.mean([
                        overlaps.get(frozenset((removed, other)), 0.0) for other in kept
                    ])
                    candidates.append((float(mean_overlap), removed, kept))
            if not candidates:
                errors.append(f"Could not construct robustly connected nested {target}-view set")
                break
            _, _, current = min(candidates, key=lambda item: (item[0], item[1]))
        if len(current) == target:
            sets[str(target)] = current.copy()
        else:
            break
    return sets, errors


def prepare_scene(scene_name: str) -> dict:
    scene, cameras, all_frames = read_scene(scene_name)
    indexes = uniform_indices(len(all_frames))
    frames = [all_frames[index].copy() for index in indexes]
    scene_out = OUT_ROOT / scene_name
    if scene_out.exists() and any(scene_out.iterdir()):
        raise FileExistsError(f"Refusing to mix with existing output: {scene_out}")
    scene_out.mkdir(parents=True, exist_ok=True)

    points = aligned_laser_points(scene_name)
    prepared = []
    contact = Image.new("RGB", (5 * 252, 4 * 188), "white")
    common_width, common_height = OUTPUT_SIZE

    for index, frame in enumerate(frames):
        camera = cameras[frame["camera_id"]]
        image_path = scene / "images/dslr_images_undistorted" / frame["filename"]
        with Image.open(image_path) as source_image:
            source_image = source_image.convert("RGB")
            original_size = source_image.size
            if original_size != camera["size"]:
                raise ValueError(f"{scene_name}/{frame['filename']}: image size {original_size} "
                                 f"does not match CAMERA_ID size {camera['size']}")
            source_w, source_h = original_size
            scaled_w = OVERSCAN_WIDTH
            scaled_h = round(source_h * scaled_w / source_w)
            sx, sy = scaled_w / source_w, scaled_h / source_h
            Ksource = np.diag([sx, sy, 1.]) @ camera["K"]
            left, top = (scaled_w - common_width) // 2, (scaled_h - common_height) // 2
            if left < 0 or top < 0:
                raise ValueError(f"{scene_name}/{frame['filename']}: image too small for fixed crop")
            Kcrop = Ksource.copy()
            Kcrop[0, 2] -= left
            Kcrop[1, 2] -= top
            rgb = np.asarray(source_image.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS))

        folder = scene_out / frame["id"]
        folder.mkdir()
        variants = {}
        for severity in SEVERITIES:
            blurred, support = exposure(rgb, Ksource, severity)
            crop_support = support[top:top + common_height, left:left + common_width]
            if crop_support.shape != (common_height, common_width) or np.min(crop_support) < .999:
                raise ValueError(f"{scene_name}/{frame['filename']}: yaw {severity} has empty warp border")
            path = folder / f"yaw_{severity:g}.png"
            Image.fromarray(blurred[top:top + common_height, left:left + common_width]).save(path)
            variants[str(severity)] = {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}

        depth = rasterize(points, frame["c2w"], Kcrop, (common_height, common_width))
        depth_path = folder / "laser_depth.npy"
        np.save(depth_path, depth)
        frame.update({
            "source": image_path.relative_to(ROOT).as_posix(),
            "source_sha256": digest(image_path),
            "source_size": list(original_size),
            "intrinsics": Kcrop.tolist(),
            "Kcrop": Kcrop,
            "variants": variants,
            "depth": depth_path.relative_to(ROOT).as_posix(),
            "depth_sha256": digest(depth_path),
            "laser_coverage": float(np.mean(depth > 0)),
        })
        prepared.append(frame)
        thumb = Image.open(folder / "yaw_0.png").resize((252, 168))
        contact.paste(thumb, ((index % 5) * 252, (index // 5) * 188))
        ImageDraw.Draw(contact).text(((index % 5) * 252 + 4, (index // 5) * 188 + 169),
                                     frame["id"], fill="black")
        print(f"{scene_name}: prepared {index + 1}/15 {frame['id']} "
              f"(laser coverage {frame['laser_coverage']:.1%})", flush=True)

    descriptors = []
    overlaps = {}
    for i, a in enumerate(prepared):
        da = np.load(ROOT / a["depth"])
        for b in prepared[i + 1:]:
            db = np.load(ROOT / b["depth"])
            overlap = .5 * (visible_overlap(a, b, da, db) + visible_overlap(b, a, db, da))
            overlaps[frozenset((a["id"], b["id"]))] = overlap
            Ta, Tb = a["c2w"], b["c2w"]
            descriptors.append({
                "a": a["id"], "b": b["id"],
                "baseline_m": float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])),
                "rotation_deg": float(np.rad2deg(
                    Rotation.from_matrix(Ta[:3, :3].T @ Tb[:3, :3]).magnitude())),
                "laser_visibility_overlap": overlap,
            })

    sets, selection_errors = nested_sets(prepared, overlaps)
    for frame in prepared:
        frame.pop("Kcrop")
        frame["c2w"] = frame["c2w"].tolist()
    contact.save(scene_out / "selected_views.png")
    manifest = {
        "scene": scene_name,
        "dataset": "ETH3D high-res multi-view training",
        "source_archive": "data/eth3d/high_res_multi_view/multi_view_training_dslr_undistorted.7z",
        "view_count": len(prepared),
        "staging_sets": sets,
        "severities_deg": list(SEVERITIES),
        "frames": prepared,
        "official_selection_status": "pending_section_6_using_official_depth_gt",
        "staging_selection": {
            "status": "temporary_data_organization_only_not_official_selection",
            "calibrated_views": len(all_frames),
            "sorted_by": "calibration image filename, ascending lexical order",
            "max_set": "15 indexes selected by round(linspace(0, N-1, 15)); Python round-to-even ties",
            "nested_sets": "Greedily remove the lowest mean laser visibility-overlap view while preserving graph connectivity after every single-view DROP",
            "overlap_edge_threshold": OVERLAP_THRESHOLD,
            "uses_model_outputs": False,
            "precheck_errors": selection_errors,
        },
        "preparation": {
            "image_size": list(OUTPUT_SIZE),
            "overscan_width": OVERSCAN_WIDTH,
            "blur_levels_deg": list(SEVERITIES),
            "yaw_axis": "camera y",
            "blur_path": "symmetric constant angular velocity",
            "exposure_samples": 33,
            "rgb_integration": "linear sRGB radiance",
            "geometry_reference": "Aligned ETH3D laser scans rasterized by nearest-depth z-buffer; no official occlusion masks",
            "official_protocol_note": "All 12 planned blur levels are included. laser_depth.npy remains a preview reference; official occlusion-aware depth GT and the approved evaluation map are still required before model inference.",
        },
        "pair_geometry": descriptors,
    }
    (scene_out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def add_missing_protocol_variants() -> None:
    """Add the six finer study blur levels while preserving existing files."""
    for scene_name in SCENES:
        scene_out = OUT_ROOT / scene_name
        manifest_path = scene_out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for number, frame in enumerate(manifest["frames"], start=1):
            folder = scene_out / frame["id"]
            source_path = ROOT / frame["source"]
            if digest(source_path) != frame["source_sha256"]:
                raise ValueError(f"{scene_name}/{frame['id']}: source hash changed")
            with Image.open(source_path) as source_image:
                source_image = source_image.convert("RGB")
                source_w, source_h = source_image.size
                if [source_w, source_h] != frame["source_size"]:
                    raise ValueError(f"{scene_name}/{frame['id']}: source dimensions changed")
                scaled_w = OVERSCAN_WIDTH
                scaled_h = round(source_h * scaled_w / source_w)
                left = (scaled_w - OUTPUT_SIZE[0]) // 2
                top = (scaled_h - OUTPUT_SIZE[1]) // 2
                Ksource = np.asarray(frame["intrinsics"], dtype=np.float64).copy()
                Ksource[0, 2] += left
                Ksource[1, 2] += top
                rgb = np.asarray(source_image.resize(
                    (scaled_w, scaled_h), Image.Resampling.LANCZOS
                ))

            variants = frame.setdefault("variants", {})
            for severity in EXISTING_SEVERITIES:
                key = f"{severity:g}"
                path = folder / f"yaw_{severity:g}.png"
                if key not in variants or not path.is_file():
                    raise FileNotFoundError(f"{scene_name}/{frame['id']}: missing existing variant {path.name}")
                if digest(path) != variants[key]["sha256"]:
                    raise ValueError(f"{scene_name}/{frame['id']}: existing variant hash mismatch: {path.name}")
            for severity in MISSING_SEVERITIES:
                key = str(severity)
                path = folder / f"yaw_{severity:g}.png"
                if key in variants and path.exists():
                    if digest(path) != variants[key]["sha256"]:
                        raise ValueError(f"{scene_name}/{frame['id']}: existing variant hash mismatch: {path.name}")
                    continue
                if path.exists():
                    raise FileExistsError(f"Refusing to overwrite unrecorded variant: {path}")
                blurred, support = exposure(rgb, Ksource, severity)
                crop_support = support[top:top + OUTPUT_SIZE[1], left:left + OUTPUT_SIZE[0]]
                if crop_support.shape != (OUTPUT_SIZE[1], OUTPUT_SIZE[0]) or np.min(crop_support) < .999:
                    raise ValueError(f"{scene_name}/{frame['id']}: yaw {severity:g} has empty warp border")
                cropped = blurred[top:top + OUTPUT_SIZE[1], left:left + OUTPUT_SIZE[0]]
                temp_path = path.with_name(path.name + ".tmp")
                Image.fromarray(cropped).save(temp_path, format="PNG")
                temp_path.replace(path)
                variants[key] = {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": digest(path),
                }
            print(f"{scene_name}: completed blur levels {number}/15 for {frame['id']}", flush=True)

        manifest["severities_deg"] = list(SEVERITIES)
        manifest["preparation"]["blur_levels_deg"] = list(SEVERITIES)
        manifest["preparation"]["official_protocol_note"] = (
            "All 12 planned blur levels are present. laser_depth.npy remains a preview reference; "
            "official occlusion-aware depth GT and the approved evaluation map are still required "
            "before model inference."
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{scene_name}: manifest updated", flush=True)

    index_path = OUT_ROOT / "dataset_index.json"
    index = json.loads(index_path.read_text())
    for scene_name, summary in index.items():
        summary["blur_levels_deg"] = list(SEVERITIES)
        summary["blur_level_count"] = len(SEVERITIES)
    index_path.write_text(json.dumps(index, indent=2) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--complete-blur-levels", action="store_true",
                        help="add missing planned yaw variants to existing source folders")
    args = parser.parse_args()
    if not OUT_ROOT.exists():
        OUT_ROOT.mkdir(parents=True)
    if args.complete_blur_levels:
        add_missing_protocol_variants()
        return
    summaries = {}
    for scene_name in SCENES:
        summaries[scene_name] = prepare_scene(scene_name)
    summary = {
        scene: {
            "views": manifest["view_count"],
            "staging_sets": {key: len(ids) for key, ids in manifest["staging_sets"].items()},
            "staging_selection_status": manifest["staging_selection"]["status"],
            "staging_screen_errors": manifest["staging_selection"]["precheck_errors"],
            "official_selection_status": "pending_section_6_using_official_depth_gt",
            "blur_levels_deg": list(SEVERITIES),
            "blur_level_count": len(SEVERITIES),
        }
        for scene, manifest in summaries.items()
    }
    (OUT_ROOT / "dataset_index.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
