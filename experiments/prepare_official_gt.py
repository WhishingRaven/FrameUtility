"""Create official ETH3D depth and validity-mask snapshots for selected views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from select_official_scenes import (
    ARCHIVE_ROOT,
    OUTPUT_PATH as SELECTION_PATH,
    ROOT,
    prepare_official_depth_samples,
    read_scene_calibrations,
    sha256,
)


PROTOCOL_PATH = ROOT / "experiments/study_v1.json"
DEFAULT_OUTPUT_ROOT = ROOT / "data/experiments/eth3d_multi_view"
DEFAULT_TEMP_ROOT = Path("/private/tmp")
GRID_SIZE = (252, 160)


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temp.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sample_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def generate_scene(scene: str, selection: dict, protocol: dict,
                   output_root: Path, temp_root: Path) -> str:
    result = selection["scenes"].get(scene)
    if not result or result.get("status") != "pass":
        raise ValueError(f"{scene}: no passing official 15/10/5-view selection")
    sets = result["sets"]
    ids = sets["15"]
    if len(ids) != 15 or not set(sets["10"]) <= set(ids) or not set(sets["5"]) <= set(sets["10"]):
        raise ValueError(f"{scene}: official selection sets are not correctly nested")

    archive = ARCHIVE_ROOT / result["depth_archive"]
    if not archive.is_file() or sha256(archive) != result["depth_archive_sha256"]:
        raise ValueError(f"{scene}: official depth archive missing or changed")
    frames, calibration_hashes = read_scene_calibrations(scene)
    if calibration_hashes != result["calibration_hashes"]:
        raise ValueError(f"{scene}: calibration inputs differ from official selection")
    frames_by_id = {frame["id"]: frame for frame in frames}
    if not set(ids) <= set(frames_by_id):
        raise ValueError(f"{scene}: selected photo is missing from calibration files")

    scene_root = output_root / scene
    manifest_path = scene_root / "official_ground_truth_manifest.json"
    if manifest_path.is_file():
        existing = load_json(manifest_path)
        if (existing.get("source_archive_sha256") == result["depth_archive_sha256"]
                and existing.get("official_selection_sha256") == sha256(SELECTION_PATH)
                and existing.get("protocol_id") == protocol["protocol_id"]):
            for item in existing["images"]:
                for key in ("depth_file", "mask_file", "metadata_file"):
                    file_path = scene_root / item[key]
                    if not file_path.is_file() or sha256(file_path) != item[key.replace("file", "sha256")]:
                        raise ValueError(f"{scene}: saved GT snapshot is missing or changed: {file_path}")
            return "already_complete"
        raise FileExistsError(f"Refusing to replace a different GT snapshot: {manifest_path}")

    scene_root.mkdir(parents=True, exist_ok=True)
    staging = scene_root / f".official-gt-staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Temporary output already exists: {staging}")
    staging.mkdir()
    expected_validity = {row["image_id"]: row for row in result["photo_validity"]}
    image_rows = []
    try:
        with tempfile.TemporaryDirectory(prefix=f"eth3d-gt-{scene}-", dir=temp_root) as temp_dir:
            subprocess.run(["bsdtar", "-xf", str(archive), "-C", temp_dir], check=True)
            extraction_root = Path(temp_dir) / scene
            if not extraction_root.is_dir():
                raise FileNotFoundError(f"{archive}: expected {scene}/ folder is missing")
            chosen_frames = [frames_by_id[view_id] for view_id in ids]
            samples = prepare_official_depth_samples(chosen_frames, extraction_root)
            for frame in samples:
                image_id = frame["id"]
                expected = expected_validity[image_id]
                if frame["valid_sample_count"] != expected["valid_gt_samples"]:
                    raise ValueError(f"{scene}/{image_id}: official valid-pixel count differs from selection")
                if frame["valid_grid_mask_sha256"] != expected["valid_grid_mask_sha256"]:
                    raise ValueError(f"{scene}/{image_id}: official mask differs from selection")

                depth = frame["depth_samples"].reshape(GRID_SIZE[1], GRID_SIZE[0])
                mask = frame["valid_samples"].reshape(GRID_SIZE[1], GRID_SIZE[0])
                image_dir = scene_root / image_id
                image_dir.mkdir(parents=True, exist_ok=True)
                depth_rel = f"{image_id}/official_depth_m.npy"
                mask_rel = f"{image_id}/official_valid_mask.npy"
                metadata_rel = f"{image_id}/official_gt_metadata.json"
                depth_path = image_dir / "official_depth_m.npy"
                mask_path = image_dir / "official_valid_mask.npy"
                metadata_path = image_dir / "official_gt_metadata.json"
                depth_array = depth.astype("<f4", copy=False)
                mask_array = mask.astype(np.bool_, copy=False)
                if depth_path.exists():
                    if not np.array_equal(np.load(depth_path, allow_pickle=False), depth_array):
                        raise ValueError(f"Existing depth differs from official source: {depth_path}")
                else:
                    atomic_npy(depth_path, depth_array)
                if mask_path.exists():
                    if not np.array_equal(np.load(mask_path, allow_pickle=False), mask_array):
                        raise ValueError(f"Existing mask differs from official source: {mask_path}")
                else:
                    atomic_npy(mask_path, mask_array)
                image_metadata = {
                    "protocol_id": protocol["protocol_id"],
                    "scene": scene,
                    "image_id": image_id,
                    "source_depth_filename": frame["gt_depth_file"],
                    "source_archive": archive.relative_to(ROOT).as_posix(),
                    "source_archive_sha256": result["depth_archive_sha256"],
                    "official_selection_sha256": sha256(SELECTION_PATH),
                    "distorted_camera_id": frame["distorted_camera_id"],
                    "undistorted_camera_id": frame["undistorted_camera_id"],
                    "source_depth_size": list(frame["distorted_size"]),
                    "evaluation_intrinsics": frame["Kcrop"].tolist(),
                    "grid_size": list(GRID_SIZE),
                    "depth_units": "meters; camera optical-axis z depth",
                    "sampling": "COLMAP camera projection followed by nearest integer pixel sample; no interpolation or hole filling",
                    "valid_pixel_count": frame["valid_sample_count"],
                    "valid_mask_sha256": frame["valid_grid_mask_sha256"],
                    "depth_array_sha256": sample_hash(depth_array),
                }
                if metadata_path.exists():
                    if load_json(metadata_path) != image_metadata:
                        raise ValueError(f"Existing GT metadata differs from official source: {metadata_path}")
                else:
                    atomic_json(metadata_path, image_metadata)
                image_rows.append({
                    "image_id": image_id,
                    "source_depth_filename": frame["gt_depth_file"],
                    "distorted_camera_id": frame["distorted_camera_id"],
                    "undistorted_camera_id": frame["undistorted_camera_id"],
                    "source_depth_size": list(frame["distorted_size"]),
                    "evaluation_intrinsics": frame["Kcrop"].tolist(),
                    "valid_pixel_count": frame["valid_sample_count"],
                    "valid_mask_sha256": frame["valid_grid_mask_sha256"],
                    "depth_array_sha256": sample_hash(depth_array),
                    "directory": image_id,
                    "depth_file": depth_rel,
                    "depth_sha256": sha256(depth_path),
                    "mask_file": mask_rel,
                    "mask_sha256": sha256(mask_path),
                    "metadata_file": metadata_rel,
                    "metadata_sha256": sha256(metadata_path),
                })
                print(f"{scene}: prepared official GT {image_id} "
                      f"({frame['valid_sample_count']}/40320 valid grid pixels)", flush=True)

        expected_ids = set(ids)
        if {row["image_id"] for row in image_rows} != expected_ids:
            raise ValueError(f"{scene}: generated GT image list differs from official 15-view set")
        snapshot = {
            "protocol_id": protocol["protocol_id"],
            "scene": scene,
            "source": "ETH3D official DSLR depth archive; invalid depth represented by +infinity",
            "source_archive": archive.relative_to(ROOT).as_posix(),
            "source_archive_sha256": result["depth_archive_sha256"],
            "official_selection": SELECTION_PATH.relative_to(ROOT).as_posix(),
            "official_selection_sha256": sha256(SELECTION_PATH),
            "calibration_hashes": calibration_hashes,
            "coordinate_mapping": selection["method"]["common_grid"]["distorted_gt_mapping"],
            "sample_centers": selection["method"]["common_grid"]["sample_centers"],
            "grid_size": list(GRID_SIZE),
            "depth_units": "meters; camera optical-axis z depth",
            "sampling": "COLMAP camera projection followed by nearest integer pixel sample; no interpolation or hole filling",
            "validity": "finite positive official depth; fixed per image, independent of predictions or blur",
            "selected_sets": sets,
            "images": image_rows,
        }
        atomic_json(staging / "official_ground_truth_manifest.json", snapshot)
        (staging / "official_ground_truth_manifest.json").replace(manifest_path)
        shutil.rmtree(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", nargs="+", help="Defaults to every approved primary scene")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--temp-root", type=Path, default=DEFAULT_TEMP_ROOT)
    args = parser.parse_args()
    if shutil.which("bsdtar") is None:
        raise RuntimeError("bsdtar is required to read official ETH3D depth archives")

    protocol = load_json(PROTOCOL_PATH)
    selection = load_json(SELECTION_PATH)
    scenes = args.scene or protocol["primary_scenes"]
    unknown = sorted(set(scenes) - set(protocol["primary_scenes"]))
    if unknown:
        raise ValueError(f"Scenes are not in the approved primary cohort: {unknown}")
    if args.output_root.resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise ValueError("GT snapshots must stay under data/experiments/eth3d_multi_view")
    if args.temp_root.resolve() == args.output_root.resolve() or args.temp_root.resolve().is_relative_to(args.output_root.resolve()):
        raise ValueError("Temporary extraction directory must be outside the experiment output")
    args.temp_root.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        state = generate_scene(scene, selection, protocol, args.output_root, args.temp_root)
        print(f"{scene}: {state}", flush=True)


if __name__ == "__main__":
    main()
