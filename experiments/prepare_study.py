"""Prepare the approved per-scene RGB variants and manifest for inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_source_data import DATA_ROOT, exposure, read_scene
from select_official_scenes import evaluation_grid, read_scene_calibrations


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "data/experiments/eth3d_multi_view"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def write_variant(path: Path, pixels: np.ndarray) -> None:
    if path.exists():
        with Image.open(path) as existing:
            if not np.array_equal(np.asarray(existing.convert("RGB")), pixels):
                raise ValueError(f"Existing prepared image differs from approved procedure: {path}")
        return
    temp = path.with_name(path.stem + f".{os.getpid()}.tmp.png")
    Image.fromarray(pixels).save(temp)
    temp.replace(path)


def validate_camera_roundtrip(scene_name: str, ids: list[str], protocol: dict) -> float:
    calibrated, _ = read_scene_calibrations(scene_name)
    by_id = {frame["id"]: frame for frame in calibrated}
    _, cameras, all_frames = read_scene(scene_name)
    sources = {frame["id"]: frame for frame in all_frames}
    x, y = evaluation_grid()
    max_error = 0.0
    for image_id in ids:
        frame = by_id[image_id]
        source = sources[image_id]
        camera = cameras[source["camera_id"]]
        source_width, source_height = camera["size"]
        width, height = protocol["preparation"]["crop_width"], protocol["preparation"]["crop_height"]
        scaled_width = protocol["preparation"]["source_width"]
        scaled_height = round(source_height * scaled_width / source_width)
        left, top = (scaled_width - width) // 2, (scaled_height - height) // 2
        K = np.diag([scaled_width / source_width, scaled_height / source_height, 1.0]) @ camera["K"]
        K[0, 2] -= left
        K[1, 2] -= top
        xy = np.column_stack(((x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1]))
        distorted_camera = frame["distorted_camera"]
        pixels = np.asarray(distorted_camera.img_from_cam(
            np.column_stack((xy, np.ones(len(xy)))), check_cheirality=True), dtype=np.float64)
        recovered = np.asarray(distorted_camera.cam_from_img(pixels), dtype=np.float64)
        focal = np.asarray(distorted_camera.params[:2], dtype=np.float64)
        error = np.linalg.norm((recovered - xy) * focal[None, :], axis=1)
        max_error = max(max_error, float(error.max()))
    return max_error


def validate_blur_convergence(rgb: np.ndarray, K: np.ndarray, protocol: dict,
                              crop: tuple[int, int, int, int]) -> dict:
    settings = protocol["preparation"]["validation"]
    reference_samples = settings["blur_integration_reference_samples"][-1]
    degree = settings["blur_integration_check_deg"]
    base_samples = protocol["preparation"]["exposure_samples"]
    base, _ = exposure(rgb, K, degree, base_samples, force_pipeline=True)
    reference, _ = exposure(rgb, K, degree, reference_samples, force_pipeline=True)
    left, top, width, height = crop
    a = base[top:top + height, left:left + width]
    b = reference[top:top + height, left:left + width]

    def linear(pixels):
        srgb = pixels.astype(np.float32) / 255
        return np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)

    difference = np.abs(linear(a) - linear(b))
    mean_error = float(difference.mean())
    p99_error = float(np.quantile(difference, .99))
    if (mean_error > settings["blur_integration_linear_rgb_mean_absolute_error_max"]
            or p99_error > settings["blur_integration_linear_rgb_absolute_error_p99_max"]):
        raise ValueError(f"{degree:g}-degree blur integration failed the {base_samples}/{reference_samples}-sample convergence check")
    return {"degree": degree, "samples": [base_samples, reference_samples],
            "linear_rgb_mean_absolute_error": mean_error,
            "linear_rgb_absolute_error_p99": p99_error,
            "crop_support_min": 1.0}


def prepare_scene(scene_name: str, protocol: dict, selection: dict,
                  input_root: Path) -> dict:
    selected = selection["scenes"].get(scene_name)
    if not selected or selected.get("status") != "pass":
        raise ValueError(f"{scene_name}: scene lacks an official passing selection")
    sets = selected["sets"]
    ids = sets["15"]
    if (len(ids) != 15 or not set(sets["10"]) <= set(ids)
            or not set(sets["5"]) <= set(sets["10"])):
        raise ValueError(f"{scene_name}: official sets are not nested 15/10/5")

    scene_root, cameras, all_frames = read_scene(scene_name)
    frame_by_id = {frame["id"]: frame for frame in all_frames}
    if set(ids) - frame_by_id.keys():
        raise ValueError(f"{scene_name}: official selection references missing calibrated photos")
    output_scene = input_root / scene_name
    output_scene.mkdir(parents=True, exist_ok=True)
    old_manifest_path = output_scene / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text()) if old_manifest_path.is_file() else {}
    old_frames = {frame["id"]: frame for frame in old_manifest.get("frames", [])}

    gt_manifest_path = output_scene / "official_ground_truth_manifest.json"
    gt_manifest = json.loads(gt_manifest_path.read_text())
    selection_path = input_root / "official_selection.json"
    selection_sha = sha256(selection_path)
    if (gt_manifest.get("protocol_id") != protocol["protocol_id"]
            or gt_manifest.get("official_selection_sha256") != selection_sha
            or gt_manifest.get("selected_sets") != sets):
        raise ValueError(f"{scene_name}: official GT does not match the approved selection/protocol")
    gt_by_id = {row["image_id"]: row for row in gt_manifest["images"]}
    if set(gt_by_id) != set(ids):
        raise ValueError(f"{scene_name}: official GT files do not match the selected 15 views")

    preparation = protocol["preparation"]
    camera_error = validate_camera_roundtrip(scene_name, ids, protocol)
    camera_limit = preparation["validation"]["camera_distortion_roundtrip_max_original_px"]
    if camera_error > camera_limit:
        raise ValueError(f"{scene_name}: camera distortion roundtrip error {camera_error} px exceeds {camera_limit} px")
    width, height = preparation["crop_width"], preparation["crop_height"]
    overscan_width = preparation["source_width"]
    samples = preparation["exposure_samples"]
    severities = protocol["blur_degrees"]
    frames = []
    blur_convergence = None
    for number, image_id in enumerate(ids, start=1):
        frame = frame_by_id[image_id]
        camera = cameras[frame["camera_id"]]
        source_path = scene_root / "images/dslr_images_undistorted" / frame["filename"]
        source_hash = sha256(source_path)
        old_frame = old_frames.get(image_id)
        if old_frame:
            if old_frame.get("source_sha256") != source_hash:
                raise ValueError(f"{scene_name}/{image_id}: previous source hash differs")
            for degree, variant in old_frame.get("variants", {}).items():
                if degree in {f"{b:g}" for b in severities}:
                    old_path = ROOT / variant["path"]
                    if not old_path.is_file() or sha256(old_path) != variant["sha256"]:
                        raise ValueError(f"{scene_name}/{image_id}: existing variant changed: {old_path}")

        with Image.open(source_path) as opened:
            source_image = opened.convert("RGB")
            source_width, source_height = source_image.size
            if (source_width, source_height) != camera["size"]:
                raise ValueError(f"{scene_name}/{image_id}: source dimensions differ from its CAMERA_ID")
            scaled_height = round(source_height * overscan_width / source_width)
            sx, sy = overscan_width / source_width, scaled_height / source_height
            Ksource = np.diag([sx, sy, 1.0]) @ camera["K"]
            left, top = (overscan_width - width) // 2, (scaled_height - height) // 2
            if left < 0 or top < 0:
                raise ValueError(f"{scene_name}/{image_id}: source is too small for the approved crop")
            Kcrop = Ksource.copy()
            Kcrop[0, 2] -= left
            Kcrop[1, 2] -= top
            rgb = np.asarray(source_image.resize(
                (overscan_width, scaled_height), Image.Resampling.LANCZOS
            ))

        if blur_convergence is None:
            blur_convergence = validate_blur_convergence(
                rgb, Ksource, protocol, (left, top, width, height)
            )

        image_dir = output_scene / image_id
        image_dir.mkdir(parents=True, exist_ok=True)
        variants = {}
        zero_motion_verified = False
        for degree in severities:
            rendered, support = exposure(rgb, Ksource, degree, samples, force_pipeline=True)
            cropped_support = support[top:top + height, left:left + width]
            if cropped_support.shape != (height, width) or float(cropped_support.min()) < preparation["minimum_warp_support"]:
                raise ValueError(f"{scene_name}/{image_id}: blur {degree:g} leaves an invalid crop border")
            pixels = rendered[top:top + height, left:left + width]
            variant_path = image_dir / f"yaw_{degree:g}.png"
            write_variant(variant_path, pixels)
            if degree == 0:
                clean_reference = rgb[top:top + height, left:left + width]
                if not np.array_equal(clean_reference, pixels):
                    raise ValueError(f"{scene_name}/{image_id}: full zero-blur pipeline differs from clean RGB")
                if old_frame and "0" in old_frame.get("variants", {}):
                    previous_clean = np.asarray(Image.open(ROOT / old_frame["variants"]["0"]["path"]).convert("RGB"))
                    if not np.array_equal(previous_clean, pixels):
                        raise ValueError(f"{scene_name}/{image_id}: previous clean RGB differs from full pipeline")
                zero_motion_verified = True
            variants[f"{degree:g}"] = {
                "path": variant_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(variant_path),
            }

        gt = gt_by_id[image_id]
        depth_path = output_scene / gt["depth_file"]
        mask_path = output_scene / gt["mask_file"]
        metadata_path = output_scene / gt["metadata_file"]
        frames.append({
            "id": image_id,
            "filename": frame["filename"],
            "camera_id": frame["camera_id"],
            "c2w": frame["c2w"].tolist(),
            "source": source_path.relative_to(ROOT).as_posix(),
            "source_sha256": source_hash,
            "source_size": [source_width, source_height],
            "intrinsics": Kcrop.tolist(),
            "variants": variants,
            "depth": depth_path.relative_to(ROOT).as_posix(),
            "depth_sha256": gt["depth_sha256"],
            "official_valid_mask": mask_path.relative_to(ROOT).as_posix(),
            "official_valid_mask_sha256": gt["mask_sha256"],
            "official_gt_metadata": metadata_path.relative_to(ROOT).as_posix(),
            "official_gt_metadata_sha256": gt["metadata_sha256"],
            "official_gt_valid_pixels": gt["valid_pixel_count"],
            "zero_motion_full_pipeline_matches_clean": zero_motion_verified,
        })
        print(f"{scene_name}: prepared {number}/15 {image_id}", flush=True)

    manifest = {
        "protocol_id": protocol["protocol_id"],
        "scene": scene_name,
        "dataset": "ETH3D high-res multi-view training",
        "official_selection_status": "pass",
        "official_selection_sha256": selection_sha,
        "official_ground_truth_manifest": gt_manifest_path.relative_to(ROOT).as_posix(),
        "official_ground_truth_manifest_sha256": sha256(gt_manifest_path),
        "sets": sets,
        "severities_deg": severities,
        "frames": frames,
        "preparation_validation": {
            "zero_motion_all_frames_matches_clean": all(
                frame["zero_motion_full_pipeline_matches_clean"] for frame in frames
            ),
            "all_blur_crops_meet_minimum_warp_support": True,
            "official_gt_mask_count_matches_selection": True,
            "preflight_status": "pass",
            "camera_distortion_roundtrip_max_original_px": camera_error,
            "camera_distortion_roundtrip_limit_original_px": camera_limit,
            "blur_integration_convergence": blur_convergence,
        },
        "preparation": {
            "image_size": [width, height],
            "overscan_width": overscan_width,
            "exposure_samples": samples,
            "yaw_axis": preparation["exposure_axis"],
            "blur_path": preparation["exposure_path"],
            "rgb_integration": preparation["integration"],
            "zero_motion_uses_full_pipeline": preparation["zero_motion_uses_full_pipeline"],
            "ground_truth": preparation["ground_truth"],
            "evaluation_roi": preparation["evaluation_roi"],
            "evaluation_grid": preparation["evaluation_grid"],
        },
    }
    atomic_json(output_scene / "manifest.json", manifest)
    print(f"{scene_name}: wrote approved manifest with 15 official views", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/study_v1.json"))
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--scenes", nargs="+", required=True)
    args = parser.parse_args()
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    input_root = args.input_root if args.input_root.is_absolute() else ROOT / args.input_root
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "approved":
        raise RuntimeError("The study plan must be approved before preparation")
    if protocol.get("approved_plan_sha256") != sha256(ROOT / protocol["plan"]):
        raise ValueError("The approved plan text has changed")
    selection_path = input_root / "official_selection.json"
    selection = json.loads(selection_path.read_text())
    for scene in args.scenes:
        if scene not in protocol["primary_scenes"]:
            raise ValueError(f"Scene is not in the approved primary cohort: {scene}")
        prepare_scene(scene, protocol, selection, input_root)


if __name__ == "__main__":
    main()
