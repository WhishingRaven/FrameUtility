"""Freeze ETH3D inputs, nested sets, rotational exposure integration and laser references."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SEVERITIES = [0, 0.25, 0.5, 1, 2, 4]  # full exposure yaw path, degrees


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_calibration(scene):
    camera = next(line.split() for line in (scene / "dslr_calibration_undistorted/cameras.txt").read_text().splitlines()
                  if line and not line.startswith("#"))
    assert camera[1] == "PINHOLE"
    w, h = map(int, camera[2:4])
    fx, fy, cx, cy = map(float, camera[4:])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    lines = (scene / "dslr_calibration_undistorted/images.txt").read_text().splitlines()
    frames = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        f = line.split()
        q = np.array(list(map(float, f[1:5])))  # COLMAP: qw qx qy qz, world-to-camera
        R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        t = np.array(list(map(float, f[5:8])))
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R.T, -R.T @ t
        frames.append({"id": Path(f[9]).stem, "filename": Path(f[9]).name, "c2w": c2w.tolist()})
        i += 1  # observations line, possibly empty
    return K, (w, h), sorted(frames, key=lambda f: f["filename"])


def exposure(image, K, degrees, samples=33):
    if degrees == 0:
        return image.copy(), 0.
    srgb = image.astype(np.float32) / 255
    linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
    integrated = np.zeros_like(linear)
    h, w = image.shape[:2]
    support = np.ones((h, w), np.float32)
    for angle in np.linspace(-degrees / 2, degrees / 2, samples):
        H = K @ Rotation.from_euler("y", angle, degrees=True).as_matrix() @ np.linalg.inv(K)
        integrated += cv2.warpPerspective(linear, H, (w, h), flags=cv2.INTER_LINEAR) / samples
        support = np.minimum(support, cv2.warpPerspective(np.ones((h, w), np.float32), H, (w, h)))
    out = np.where(integrated <= .0031308, 12.92 * integrated,
                   1.055 * np.maximum(integrated, 0) ** (1 / 2.4) - .055)
    return np.clip(np.rint(out * 255), 0, 255).astype(np.uint8), support


def laser_points(scene):
    folder = scene / "dslr_scan_eval"
    chunks = []
    for mesh in ET.parse(folder / "scan_alignment.mlp").iter("MLMesh"):
        T = np.fromstring(mesh.find("MLMatrix44").text, sep=" ").reshape(4, 4)
        vertices = PlyData.read(folder / mesh.attrib["filename"])["vertex"]
        xyz = np.column_stack([vertices[n] for n in ("x", "y", "z")]).astype(np.float32)
        chunks.append(xyz @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(chunks).astype(np.float32)


def rasterize(points, c2w, K, shape):
    """Nearest laser sample per pixel; no mesh filling or benchmark occlusion masking."""
    h, w = shape
    zbuf = np.full(h * w, np.inf, np.float32)
    for xyz in np.array_split(points, max(1, len(points) // 500_000)):
        cam = (xyz - c2w[:3, 3]) @ c2w[:3, :3]
        cam = cam[cam[:, 2] > 0]
        uv = cam @ K.T
        uv = np.floor(uv[:, :2] / uv[:, 2:] + .5).astype(np.int64)
        valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        np.minimum.at(zbuf, uv[valid, 1] * w + uv[valid, 0], cam[valid, 2])
    zbuf[~np.isfinite(zbuf)] = 0
    return zbuf.reshape(h, w)


def visible_overlap(fa, fb, out, K):
    da = np.load(out / fa["id"] / "laser_depth.npy")
    db = np.load(out / fb["id"] / "laser_depth.npy")
    y, x = np.mgrid[0:da.shape[0]:8, 0:da.shape[1]:8]
    z = da[y, x].ravel()
    pixels = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)])
    valid = z > 0
    cam = (pixels[valid] @ np.linalg.inv(K).T) * z[valid, None]
    Ta, Tb = np.array(fa["c2w"]), np.array(fb["c2w"])
    world = cam @ Ta[:3, :3].T + Ta[:3, 3]
    other = (world - Tb[:3, 3]) @ Tb[:3, :3]
    front = other[:, 2] > 0
    other = other[front]
    uv = other @ K.T
    uv = np.floor(uv[:, :2] / uv[:, 2:] + .5).astype(int)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < db.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < db.shape[0])
    ref = db[uv[inside, 1], uv[inside, 0]]
    visible = (ref > 0) & (np.abs(ref-other[inside, 2]) < np.maximum(.05, .02*ref))
    return float(visible.sum() / max(1, valid.sum()))


def connected_after_any_drop(ids, overlaps, threshold=.1):
    for removed in ids:
        remaining = set(ids) - {removed}
        seen, frontier = set(), [min(remaining)]
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(other for other in remaining-seen
                            if overlaps[frozenset((current, other))] >= threshold)
        if seen != remaining:
            return False
    return True


def select_nested_sets(frames, descriptors):
    """GT-only selection before inference: every DROP retains an overlap-connected graph."""
    overlaps = {frozenset((p["a"], p["b"])): p["laser_visibility_overlap"] for p in descriptors}
    ids = [f["id"] for f in frames]
    if not connected_after_any_drop(ids, overlaps):
        raise ValueError("20-view selection is not robustly overlap-connected; choose a different scene/set")
    sets = {"20": ids.copy()}
    while len(ids) > 5:
        candidates = []
        for removed in ids:
            kept = [i for i in ids if i != removed]
            if connected_after_any_drop(kept, overlaps):
                average_overlap = np.mean([overlaps[frozenset((removed, other))] for other in kept])
                candidates.append((average_overlap, removed, kept))
        if not candidates:
            raise ValueError("Cannot build a robustly connected nested subset")
        _, _, ids = min(candidates, key=lambda item: (item[0], item[1]))
        if len(ids) in (10, 5):
            sets[str(len(ids))] = ids.copy()
    return sets


def prepare(scene_name="office"):
    scene = ROOT / "data/prepared/eth3d" / scene_name
    out = ROOT / "data/experiments" / scene_name
    out.mkdir(parents=True, exist_ok=True)
    K, original_size, frames = read_calibration(scene)
    if len(frames) < 20:
        raise ValueError("Need at least 20 calibrated images")
    frames = [frames[i] for i in np.rint(np.linspace(0, len(frames) - 1, 20)).astype(int)]
    sets = {"20": [f["id"] for f in frames]}
    sets["10"], sets["5"] = sets["20"][::2], sets["20"][::4]
    # Overscan is retained during integration; the same fixed crop is applied to every clean/blur image.
    ow, oh = original_size
    sw, sh = 1120, round(oh * 1120 / ow)
    w, h = 1008, 672
    left, top = (sw - w) // 2, (sh - h) // 2
    Ksource = np.diag([sw / ow, sh / oh, 1.]) @ K
    Kcrop = Ksource.copy()
    Kcrop[0, 2] -= left
    Kcrop[1, 2] -= top
    print("Loading aligned laser scans", flush=True)
    points = laser_points(scene)
    print(f"Laser points: {len(points):,}", flush=True)
    contact = Image.new("RGB", (5 * 252, 4 * 188), "white")
    for j, frame in enumerate(frames):
        source = scene / "images/dslr_images_undistorted" / frame["filename"]
        rgb = np.asarray(Image.open(source).convert("RGB").resize((sw, sh), Image.Resampling.LANCZOS))
        folder = out / frame["id"]
        folder.mkdir(exist_ok=True)
        frame.update({"source": str(source.relative_to(ROOT)), "source_sha256": digest(source),
                      "intrinsics": Kcrop.tolist(), "variants": {}})
        for severity in SEVERITIES:
            blurred, support = exposure(rgb, Ksource, severity)
            if severity and np.min(support[top:top+h, left:left+w]) < .999:
                raise ValueError("Insufficient overscan: blur would introduce synthetic borders")
            path = folder / f"yaw_{severity:g}.png"
            Image.fromarray(blurred[top:top+h, left:left+w]).save(path)
            frame["variants"][str(severity)] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
        depth = rasterize(points, np.array(frame["c2w"]), Kcrop, (h, w))
        path = folder / "laser_depth.npy"
        np.save(path, depth)
        frame["depth"] = str(path.relative_to(ROOT))
        frame["depth_sha256"] = digest(path)
        frame["laser_coverage"] = float(np.mean(depth > 0))
        thumb = Image.open(folder / "yaw_0.png").resize((252, 168))
        contact.paste(thumb, ((j % 5) * 252, (j // 5) * 188))
        ImageDraw.Draw(contact).text(((j % 5) * 252 + 4, (j // 5) * 188 + 169), frame["id"], fill="black")
        print(f"Prepared {j+1}/20: {frame['id']}, laser coverage {frame['laser_coverage']:.1%}", flush=True)
    contact.save(out / "selected_views.png")
    manifest = {"scene": scene_name, "dataset": "ETH3D high-res multi-view training", "seed": 2026,
                "sets": sets, "severities_deg": SEVERITIES, "frames": frames,
                "blur": {"axis": "camera y", "path": "symmetric constant angular velocity",
                         "samples": 33, "integration": "linear sRGB radiance", "size": [w, h],
                         "nominal_pose": "mid exposure", "fixed_overscan_crop": [left, top, w, h]},
                "geometry_reference": "Aligned ETH3D laser scans rasterized by nearest-depth z-buffer; no benchmark occlusion masks",
                "planned_utilities_per_backend": sum(map(len, sets.values())) * len(SEVERITIES)}
    # Predeclare graph, translation/rotation ranges and laser visibility overlap before any reconstruction.
    descriptors = []
    for a, fa in enumerate(frames):
        Ta = np.array(fa["c2w"])
        for fb in frames[a+1:]:
            Tb = np.array(fb["c2w"])
            descriptors.append({"a": fa["id"], "b": fb["id"],
                                "baseline_m": float(np.linalg.norm(Ta[:3, 3]-Tb[:3, 3])),
                                "rotation_deg": float(np.rad2deg(Rotation.from_matrix(Ta[:3, :3].T @ Tb[:3, :3]).magnitude())),
                                "laser_visibility_overlap": .5*(visible_overlap(fa, fb, out, Kcrop) + visible_overlap(fb, fa, out, Kcrop))})
    manifest["pair_geometry"] = descriptors
    manifest["sets"] = select_nested_sets(frames, descriptors)
    manifest["selection"] = {"max_set": "20 uniformly spaced filenames among 26 calibrated views",
                             "nested_sets": "Greedily remove the lowest mean-overlap view while preserving connectivity after every single-view DROP",
                             "overlap_edge_threshold": .1, "uses_model_outputs": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    first = frames[0]
    ladder = Image.new("RGB", (6*252, 188), "white")
    for j, b in enumerate(SEVERITIES):
        path = ROOT / first["variants"][str(b)]["path"]
        ladder.paste(Image.open(path).resize((252, 168)), (j*252, 0))
        ImageDraw.Draw(ladder).text((j*252+4, 169), f"{b:g} deg exposure path", fill="black")
    ladder.save(out / "blur_ladder.png")
    return out / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="office")
    print(prepare(parser.parse_args().scene))
