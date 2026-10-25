"""Official VGGT and MASt3R-SfM, with explicit inputs and independent optimization."""
import gc
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def project_depth_to_common_grid(prediction, backend, source_size, evaluation_roi, grid_size):
    """Sample model depth at the protocol's shared prepared-image pixel centers."""
    source_width, source_height = map(int, source_size)
    roi_x, roi_y, roi_width, roi_height = map(int, evaluation_roi)
    grid_width, grid_height = map(int, grid_size)
    if (roi_width % grid_width or roi_height % grid_height):
        raise ValueError("Evaluation ROI must divide evenly into the fixed grid")

    if backend == "vggt":
        output_width = 518
        output_height = round(source_height * output_width / source_width / 14) * 14
        crop_left = 0
        crop_top = max(0, (output_height - 518) // 2)
        expected_height = min(output_height, 518)
        expected_width = output_width
    elif backend == "mast3r":
        scale = 512 / max(source_width, source_height)
        resized_width = round(source_width * scale)
        resized_height = round(source_height * scale)
        center_x, center_y = resized_width // 2, resized_height // 2
        half_width = (2 * center_x) // 16 * 8
        half_height = (2 * center_y) // 16 * 8
        crop_left, crop_top = center_x - half_width, center_y - half_height
        output_width, output_height = resized_width, resized_height
        expected_width, expected_height = half_width * 2, half_height * 2
    else:
        raise ValueError(f"Unknown backend: {backend}")

    depth = np.asarray(prediction["depth"])
    if depth.ndim != 3 or depth.shape[1:] != (expected_height, expected_width):
        raise ValueError(
            f"{backend} depth shape {depth.shape} does not match its recorded preprocessing "
            f"transform {(expected_height, expected_width)}"
        )

    # Grid coordinates are corner-origin pixel centers; resizing uses the same
    # half-pixel convention as PIL, then the model's exact center crop is removed.
    grid_x = roi_x + (np.arange(grid_width, dtype=np.float64) + 0.5) * (roi_width / grid_width)
    grid_y = roi_y + (np.arange(grid_height, dtype=np.float64) + 0.5) * (roi_height / grid_height)
    sample_x = np.floor(grid_x * output_width / source_width).astype(np.intp) - crop_left
    sample_y = np.floor(grid_y * output_height / source_height).astype(np.intp) - crop_top
    if (sample_x.min() < 0 or sample_x.max() >= expected_width or
            sample_y.min() < 0 or sample_y.max() >= expected_height):
        raise ValueError(f"{backend} preprocessing crop does not contain the full common evaluation grid")
    return depth[:, sample_y[:, None], sample_x[None, :]]


def reference_depth(path, backend):
    ref = np.load(path)
    h, w = ref.shape
    if backend == "vggt":
        width, height = 518, round(h * 518 / w / 14) * 14
        ref = cv2.resize(ref, (width, height), interpolation=cv2.INTER_NEAREST_EXACT)
        if height > 518:
            top = (height - 518) // 2
            ref = ref[top:top+518]
    else:
        width, height = round(w * 512 / max(h, w)), round(h * 512 / max(h, w))
        ref = cv2.resize(ref, (width, height), interpolation=cv2.INTER_NEAREST_EXACT)
        cx, cy = width // 2, height // 2
        hw, hh = (2*cx)//16*8, (2*cy)//16*8
        ref = ref[cy-hh:cy+hh, cx-hw:cx+hw]
    return ref


class VGGTBackend:
    def __init__(self, device, cache_root):
        sys.path.insert(0, str(ROOT / "vendor/vggt"))
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        self.device = device
        self.preprocess = load_and_preprocess_images
        self.decode_pose = pose_encoding_to_extri_intri
        self.model = VGGT(enable_point=False, enable_track=False).eval()
        state = torch.load(ROOT / "checkpoints/vggt-1b.pt", map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if not k.startswith(("point_head.", "track_head."))}
        self.model.load_state_dict(state, strict=True)
        del state
        gc.collect()
        self.model = self.model.to(device)

    def predict(self, ids, paths, key):
        images = self.preprocess(paths).to(self.device)
        with torch.inference_mode():
            output = self.model(images)
            extrinsics, intrinsics = self.decode_pose(output["pose_enc"], images.shape[-2:])
        w2c = np.broadcast_to(np.eye(4), (len(ids), 4, 4)).copy()
        w2c[:, :3] = extrinsics[0].cpu().numpy()
        result = {"ids": np.array(ids), "c2w": np.linalg.inv(w2c),
                  "depth": output["depth"][0, ..., 0].cpu().numpy(),
                  "intrinsics": intrinsics[0].cpu().numpy()}
        del output, images
        return result


class MASt3RBackend:
    def __init__(self, device, cache_root, optimizer_device=None, pair_cache=None):
        sys.path.insert(0, str(ROOT / "vendor/mast3r"))
        from mast3r.model import AsymmetricMASt3R
        from safetensors.torch import load_model
        self.device = device
        self.optimizer_device = optimizer_device or device
        config = json.loads((ROOT / "checkpoints/mast3r-config.json").read_text())
        self.model = AsymmetricMASt3R(**config).eval()
        load_model(self.model, str(ROOT / "checkpoints/mast3r.safetensors"), strict=True)
        self.model = self.model.to(device)
        self.cache = Path(cache_root)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.pair_cache = Path(pair_cache) if pair_cache is not None else self.cache / "pairs"

    def predict(self, ids, paths, key):
        from dust3r.utils.image import load_images
        from mast3r.image_pairs import make_pairs
        from mast3r.cloud_opt.sparse_ga import (convert_dust3r_pairs_naming, forward_mast3r,
                                               sparse_global_alignment)
        images = load_images(paths, size=512, verbose=False)
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        pairs = convert_dust3r_pairs_naming(paths, pairs)
        cache = self.cache / key
        if cache.exists():
            shutil.rmtree(cache)
        cache.mkdir(exist_ok=True)
        pair_root = self.pair_cache.with_name(self.pair_cache.name + "_repeat") if key.endswith("_repeat") else self.pair_cache
        # Only pairwise inference is shared. Canonical maps, spanning tree, poses and
        # optimization state MUST be recomputed for the exact ADD/DROP input set.
        for name in ("forward", "corres_conf=desc_conf_subsample=8"):
            (pair_root / name).mkdir(parents=True, exist_ok=True)
            link = cache / name
            if not link.exists():
                link.symlink_to((pair_root / name).resolve(), target_is_directory=True)
        forward_mast3r(pairs, self.model, str(cache), device=self.device, subsample=8)
        # Inference and optimization devices are explicit and independently recorded.
        # torch.load in upstream reads trusted cache files written above (including numpy arrays).
        scene = sparse_global_alignment(paths, pairs, str(cache), None, device=self.optimizer_device, subsample=8,
                                        lr1=.07, niter1=300, lr2=.01, niter2=300,
                                        matching_conf_thr=5., shared_intrinsics=False, verbose=False)
        _, depths, _ = scene.get_dense_pts3d(clean_depth=False)
        shape = images[0]["img"].shape[-2:]
        result = {"ids": np.array(ids), "c2w": scene.get_im_poses().detach().cpu().numpy(),
                  "depth": np.stack([d.detach().cpu().numpy().reshape(shape) for d in depths]),
                  "intrinsics": scene.intrinsics.detach().cpu().numpy()}
        del scene, depths, images, pairs
        shutil.rmtree(cache)
        return result


def load_backend(name, device, cache_root, optimizer_device=None, pair_cache=None):
    if name == "vggt":
        return VGGTBackend(device, cache_root)
    if name == "mast3r":
        return MASt3RBackend(device, cache_root, optimizer_device, pair_cache)
    raise ValueError(name)
