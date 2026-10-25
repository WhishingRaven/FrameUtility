"""Geometry, shared-view rendering, report preservation and figure reuse checks."""
import hashlib
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from analysis.audit import Sources
from analysis.pipeline import identical_files, reusable_figures
from analysis.reconstruction import (unproject, align_cloud, render_cloud, backend_image, sample_native_grid,
                                     make_reconstruction_figures)
from analysis.report import (insert_reconstruction_section, RECONSTRUCTION_START, RECONSTRUCTION_END)


class ReconstructionTests(unittest.TestCase):
    def test_backprojection_uses_depth_intrinsics_and_full_camera_pose(self):
        intrinsic = np.array([[2., 0, 1], [0, 4, 2], [0, 0, 1]])
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        pose[:3, 3] = [10, 20, 30]
        points = unproject(np.array([2., 4.]), intrinsic, pose, np.array([1, 3]), np.array([2, 6]))
        expected = np.array([[0, 0, 2], [4, 4, 4]]) @ pose[:3, :3].T + [10, 20, 30]
        np.testing.assert_allclose(points, expected, atol=1e-12)

    def test_similarity_alignment_preserves_shape_and_recovers_common_world(self):
        centers = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]])
        rotation = Rotation.from_euler("xyz", [10, -20, 25], degrees=True).as_matrix()
        target = 3 * centers @ rotation.T + [2, -1, 4]
        ids = list("abcd")
        predicted, gt = {}, {}
        for index, image_id in enumerate(ids):
            predicted[image_id] = np.eye(4)
            predicted[image_id][:3, 3] = centers[index]
            gt[image_id] = np.eye(4)
            gt[image_id][:3, 3] = target[index]
        cloud = np.array([[.2, .3, .4], [2., 3, 4]])
        aligned, metadata = align_cloud(cloud, predicted, gt, ids)
        np.testing.assert_allclose(aligned, 3 * cloud @ rotation.T + [2, -1, 4], atol=1e-12)
        self.assertAlmostEqual(metadata["scale"], 3)
        self.assertEqual(metadata["camera_ids"], ids)

    def test_z_buffer_shows_nearer_rgb_point_independent_of_input_order(self):
        view = {"width": 9, "height": 9, "c2w": np.eye(4).tolist(),
                "intrinsics": [[2., 0, 4], [0, 2, 4], [0, 0, 1]],
                "near_m": .1, "far_m": 10, "point_diameter_px": 1}
        points = np.array([[0., 0, 2], [0, 0, 1], [0, 0, -1], [100, 0, 1]])
        colors = np.array([[0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0]], np.uint8)
        image, metadata = render_cloud(points, colors, view)
        reversed_image, _ = render_cloud(points[::-1], colors[::-1], view)
        np.testing.assert_array_equal(image, reversed_image)
        np.testing.assert_array_equal(image[4, 4], [255, 0, 0])
        self.assertEqual(metadata["rendered_pixels"], 1)

    def test_native_rgb_preprocessing_and_common_grid_shapes_for_both_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.png"
            y, x = np.indices((672, 1008))
            Image.fromarray(np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)).save(path)
            prep = {"evaluation_roi": [0, 16, 1008, 640], "evaluation_grid": [252, 160]}
            for backend, shape, offset in (("vggt", (350, 518), [0, 0]), ("mast3r", (336, 512), [0, 2])):
                rgb, transform = backend_image(path, backend, shape)
                self.assertEqual(rgb.shape, (*shape, 3))
                self.assertEqual(transform["crop_offset"], offset)
                u, v = sample_native_grid(transform, prep)
                self.assertEqual(u.shape, (160, 252))
                self.assertTrue((u >= 0).all() and (v >= 0).all())
                self.assertTrue((u < shape[1]).all() and (v < shape[0]).all())

    def test_report_insertion_is_idempotent_and_retains_authored_depth_explanation(self):
        original = "# 보고서\n\n직접 추가한 depth 설명과 숫자.\n\n## 선정 사진과 정답 표본 수\n\n기존 표.\n"
        info = {"status": "generated", "size": 15, "reference_image_id": "photo07",
                "shared_virtual_camera": {"horizontal_fov_deg": 70, "point_diameter_px": 2},
                "contrast": {"status": "generated", "target": "photo07", "blur_deg": 4,
                             "common_image_ids": [str(i) for i in range(14)]}}
        added = insert_reconstruction_section(original, info)
        self.assertEqual(added, insert_reconstruction_section(added, info))
        pattern = re.escape(RECONSTRUCTION_START) + r"[\s\S]*?" + re.escape(RECONSTRUCTION_END) + r"\n\n"
        self.assertEqual(re.sub(pattern, "", added), original)
        self.assertIn("reconstruction_comparison.png", added)
        self.assertIn("reconstruction_conditions.png", added)
        self.assertLess(added.index(RECONSTRUCTION_START), added.index("## 선정 사진"))

    def test_another_scene_automatically_renders_both_figures_with_one_shared_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execution = root / "executions" / "atrium_fixture"
            folder = root / "figures"
            ids = ["camera_" + suffix for suffix in "abcde"]
            target = ids[2]
            common = [image_id for image_id in ids if image_id != target]
            preparation = {"evaluation_roi": [0, 16, 1008, 640], "evaluation_grid": [12, 8]}
            protocol = {"sizes": [3, 5], "blur_degrees": [0, .25, 2.5], "preparation": preparation}
            intrinsic = np.array([[700., 0, 504], [0, 700, 336], [0, 0, 1]])
            frames, poses = [], {}
            for index, image_id in enumerate(ids):
                image_path = root / (image_id + ".png")
                Image.new("RGB", (1008, 672), (40 + index * 30, 100, 180)).save(image_path)
                pose = np.eye(4)
                pose[:3, 3] = [.1 * (index % 3), .1 * (index // 3), 0]
                poses[image_id] = pose
                frames.append({"id": image_id, "c2w": pose.tolist(), "intrinsics": intrinsic.tolist(),
                               "variants": {"0": {"path": image_path.name,
                                   "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()}}})
            data = {"info": {"scene": "atrium_fixture"},
                    "manifest": {"sets": {"3": ids[:3], "5": ids}, "frames": frames,
                                 "preparation": preparation},
                    "gt_depths": {target: np.full((8, 12), 2.)},
                    "gt_masks": {target: np.ones((8, 12), bool)}}
            for backend, shape in (("vggt", (350, 518)), ("mast3r", (336, 512))):
                predictions = execution / backend / "predictions"
                predictions.mkdir(parents=True)
                native_intrinsic = np.array([[300., 0, shape[1] / 2], [0, 300, shape[0] / 2], [0, 0, 1]])
                for key, selected in (("m5_clean", ids), (f"m5_drop_{target}", common),
                                      (f"m5_add_{target}_b2.5", ids)):
                    np.savez_compressed(predictions / (key + ".npz"), ids=np.array(selected),
                        c2w=np.array([poses[i] for i in selected]),
                        intrinsics=np.repeat(native_intrinsic[None], len(selected), axis=0),
                        depth=np.full((len(selected), *shape), 2., np.float32),
                        depth_common_grid=np.full((len(selected), 8, 12), 2., np.float32))
            with patch("analysis.reconstruction.render_cloud", wraps=render_cloud) as renderer:
                info = make_reconstruction_figures(root, execution, data, protocol, folder, Sources(root))
            self.assertEqual(info["scene"], "atrium_fixture")
            self.assertEqual(info["size"], 5)
            self.assertEqual(info["reference_image_id"], target)
            self.assertFalse(info["inference_executed"])
            self.assertEqual(info["contrast"]["status"], "generated")
            self.assertEqual(info["contrast"]["blur_deg"], 2.5)
            self.assertEqual(renderer.call_count, 8)
            self.assertTrue(all(call.args[2] is info["shared_virtual_camera"] for call in renderer.call_args_list))
            for panel in info["contrast"]["panels"].values():
                self.assertEqual(panel["display_image_ids"], common)
                self.assertEqual(panel["alignment"]["camera_ids"], common)
                self.assertEqual(panel["valid_generated_points"], len(common) * 8 * 12)
            for name in ("reconstruction_comparison", "reconstruction_conditions"):
                self.assertTrue((folder / (name + ".pdf")).is_file())
                with Image.open(folder / (name + ".png")) as rendered:
                    rendered.verify()

    def test_old_statistical_figures_are_reused_only_when_hashes_and_code_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "scenes/s/figures/existing.pdf"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"unchanged figure bytes")
            code = {"experiments/analysis/core.py": "core", "experiments/analysis/plots.py": "plots"}
            previous = {"analysis_source_sha256": code,
                        "packages": {"numpy": "1", "matplotlib": "1"},
                        "output_files_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()}}
            with patch("analysis.pipeline.importlib.metadata.version", return_value="1"):
                self.assertTrue(reusable_figures(previous, root, "scenes/s/figures/", code))
                path.write_bytes(b"changed figure")
                self.assertFalse(reusable_figures(previous, root, "scenes/s/figures/", code))
            self.assertFalse(identical_files(root, root, ["missing.csv"]))


if __name__ == "__main__":
    unittest.main()
