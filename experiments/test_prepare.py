import json
import unittest

import numpy as np

from prepare import connected_after_any_drop, exposure, read_calibration, ROOT


class PreparationTests(unittest.TestCase):
    def test_clean_pixels_are_unchanged(self):
        image = np.random.default_rng(42).integers(0, 256, (32, 48, 3), dtype=np.uint8)
        out, _ = exposure(image, np.eye(3), 0)
        np.testing.assert_array_equal(out, image)

    def test_rotation_preserves_constant_radiance_inside_support(self):
        image = np.full((60, 90, 3), 128, np.uint8)
        K = np.array([[65., 0, 45], [0, 65, 30], [0, 0, 1]])
        out, mask = exposure(image, K, 4)
        np.testing.assert_array_equal(out[mask > .999], image[mask > .999])
        self.assertGreater(np.mean(mask > .999), .8)

    def test_calibration_parser_keeps_all_26_office_views(self):
        scene = ROOT / "data/prepared/eth3d/office"
        if not scene.exists():
            self.skipTest("ETH3D office has not been extracted")
        K, size, frames = read_calibration(scene)
        self.assertEqual(len(frames), 26)
        self.assertEqual(len({f["id"] for f in frames}), 26)
        self.assertEqual(size, (6221, 4146))
        self.assertGreater(K[0, 0], 0)

    def test_frozen_sets_remain_connected_after_each_drop(self):
        path = ROOT / "data/experiments/office/manifest.json"
        if not path.exists():
            self.skipTest("Frozen inputs have not been prepared")
        manifest = json.loads(path.read_text())
        overlaps = {frozenset((p["a"], p["b"])): p["laser_visibility_overlap"]
                    for p in manifest["pair_geometry"]}
        self.assertLess(set(manifest["sets"]["5"]), set(manifest["sets"]["10"]))
        self.assertLess(set(manifest["sets"]["10"]), set(manifest["sets"]["20"]))
        for size, ids in manifest["sets"].items():
            self.assertEqual(len(ids), int(size))
            self.assertTrue(connected_after_any_drop(ids, overlaps))


if __name__ == "__main__":
    unittest.main()
