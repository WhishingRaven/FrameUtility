"""Scientific controls: unchanged pixels, fixed masks and invalid-depth penalties."""
import unittest

import numpy as np

from followup_metrics import evaluate_followup, fixed_mask_geometry
from metrics import evaluate_geometry
from prepare import exposure
from analyze_followup import crossing_bracket, summarize_values


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["a", "b", "c", "d"]
        self.refs = {frame: np.full((12, 16), 3.) for frame in self.ids}
        self.depth = np.full((4, 12, 16), 6.)
        self.poses = np.repeat(np.eye(4)[None], 4, axis=0)
        self.poses[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, .3]]

    def prediction(self):
        return {"ids": self.ids, "depth": self.depth, "c2w": self.poses}

    def test_zero_motion_runs_full_pipeline_without_pixel_changes(self):
        image = np.random.default_rng(2).integers(0, 256, (32, 48, 3), dtype=np.uint8)
        K = np.array([[40., 0, 24], [0, 40, 16], [0, 0, 1.]])
        control, support = exposure(image, K, 0, force_pipeline=True)
        np.testing.assert_array_equal(control, image)
        np.testing.assert_array_equal(support, np.ones(image.shape[:2]))

    def test_valid_scores_equal_legacy_exactly(self):
        self.depth += np.random.default_rng(3).uniform(-1, 1, self.depth.shape)
        values, diagnostics, error = fixed_mask_geometry(self.prediction(), self.ids, self.refs)
        self.assertIsNone(error)
        self.assertEqual(values, evaluate_geometry(self.prediction(), self.ids, self.refs))
        self.assertEqual(diagnostics["nonpositive_pixels"], 0)

    def test_negative_and_zero_depth_are_scored_on_all_pixels(self):
        self.depth[0, 0, :2] = [-6., 0.]
        values, diagnostics, error = fixed_mask_geometry(self.prediction(), self.ids, self.refs)
        self.assertIsNone(error)
        self.assertEqual(values["evaluation_pixels"], self.depth.size)
        self.assertAlmostEqual(values["geometry_depth_absrel"], 3 / self.depth.size)
        self.assertEqual(diagnostics["nonpositive_pixels"], 2)
        with self.assertRaises(ValueError):
            evaluate_geometry(self.prediction(), self.ids, self.refs)

    def test_nonfinite_depth_fails_geometry_but_keeps_pose(self):
        self.depth[0, 0, 0] = np.nan
        result = evaluate_followup(self.prediction(), self.ids, dict(zip(self.ids, self.poses)), self.refs)
        self.assertIn("geometry", result["errors"])
        self.assertNotIn("pose", result["errors"])
        self.assertEqual(result["depth_diagnostics"]["nonfinite_pixels"], 1)
        self.assertAlmostEqual(result["metrics"]["pose_ate_normalized"], 0.)

    def test_gt_mask_and_target_exclusion_remain_fixed(self):
        self.refs["a"][0, 0] = 0
        self.depth[0, 0, 0] = np.nan
        prediction = self.prediction()
        prediction["ids"] = self.ids + ["target"]
        prediction["depth"] = np.concatenate([self.depth, np.full((1, 12, 16), np.nan)])
        values, diagnostics, error = fixed_mask_geometry(prediction, self.ids, self.refs)
        self.assertIsNone(error)
        self.assertEqual(values["evaluation_pixels"], self.depth.size - 1)
        self.assertEqual(diagnostics["nonfinite_pixels"], 0)

    def test_cross_repeat_combinations_detect_overlap_missed_by_paired_signs(self):
        stats = summarize_values([1, 2, 3], [.5, 1.5, 2.5], 3)
        self.assertEqual(stats["utility_raw_range"], [-1.5, 2.5])
        self.assertFalse(stats["all_combinations_beneficial"])

    def test_incomplete_repeats_cannot_claim_stability(self):
        stats = summarize_values([2, 2], [1, 1], 3)
        self.assertFalse(stats["complete"])
        self.assertFalse(stats["all_combinations_beneficial"])

    def test_effect_floor_distinguishes_tiny_clean_benefit(self):
        clean = summarize_values([1]*3, [.9999]*3, 3, floor=.0005)
        clean["blur_deg"] = 0
        blur = summarize_values([1]*3, [1.01]*3, 3, floor=.0005)
        blur["blur_deg"] = .25
        self.assertEqual(crossing_bracket([clean, blur]), [0, .25])
        self.assertIsNone(crossing_bracket([clean, blur], floor=True))

    def test_crossing_bracket_spans_ambiguous_repeat_stage(self):
        clean = summarize_values([1]*3, [.9]*3, 3)
        middle = summarize_values([1]*3, [.9, 1, 1.1], 3)
        blur = summarize_values([1]*3, [1.1]*3, 3)
        for degree, stats in zip([0, .1, .25], [clean, middle, blur]):
            stats["blur_deg"] = degree
        self.assertEqual(crossing_bracket([clean, middle, blur]), [0, .25])


if __name__ == "__main__":
    unittest.main()
