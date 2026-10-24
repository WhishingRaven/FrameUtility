"""Scientific invariants: gauge alignment, unchanged inputs, fixed masks, utility signs."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from metrics import METRICS, evaluate, sim3, utility


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["a", "b", "c", "d"]
        self.poses = np.repeat(np.eye(4)[None], 4, axis=0)
        self.poses[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, .3]]
        self.refs = {i: np.full((12, 16), 3.) for i in self.ids}

    def test_sim3_recovers_gauge(self):
        R = Rotation.from_euler("xyz", [20, 10, -30], degrees=True).as_matrix()
        x = self.poses[:, :3, 3]
        y = 2.5*x @ R.T + [5, 1, -3]
        s, r, t = sim3(x, y)
        np.testing.assert_allclose(s*x @ r.T+t, y, atol=1e-10)

    def test_perfect_reconstruction_has_zero_error(self):
        pred = {"ids": self.ids, "c2w": self.poses, "depth": np.ones((4, 12, 16))*6}
        values = evaluate(pred, self.ids, dict(zip(self.ids, self.poses)), self.refs)
        for metric in METRICS:
            self.assertAlmostEqual(values[metric], 0., places=10)

    def test_target_is_excluded_from_evaluation(self):
        poses = np.concatenate([self.poses, np.full((1, 4, 4), np.nan)])
        depth = np.concatenate([np.ones((4, 12, 16))*3, np.full((1, 12, 16), np.nan)])
        pred = {"ids": self.ids+["target"], "c2w": poses, "depth": depth}
        values = evaluate(pred, self.ids, dict(zip(self.ids, self.poses)), self.refs)
        self.assertAlmostEqual(values["geometry_depth_absrel"], 0.)

    def test_invalid_predictions_are_not_silently_filtered(self):
        depth = np.ones((4, 12, 16))*3
        depth[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid predicted depth"):
            evaluate({"ids": self.ids, "c2w": self.poses, "depth": depth}, self.ids,
                     dict(zip(self.ids, self.poses)), self.refs)

    def test_utility_sign_and_zero_denominator(self):
        self.assertEqual(utility(2., 1.), (1., .5))
        self.assertEqual(utility(2., 3.), (-1., -.5))
        self.assertEqual(utility(0., 0.), (0., None))


if __name__ == "__main__":
    unittest.main()
