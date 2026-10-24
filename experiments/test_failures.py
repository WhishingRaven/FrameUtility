"""Regression: invalid geometry must not stop the sweep or erase unrelated outcomes."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from metrics import condition_counts, evaluate_available
import run


class FailureTests(unittest.TestCase):
    def test_geometry_failure_keeps_valid_pose(self):
        ids = ["a", "b", "c", "d"]
        poses = np.repeat(np.eye(4)[None], 4, axis=0)
        poses[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, .3]]
        depth = np.full((4, 12, 16), 3.)
        depth[0, 0, 0] = -1
        values, errors = evaluate_available({"ids": ids, "c2w": poses, "depth": depth}, ids,
                                            dict(zip(ids, poses)), {i: np.full((12, 16), 3.) for i in ids})
        self.assertEqual(set(errors), {"geometry"})
        self.assertIn("pose_ate_normalized", values)
        self.assertNotIn("geometry_depth_absrel", values)

    def test_sweep_survives_bad_repeat_and_drop_and_keeps_failures_on_resume(self):
        ids = list("abcde")
        poses = np.repeat(np.eye(4)[None], 5, axis=0)
        poses[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, .3], [.2, .5, .4]]
        gt = dict(zip(ids, poses))
        calls = []

        class FakeBackend:
            def predict(self, selected, paths, key):
                calls.append(key)
                depth = np.full((len(selected), 12, 16), 3.)
                if "b" in selected:
                    depth[selected.index("b"), 0, 0] = -1
                return {"ids": np.array(selected), "c2w": np.array([gt[i] for i in selected]), "depth": depth}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.bin"
            image.write_bytes(b"test fixture; never used for model inference")
            reference = root / "depth.npy"
            np.save(reference, np.full((12, 16), 3.))
            frames = [{"id": i, "c2w": gt[i].tolist(), "depth": reference.name,
                       "depth_sha256": run.sha256(reference),
                       "variants": {str(b): {"path": image.name, "sha256": run.sha256(image)} for b in [0, .25]}}
                      for i in ids]
            manifest = {"scene": "test_fixture", "sets": {"5": ids}, "frames": frames, "severities_deg": [0, .25]}
            (root / "manifest.json").write_text(json.dumps(manifest))
            args = SimpleNamespace(manifest="manifest.json", output="results", backend="mast3r", device="cpu",
                                   sizes=[5], targets=None, repeat_clean=True, max_new_runs=None)
            with patch.object(run, "ROOT", root), patch.object(run, "environment", return_value={"test": True}), \
                    patch.object(run, "reference_depth", side_effect=lambda path, backend: np.load(path)), \
                    patch.object(run, "load_backend", return_value=FakeBackend()), contextlib.redirect_stdout(io.StringIO()):
                run.run(args)
                path = root / "results/mast3r/utilities.json"
                rows = json.loads(path.read_text())
                self.assertEqual(condition_counts(rows), {"attempted": 10, "ok": 2, "partial": 8, "failed": 0})
                self.assertTrue(all(r["status"] == "ok" for r in rows if r["metric"].startswith("pose_")))
                # Preserve a failure belonging to a different phase when resuming this phase.
                for metric in run.METRICS:
                    rows.append({"backend": "mast3r", "scene": "test_fixture", "size": 10, "target": "z",
                                 "blur_deg": 0, "metric": metric, "status": "failed", "error": "prior outcome"})
                path.write_text(json.dumps(rows))
                calls.clear()
                run.run(args)
                resumed = json.loads(path.read_text())
            self.assertEqual(calls, [])  # completed predictions reused; no rerun of inference
            self.assertEqual(len(resumed), 55)
            self.assertEqual(sum(r["size"] == 10 and r["status"] == "failed" for r in resumed), 5)
            self.assertEqual(len({(r["size"], r["target"], r["blur_deg"], r["metric"]) for r in resumed}), 55)


if __name__ == "__main__":
    unittest.main()
