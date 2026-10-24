"""Concurrent writers must preserve all conditions and prior failures."""
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from metrics import METRICS
from run import pair_cache_signature, save_condition


def write_conditions(path, worker):
    for index in range(10):
        rows = [{"backend": "mast3r", "size": 5, "target": f"worker{worker}_{index}", "blur_deg": 0,
                 "metric": metric, "status": "ok"} for metric in METRICS]
        save_condition(Path(path), rows)


class ParallelTests(unittest.TestCase):
    def test_shared_cache_accepts_alignment_change_but_rejects_inference_change(self):
        base = {"manifest_sha256": "images-a", "weight_sha256": "weights-a", "model_config_sha256": "config-a",
                "source_commits": {"mast3r": "source-a"}, "packages": {"torch": "test"}, "device": "mps",
                "precision": "float32", "seed": 2026, "float32_matmul_precision": "highest",
                "cudnn_allow_tf32": False, "torch_threads": 8, "mast3r": {"optimizer_device": "cpu"}}
        accelerated = json.loads(json.dumps(base))
        accelerated["mast3r"]["optimizer_device"] = "mps"
        self.assertEqual(pair_cache_signature(base), pair_cache_signature(accelerated))
        for key, value in [("device", "cuda"), ("manifest_sha256", "images-b"),
                           ("weight_sha256", "weights-b"), ("precision", "float16")]:
            incompatible = {**accelerated, key: value}
            self.assertNotEqual(pair_cache_signature(base), pair_cache_signature(incompatible))

    def test_concurrent_processes_keep_all_rows_and_prior_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "utilities.json"
            prior = [{"backend": "mast3r", "size": 10, "target": "prior", "blur_deg": 0,
                      "metric": metric, "status": "failed"} for metric in METRICS]
            path.write_text(json.dumps(prior))
            ctx = multiprocessing.get_context("spawn")
            processes = [ctx.Process(target=write_conditions, args=(str(path), i)) for i in range(3)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)
            rows = json.loads(path.read_text())
            self.assertEqual(len(rows), 155)
            self.assertEqual(sum(row["status"] == "failed" for row in rows), 5)
            self.assertEqual(len({(r["size"], r["target"], r["blur_deg"], r["metric"]) for r in rows}), 155)


if __name__ == "__main__":
    unittest.main()
