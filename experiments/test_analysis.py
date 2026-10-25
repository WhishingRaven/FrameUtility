"""Meaningful boundary, missingness, provenance and aggregation checks."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from filelock import FileLock, Timeout

from analysis.audit import diagnostic, expected_predictions, runtime_summary
from analysis.core import (PRIMARY, classify, cohort_summaries, normalize_rows,
                           representative_ids, transition_details, curve_summaries, transition_summaries)
from analysis.pipeline import analyze, publish


class AnalysisRulesTests(unittest.TestCase):
    def test_effect_floor_boundaries_are_neutral(self):
        for value in (-.0005, 0, .0005):
            self.assertEqual(classify(value, .0005), "within_threshold")
        self.assertEqual(classify(-.00050001, .0005), "harmful")
        self.assertEqual(classify(.00050001, .0005), "helpful")
        self.assertEqual(classify(None, .0005), "unavailable")

    def test_crossing_uses_last_positive_and_retains_neutral_and_missing_interior(self):
        result = transition_details([0, .025, .05, .1, 4], [.01, .002, 0, None, -.01], .0005)
        bracket = result["first_positive_to_negative_bracket"]
        self.assertEqual([bracket["lower_deg"], bracket["upper_deg"]], [.025, 4])
        self.assertEqual(bracket["intermediate_within_threshold"], [.05])
        self.assertEqual(bracket["intermediate_unavailable"], [.1])
        self.assertTrue(bracket["interior_path_unresolved"])

    def test_reversals_are_all_retained_and_initial_harm_is_not_clean_positive_transition(self):
        result = transition_details([0, 1, 2, 3], [1, -1, 1, -1], 0)
        self.assertEqual([e["direction"] for e in result["events"]],
                         ["positive_to_negative", "negative_to_positive", "positive_to_negative"])
        self.assertTrue(result["returns_positive_after_first_negative"])
        result = transition_details([0, 1, 2], [-1, 1, -1], 0)
        self.assertFalse(result["observed_clean_positive_to_negative"])
        self.assertEqual(len(result["events"]), 2)

    def test_no_observed_harm_does_not_create_extrapolated_threshold(self):
        result = transition_details([0, .1, 4], [1, 0, 1], 0)
        self.assertIsNone(result["first_positive_to_negative_bracket"])
        self.assertFalse(result["any_harm_observed"])

    def test_incomplete_observed_crossing_is_not_counted_as_complete_curve_transition(self):
        rows = [{"scene": "s", "backend": "vggt", "size": 1, "target": "a", "metric": PRIMARY,
                 "blur_deg": b, "utility_raw": u} for b, u in zip((0, 1, 4), (.01, None, -.01))]
        protocol = {"blur_degrees": [0, 1, 4], "evaluation": {"primary_absrel_effect_floor": .0005,
                    "sensitivity_absrel_effect_floors": [0, .0005]}}
        curves = curve_summaries(rows, protocol)
        result = next(r for r in transition_summaries(curves) if r["policy"] == "threshold")
        self.assertEqual(result["planned_curves"], 1)
        self.assertEqual(result["transitions_complete"], 0)
        self.assertEqual(result["observed_transitions_incomplete"], 1)
        self.assertEqual(result["incomplete_curves"], 1)

    def fixture(self):
        manifest = {"sets": {"2": ["a", "b"]}}
        protocol = {"sizes": [2], "blur_degrees": [0, 4], "evaluation": {"primary_absrel_effect_floor": .0005}}
        row = {"scene": "s", "backend": "vggt", "size": 2, "n_context": 1, "target": "a", "blur_deg": 0,
               "metric": PRIMARY, "status": "ok", "E_drop": .1, "E_add": .09, "utility_raw": .01, "utility": .1}
        return manifest, protocol, row

    def test_missing_metrics_are_retained_and_relative_utility_is_not_used(self):
        manifest, protocol, row = self.fixture()
        rows = normalize_rows([row], "s", "vggt", manifest, protocol)
        self.assertEqual(len(rows), 2 * 2 * 5)
        self.assertEqual(sum(r["status"] == "missing" for r in rows), 19)
        actual = next(r for r in rows if r["status"] == "ok")
        self.assertAlmostEqual(actual["utility_raw"], .01)
        self.assertNotEqual(actual["utility_raw"], actual["utility"])

    def test_duplicate_and_incorrect_U_cannot_be_published(self):
        manifest, protocol, row = self.fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_rows([row, row], "s", "vggt", manifest, protocol)
        with self.assertRaisesRegex(ValueError, "DROP minus ADD"):
            normalize_rows([{**row, "utility_raw": .2}], "s", "vggt", manifest, protocol)

    def test_cohort_uses_equal_scene_weights_and_reports_unobserved_scene(self):
        def row(scene, planned, harmful):
            output = {"scene": scene, "backend": "vggt", "size": 5, "metric": PRIMARY, "blur_deg": 4,
                      "epsilon": .0005, "planned": planned, "valid": planned, "harmful": harmful}
            for category in ("helpful", "harmful", "within_threshold", "unavailable"):
                output[category + "_fraction_planned"] = harmful / planned if category == "harmful" else 0
            for field in ("utility_raw", "E_drop", "E_add"):
                output[field] = {"mean": .1, "median": .1}
            return output
        info = {s: {"device": "mps", "runtime_group": "test"} for s in ("a", "b")}
        result = cohort_summaries([row("a", 5, 5), row("b", 10, 0)], info, ["a", "b", "pending"])
        main = next(r for r in result if r["stratum"] == "all")
        self.assertEqual(main["harmful_scene_fraction"]["mean"], .5)
        self.assertAlmostEqual(main["pooled_targets_secondary"]["harmful_fraction_planned"], 1 / 3)
        self.assertEqual(main["n_scenes_used"], 2)
        self.assertEqual(main["n_scenes_requested"], 3)

    def test_reconstruction_inventory_matches_approved_5_10_15_design(self):
        manifest = {"sets": {str(m): list(map(str, range(m))) for m in (5, 10, 15)}}
        protocol = {"sizes": [5, 10, 15], "blur_degrees": [0, .025, .05, .075, .1, .15, .2, .25, .5, 1, 2, 4]}
        expected = expected_predictions(manifest, protocol)
        self.assertEqual(len(expected), 363)
        self.assertEqual(len(expected["m15_drop_0"]["ids"]), 14)
        self.assertEqual(expected["m5_clean"]["targets"], manifest["sets"]["5"])

    def test_diagnostics_count_signed_depth_and_nonfinite_on_fixed_mask_only(self):
        prediction = {"ids": ["a"], "depth_common_grid": np.array([[[-1., np.nan], [0., np.inf]]])}
        result = diagnostic(prediction, ["a"], {"a": np.array([[True, True], [True, False]])})
        self.assertEqual(result["evaluation_pixels"], 3)
        self.assertEqual(result["nonpositive_pixels"], 2)
        self.assertEqual(result["nonfinite_pixels"], 1)
        self.assertFalse(result["strict_positive_depth_valid"])

    def test_parallel_times_are_not_summed_as_elapsed_time(self):
        runs = [{"start_unix": 0, "seconds": 10, "status": "ok"},
                {"start_unix": 2, "seconds": 3, "status": "ok"}]
        result = runtime_summary(runs)
        self.assertEqual(result["summed_worker_inference_seconds"], 13)
        self.assertEqual(result["inference_interval_union_seconds"], 10)
        self.assertEqual(result["first_to_last_prediction_seconds"], 10)

    def test_representatives_are_fixed_by_name(self):
        self.assertEqual(representative_ids(["e", "b", "d", "c", "a", "a"]), ["a", "c", "e"])

    def test_failed_publication_restores_previous_successful_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination, stage = root / "analysis", root / "stage"
            destination.mkdir()
            stage.mkdir()
            (destination / "report.md").write_text("previous verified report")
            (stage / "report.md").write_text("new report")
            original = Path.rename

            def rename(path, target):
                if path == stage:
                    raise OSError("Simulated interrupted publication")
                return original(path, target)

            with patch.object(Path, "rename", rename), self.assertRaises(OSError):
                publish(stage, destination)
            self.assertEqual((destination / "report.md").read_text(), "previous verified report")

    def test_concurrent_analysis_is_rejected_before_reading_or_writing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with FileLock(str(root / ".analysis.lock")):
                with patch("analysis.pipeline._analyze") as worker:
                    with self.assertRaises(Timeout):
                        analyze(root / "protocol.json", root, ["office"])
                    worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
