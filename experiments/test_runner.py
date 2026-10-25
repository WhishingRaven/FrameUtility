"""GPU defaults, shared entrypoint, approval gate and protocol consistency."""
import json
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import run
import run_all
import run_parallel
from prepare import read_calibration


class RunnerTests(unittest.TestCase):
    def protocol(self):
        return {"status": "approved", "implementation_ready": True, "protocol_id": "fixture",
                "primary_scenes": ["new_scene", "office"], "initial_scenes": ["office"],
                "expansion_scenes": ["new_scene"],
                "seeds": [2026, 2027, 2028], "sizes": [5], "blur_degrees": [0, .025, .25],
                "backends": ["vggt", "mast3r"], "execution": {"workers": 3, "threads_per_worker": 8}}

    def manifest(self):
        return {"protocol_id": "fixture", "scene": "new_scene", "severities_deg": [0, .025, .25],
                "sets": {"5": list("abcde")}}

    def test_entrypoints_share_the_accelerated_driver(self):
        self.assertIs(run_all.run_worker_group, run_parallel.run)
        local = run_all.build_parser().parse_args(["run", "--scene", "office"])
        server = run_all.build_parser().parse_args(["run", "--scene", "courtyard"])
        self.assertEqual(local.device, "auto")
        self.assertEqual(server.device, "auto")
        self.assertIsNone(local.seed)
        self.assertIsNone(server.seed)

    def test_auto_prefers_cuda_then_mps_and_rejects_silent_cpu_fallback(self):
        with patch.object(run.torch.cuda, "is_available", return_value=True):
            self.assertEqual(run.resolve_device("auto"), "cuda")
        with patch.object(run.torch.cuda, "is_available", return_value=False), \
                patch.object(run.torch.backends.mps, "is_available", return_value=True):
            self.assertEqual(run.resolve_device("auto"), "mps")
        with patch.object(run.torch.cuda, "is_available", return_value=False), \
                patch.object(run.torch.backends.mps, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "No GPU"):
                run.resolve_device("auto")
        self.assertEqual(run.resolve_device("cpu"), "cpu")

    def test_draft_gate_runs_before_input_read_gpu_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"protocol.json").write_text(json.dumps({"status": "awaiting_approval"}))
            args = SimpleNamespace(protocol="protocol.json", manifest="missing.json", cohort="outputs",
                                   device="auto", seed=2026)
            with patch.object(run_parallel, "ROOT", root), patch.object(run_parallel, "resolve_device") as gpu:
                with self.assertRaisesRegex(RuntimeError, "approval"):
                    run_parallel.run(args)
                gpu.assert_not_called()
            self.assertFalse((root/"outputs").exists())

    def test_approval_alone_does_not_skip_implementation_readiness(self):
        protocol = self.protocol()
        protocol["implementation_ready"] = False
        with self.assertRaisesRegex(RuntimeError, "implemented"):
            run_parallel.validate_protocol(protocol, self.manifest(), 2026)

    def test_manifest_ladder_scene_and_repeat_must_match_the_protocol(self):
        run_parallel.validate_protocol(self.protocol(), self.manifest(), 2026)
        for changed in ({"protocol_id": "old"}, {"scene": "unapproved"}, {"severities_deg": [0, 4]}):
            with self.assertRaises(ValueError):
                run_parallel.validate_protocol(self.protocol(), {**self.manifest(), **changed}, 2026)
        with self.assertRaises(ValueError):
            run_parallel.validate_protocol(self.protocol(), self.manifest(), 999)

    def test_worker_uses_gpu_alignment_explicit_inputs_and_block_seed(self):
        command = run_parallel.worker_command(Path("manifest.json"), Path("output"), "mast3r", "mps",
                                               2027, 8, [5, 10, 15], Path("stop"), 2, 3)
        for flag, value in (("--device", "mps"), ("--optimizer-device", "mps"), ("--threads", "8"),
                            ("--seed", "2027"), ("--shard-count", "3"), ("--shard-index", "2")):
            self.assertEqual(command[command.index(flag)+1], value)
        self.assertNotIn("--pair-cache-source", command)
        self.assertNotIn("--repeat-clean", command)

    def test_old_preparer_cannot_silently_apply_first_camera_to_all_images(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            calibration = scene/"dslr_calibration_undistorted"
            calibration.mkdir()
            (calibration/"cameras.txt").write_text(
                "0 PINHOLE 640 480 500 500 320 240\n1 PINHOLE 800 600 700 700 400 300\n")
            with self.assertRaisesRegex(ValueError, "CAMERA_ID"):
                read_calibration(scene)

    def test_cohort_lock_rejects_changes_between_scenes_or_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"plan.md").write_text("approved fixture")
            protocol = self.protocol()
            protocol.update(plan="plan.md", approved_plan_sha256=run.sha256(root/"plan.md"))
            (root/"protocol.json").write_text(json.dumps(protocol))
            (root/"manifest.json").write_text(json.dumps({**self.manifest(), "scene": "office"}))
            args = SimpleNamespace(protocol="protocol.json", manifest="manifest.json", cohort="outputs",
                                   device="mps", seed=2026)
            with (
                patch.object(run_parallel, "ROOT", root),
                patch.object(run_parallel, "resolve_device", return_value="mps"),
                patch.object(run_parallel, "cohort_configuration", return_value={"source": "fixed"}) as config,
                patch.object(run_parallel, "execute") as execute,
            ):
                run_parallel.run(args)
                args.seed = 2027
                run_parallel.run(args)
                self.assertEqual(execute.call_count, 2)
                config.return_value = {"source": "changed"}
                with self.assertRaisesRegex(ValueError, "changed"):
                    run_parallel.run(args)
                self.assertEqual(execute.call_count, 2)

    def test_two_backends_use_one_then_three_workers_and_manifest_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root/"out"
            out.mkdir()
            launched = []

            class FakeWorker:
                def __init__(self, command, **kwargs):
                    backend = command[command.index("--backend")+1]
                    index = int(command[command.index("--shard-index")+1])
                    count = int(command[command.index("--shard-count")+1])
                    launched.append((backend, index, count))
                    path = out/backend/"utilities.json"
                    path.parent.mkdir(exist_ok=True)
                    rows = json.loads(path.read_text()) if path.exists() else []
                    for i, target in enumerate("abcde"):
                        if i % count != index:
                            continue
                        for degree in (0, .025, .25):
                            for metric in run.METRICS:
                                rows.append({"backend": backend, "size": 5, "target": target,
                                             "blur_deg": degree, "metric": metric, "status": "ok"})
                    path.write_text(json.dumps(rows))
                    self.pid, self.returncode = 100+len(launched), 0

                def poll(self):
                    return self.returncode

            with (
                patch.object(run_parallel, "ROOT", root),
                patch.object(run_parallel, "cohort_configuration", return_value={}),
                patch.object(run_parallel.subprocess, "Popen", side_effect=FakeWorker),
            ):
                run_parallel.execute(self.protocol(), root/"protocol.json", self.manifest(),
                                     root/"manifest.json", out, "mps", 2026, {})
            self.assertEqual(launched, [("vggt", 0, 1), ("mast3r", 0, 3), ("mast3r", 1, 3), ("mast3r", 2, 3)])
            state = json.loads((out/"batch_status.json").read_text())
            self.assertEqual(state["planned_conditions"], 30)
            self.assertEqual(state["attempted_conditions"], 30)
            self.assertEqual(state["status"], "complete")


class InterfaceTests(unittest.TestCase):
    def make_fixture(self, root):
        (root/"plan.md").write_text("approved interface fixture")
        protocol = {"protocol_id": "fixture", "status": "approved", "implementation_ready": True,
                    "plan": "plan.md", "approved_plan_sha256": run.sha256(root/"plan.md"),
                    "primary_scenes": ["office", "courtyard"], "initial_scenes": ["office"],
                    "expansion_scenes": ["courtyard"], "sizes": [5], "blur_degrees": [0, .25],
                    "seeds": [2026], "backends": ["vggt", "mast3r"],
                    "execution": {"workers": 3, "threads_per_worker": 8}}
        (root/"protocol.json").write_text(json.dumps(protocol))
        scripts = root/"experiments"
        scripts.mkdir()
        (scripts/"prepare_study.py").touch()
        (scripts/"analyze_study.py").touch()
        folder = root/"data/experiments/fixture/office"
        folder.mkdir(parents=True)
        (folder/"manifest.json").write_text(json.dumps(
            {"protocol_id": "fixture", "scene": "office", "sets": {"5": list("abcde")},
             "severities_deg": [0, .25]}))
        return protocol

    def test_same_user_command_selects_mps_or_cuda_and_runs_one_seed(self):
        for device in ("mps", "cuda"):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_fixture(root)
                args = run_all.build_parser().parse_args(["run", "--protocol", "protocol.json", "--scene", "office"])
                scientific = {key: "fixed" for key in
                              ("protocol_sha256", "plan_sha256", "precision", "workers", "threads_per_worker",
                               "seeds", "vendor_versions", "mast3r_model_config_sha256", "source_sha256")}
                with (
                    patch.object(run_all, "ROOT", root),
                    patch.object(run_parallel, "ROOT", root),
                    patch.object(run_all, "resolve_device", return_value=device),
                    patch.object(run_all, "cohort_configuration", return_value=scientific),
                    patch.object(run_all, "run_worker_group") as launch,
                    patch.object(run_all.subprocess, "run") as analyze,
                ):
                    run_all.run(args)
                calls = [item.args[0] for item in launch.call_args_list]
                self.assertEqual([c.seed for c in calls], [2026])
                self.assertTrue(all(c.device == device for c in calls))
                self.assertTrue(all(c.manifest.endswith("fixture/office/manifest.json") for c in calls))
                self.assertTrue(all(f"/{device}-" in c.cohort for c in calls))
                self.assertEqual(analyze.call_count, 1)
                assignments = json.loads((root/"results/fixture/scene_assignments.json").read_text())
                self.assertTrue(assignments["office"].startswith(device+"-"))

    def test_human_entrypoint_blocks_unapproved_work_before_gpu_or_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = self.make_fixture(root)
            protocol["status"] = "awaiting_approval"
            (root/"protocol.json").write_text(json.dumps(protocol))
            for action in ("prepare", "run", "analyze"):
                args = run_all.build_parser().parse_args([action, "--protocol", "protocol.json", "--scene", "office"])
                with patch.object(run_all, "ROOT", root), patch.object(run_all, "resolve_device") as gpu:
                    with self.assertRaisesRegex(RuntimeError, "approval"):
                        run_all.run(args)
                    gpu.assert_not_called()
            self.assertFalse((root/"results").exists())

    def test_status_is_read_only_and_does_not_require_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            args = run_all.build_parser().parse_args(["status", "--protocol", "protocol.json", "--scene", "office"])
            with (
                patch.object(run_all, "ROOT", root),
                patch.object(run_all, "resolve_device") as gpu,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = run_all.run(args)
                gpu.assert_not_called()
            self.assertEqual(result["runs"], [{"scene": "office", "status": "not_started"}])
            self.assertFalse((root/"results").exists())

    def test_expansion_does_not_silently_run_on_local_mps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            args = run_all.build_parser().parse_args(["run", "--protocol", "protocol.json", "--remaining"])
            with (
                patch.object(run_all, "ROOT", root),
                patch.object(run_parallel, "ROOT", root),
                patch.object(run_all, "resolve_device", return_value="mps"),
                patch.object(run_all, "run_worker_group") as launch,
            ):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    run_all.run(args)
                launch.assert_not_called()
            self.assertFalse((root/"results").exists())

    def test_server_expansion_keeps_office_in_primary_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = self.make_fixture(root)
            data = root/"data/experiments/fixture/courtyard"
            data.mkdir()
            (data/"manifest.json").write_text(json.dumps(
                {"protocol_id": "fixture", "scene": "courtyard", "sets": {"5": list("abcde")},
                 "severities_deg": [0, .25]}))
            results = root/"results/fixture"
            for seed in protocol["seeds"]:
                state = results/"executions/mps-local/office"/f"seed_{seed}"
                state.mkdir(parents=True)
                (state/"batch_status.json").write_text(json.dumps({"status": "complete", "seed": seed}))
            (results/"scene_assignments.json").write_text(json.dumps({"office": "mps-local"}))
            scientific = {key: "fixed" for key in
                          ("protocol_sha256", "plan_sha256", "precision", "workers", "threads_per_worker",
                           "seeds", "vendor_versions", "mast3r_model_config_sha256", "source_sha256")}
            (results/"study_configuration.json").write_text(json.dumps(scientific))
            args = run_all.build_parser().parse_args(["run", "--protocol", "protocol.json", "--remaining"])
            with (
                patch.object(run_all, "ROOT", root),
                patch.object(run_parallel, "ROOT", root),
                patch.object(run_all, "resolve_device", return_value="cuda"),
                patch.object(run_all, "cohort_configuration", return_value=scientific),
                patch.object(run_all, "run_worker_group") as launch,
                patch.object(run_all.subprocess, "run") as analyze,
            ):
                run_all.run(args)
            self.assertEqual(launch.call_count, 1)
            self.assertTrue(all(call.args[0].device == "cuda" for call in launch.call_args_list))
            assignments = json.loads((results/"scene_assignments.json").read_text())
            self.assertEqual(assignments["office"], "mps-local")
            self.assertTrue(assignments["courtyard"].startswith("cuda-"))
            command = analyze.call_args.args[0]
            self.assertEqual(command[command.index("--scenes")+1:], ["office", "courtyard"])


if __name__ == "__main__":
    unittest.main()
