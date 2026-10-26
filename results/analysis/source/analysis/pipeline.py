"""Audited, reproducible analysis orchestration; inference files are read-only."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

from filelock import FileLock

from .audit import Sources, load_scene
from .context import context_descriptors
from .core import (VERSION, METRICS, curve_summaries, stage_summaries, transition_summaries,
                   cohort_summaries, cohort_transition_summaries, paired_comparisons)
from .plots import make_scene_figures, make_cohort_figure
from .report import scene_report, cohort_report, insert_reconstruction_section
from .reconstruction import RENDER_POLICY, make_reconstruction_figures

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def flatten(record, prefix=""):
    output = {}
    for key, value in record.items():
        name = prefix + key
        if isinstance(value, dict):
            output.update(flatten(value, name + "_"))
        else:
            output[name] = json.dumps(value, ensure_ascii=False) if isinstance(value, (tuple, list)) else value
    return output


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten(row) for row in records]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analysis_definition(protocol):
    return {
        "analysis_version": VERSION, "protocol_id": protocol["protocol_id"],
        "primary_metric": protocol["evaluation"]["primary_metric"],
        "secondary_metrics": protocol["evaluation"]["secondary_metrics"],
        "sizes": protocol["sizes"], "blur_degrees": protocol["blur_degrees"], "seeds": protocol["seeds"],
        "plot_x_axis": {"positions": "equally spaced observed stages",
                        "labels": "actual yaw span in degrees on every panel",
                        "spacing_proportional_to_angle": False},
        "utility": "E_DROP - E_ADD in original metric units; never the stored relative pose utility",
        "primary_effect_floor": protocol["evaluation"]["primary_absrel_effect_floor"],
        "sensitivity_effect_floors": protocol["evaluation"]["sensitivity_absrel_effect_floors"],
        "classes": {"helpful": "U > +epsilon", "harmful": "U < -epsilon", "within_threshold": "-epsilon <= U <= +epsilon",
                    "unavailable": "missing or invalid DROP/ADD metric"},
        "stage_denominator": "all planned targets m, with missing and invalid counts explicit",
        "target_distribution": "mean, median, Q25, Q75, min, max of valid observed target values; not confidence intervals",
        "transition": "clean helpful followed by observed harmful; bracket last helpful sample before first harmful sample; no interpolation",
        "transition_interior": "retain within-threshold and unavailable samples inside bracket; label unresolved interior path",
        "transition_reversals": "record all helpful/harmful switches, skipping neutral/NA but preserving their locations",
        "transition_complete_primary": "all planned curve slots valid; incomplete observed transitions reported separately",
        "cohort": "scene fractions first, then equal scene weights; mean/median/min/max across scenes; target pooling secondary",
        "strata": ["all available requested scenes", "device", "runtime_group"],
        "missing_scene": "not imputed; requested and available counts always reported",
        "representatives": "lexicographic first, floor(N/2), last image ID within each selected set",
        "statistical_inference": "none: no repeats, confidence intervals, p-values, population or causal claims",
        "platform_equivalence": protocol["evaluation"]["platform_equivalence"],
        "score_audit": "SHA-256 of every saved prediction + CPU rescore all recorded metrics from frozen common-grid depths/poses and official GT",
        "score_tolerance": {"relative": 1e-8, "absolute": 1e-10},
        "diagnostics": "nonpositive and nonfinite depth fractions on entire fixed GT mask, plus strict-positive diagnostic only",
        "exploratory_descriptors": "pre-inference pair overlaps; baseline, rotation and GT depth distributions derived during analysis from frozen GT snapshot",
        "qualitative_reconstruction": RENDER_POLICY,
    }


def identical_files(previous, current, names):
    return all((previous / name).is_file() and (current / name).is_file()
               and (previous / name).read_bytes() == (current / name).read_bytes() for name in names)


def unchanged_analysis_code(previous_manifest, current_code, modules):
    previous = previous_manifest.get("analysis_source_sha256", {})
    return all(previous.get("experiments/analysis/" + name) == current_code.get("experiments/analysis/" + name)
               for name in modules)


def reusable_figures(previous_manifest, previous_root, prefix, current_code):
    """Keep existing PNG/PDF bytes when statistical inputs and plot code match."""
    if not unchanged_analysis_code(previous_manifest, current_code, ("core.py", "plots.py")):
        return False
    if any(previous_manifest.get("packages", {}).get(name) != importlib.metadata.version(name)
           for name in ("numpy", "matplotlib")):
        return False
    expected = {name: digest for name, digest in previous_manifest.get("output_files_sha256", {}).items()
                if name.startswith(prefix) and "/reconstruction_" not in name}
    return bool(expected) and all((previous_root / name).is_file()
        and hashlib.sha256((previous_root / name).read_bytes()).hexdigest() == digest for name, digest in expected.items())


def publish(stage, destination):
    backup = destination.with_name(destination.name + f".previous-{os.getpid()}")
    if backup.exists():
        raise RuntimeError(f"Unfinished previous publication needs inspection: {backup}")
    if destination.exists():
        destination.rename(backup)
    try:
        stage.rename(destination)
    except BaseException:
        if backup.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def analyze(protocol_path, results_root, requested):
    with FileLock(str(results_root / ".analysis.lock"), timeout=0):
        return _analyze(protocol_path, results_root, requested)


def _analyze(protocol_path, results_root, requested):
    started = datetime.now(timezone.utc).isoformat()
    sources = Sources(ROOT)
    protocol = sources.json(protocol_path)
    if protocol.get("status") != "approved" or not protocol.get("implementation_ready"):
        raise ValueError("Analysis requires the approved implemented protocol")
    if not requested or len(requested) != len(set(requested)) or set(requested) - set(protocol["primary_scenes"]):
        raise ValueError("Requested scenes must be unique approved primary scenes")
    sources.check(ROOT / protocol["plan"], protocol["approved_plan_sha256"])
    if protocol["evaluation"]["policy"] != "fixed_common_grid_signed_depth_v1" or not protocol["evaluation"]["target_excluded"]:
        raise ValueError("Unsupported evaluation policy; create a reviewed analysis version for a new scientific definition")
    study = sources.json(results_root / "study_configuration.json")
    if study["protocol_sha256"] != sources.check(protocol_path) or study["plan_sha256"] != protocol["approved_plan_sha256"]:
        raise ValueError("Protocol/plan differ from frozen inference configuration")
    for filename, expected in study["source_sha256"].items():
        sources.check(ROOT / filename, expected)
    code_files = [ROOT / "experiments/analyze_study.py", ROOT / "experiments/metrics.py",
                  *sorted((ROOT / "experiments/analysis").glob("*.py"))]
    code_sha = {str(path.relative_to(ROOT)): sources.check(path) for path in code_files}
    reconciliation = results_root / "plan_fingerprint_reconciliation.json"
    if reconciliation.exists():
        sources.check(reconciliation)
    roster, available = {}, []
    for scene in protocol["primary_scenes"]:
        folder = results_root / "executions" / scene
        status_path = folder / "batch_status.json"
        # Unrequested scene status is only a point-in-time roster label, not an
        # input to this scope's numerical analysis. Another scene may be running.
        status = (sources.json(status_path) if scene in requested else json.loads(status_path.read_text())) if status_path.exists() else {}
        roster[scene] = status.get("status", "unknown") if status else "not_started"
        if scene in requested and all((folder / name).is_file() for name in ("inputs.json", "batch_status.json", "cohort_configuration.json")):
            available.append(scene)
    stage = Path(tempfile.mkdtemp(prefix=".analysis-", dir=results_root))
    destination = results_root / "analysis"
    previous_manifest_path = destination / "analysis_manifest.json"
    previous_manifest = json.loads(previous_manifest_path.read_text()) if previous_manifest_path.exists() else {}
    try:
        if (destination / "scenes").exists():
            shutil.copytree(destination / "scenes", stage / "scenes")
        if (destination / "figures").exists():
            shutil.copytree(destination / "figures", stage / "figures")
        data_by_scene = {}
        qualitative_status, retained_figure_paths = {}, []
        preserved_report_bases = {}
        all_rows, all_stages, all_curves, all_transitions, all_sensitivity_stages, all_sensitivity_transitions, all_paired = [], [], [], [], [], [], []
        for scene in available:
            print(f"AUDIT {scene}: verifying saved predictions and rescoring all metrics on CPU", flush=True)
            data = load_scene(ROOT, results_root, scene, protocol, study, sources)
            data_by_scene[scene] = data
            rows = data["rows"]
            stages = stage_summaries(rows, protocol)
            curves = curve_summaries(rows, protocol)
            transitions = transition_summaries(curves)
            sensitivity_stages = stage_summaries(rows, protocol, sensitivity=True)
            sensitivity_transitions = transition_summaries(curves, sensitivity=True)
            paired = paired_comparisons(rows, protocol)
            contexts = context_descriptors(data, protocol)
            data["sensitivity_stages"] = sensitivity_stages
            scene_folder = stage / "scenes" / scene
            previous_scene = destination / "scenes" / scene
            scene_folder.mkdir(parents=True, exist_ok=True)
            for name, records in (("metrics", rows), ("stage_summary", stages), ("curves", curves),
                                  ("transitions", transitions), ("sensitivity_stages", sensitivity_stages),
                                  ("sensitivity_transitions", sensitivity_transitions), ("paired_comparisons", paired),
                                  ("depth_diagnostics", data["diagnostics"]), ("context_descriptors", contexts)):
                write_csv(scene_folder / (name + ".csv"), records)
            write_json(scene_folder / "curves.json", curves)
            write_json(scene_folder / "audit.json", data["audit"])
            write_json(scene_folder / "costs.json", data["costs"])
            write_csv(scene_folder / "reference_pixels.csv", [
                {"image_id": frame["id"], "camera_id": frame["camera_id"],
                 "valid_gt_pixels": frame["official_gt_valid_pixels"],
                 "depth_sha256": frame["depth_sha256"], "mask_sha256": frame["official_valid_mask_sha256"],
                 **{f"in_set_{size}": frame["id"] in data["manifest"]["sets"][str(size)] for size in protocol["sizes"]}}
                for frame in data["manifest"]["frames"]])
            numeric_names = [name + ".csv" for name in ("metrics", "stage_summary", "curves", "transitions",
                "sensitivity_stages", "sensitivity_transitions", "paired_comparisons", "depth_diagnostics", "context_descriptors", "reference_pixels")]
            numeric_names += ["curves.json", "audit.json", "costs.json"]
            same_statistics = identical_files(previous_scene, scene_folder, numeric_names) and unchanged_analysis_code(
                previous_manifest, code_sha, ("core.py", "audit.py", "context.py"))
            prefix = f"scenes/{scene}/figures/"
            if same_statistics and reusable_figures(previous_manifest, destination, prefix, code_sha):
                retained_figure_paths.extend(name for name in previous_manifest["output_files_sha256"]
                                             if name.startswith(prefix) and "/reconstruction_" not in name)
            else:
                make_scene_figures(data, stages, contexts, protocol, scene_folder / "figures")
            qualitative = make_reconstruction_figures(ROOT, results_root / "executions" / scene,
                data, protocol, scene_folder / "figures", sources)
            write_json(scene_folder / "reconstruction_visualization.json", qualitative)
            qualitative_status[scene] = qualitative["status"]
            if same_statistics and (previous_scene / "report.md").exists():
                previous_report_path = previous_scene / "report.md"
                previous_report_bytes = previous_report_path.read_bytes()
                preserved_report_bases[previous_report_path] = hashlib.sha256(previous_report_bytes).hexdigest()
                report = previous_report_bytes.decode()
            else:
                report = scene_report(data, stages, curves, transitions, sensitivity_transitions, contexts, protocol)
            (scene_folder / "report.md").write_text(insert_reconstruction_section(report, qualitative))
            all_rows.extend(rows)
            all_stages.extend(stages)
            all_curves.extend(curves)
            all_transitions.extend(transitions)
            all_sensitivity_stages.extend(sensitivity_stages)
            all_sensitivity_transitions.extend(sensitivity_transitions)
            all_paired.extend(paired)
            print(f"ANALYZED {scene}: {len(rows)} planned metric cells; verified hashes and CPU scores", flush=True)
        info = {scene: data["info"] for scene, data in data_by_scene.items()}
        cohort = cohort_summaries(all_stages, info, requested)
        cohort_transitions = cohort_transition_summaries(all_transitions, info, requested)
        cohort_sensitivity = cohort_summaries(all_sensitivity_stages, info, requested)
        cohort_sensitivity_transitions = cohort_transition_summaries(all_sensitivity_transitions, info, requested)
        for name, records in (("scene_stage_summary", all_stages), ("cohort_summary", cohort),
                              ("scene_transitions", all_transitions), ("cohort_transitions", cohort_transitions),
                              ("cohort_sensitivity", cohort_sensitivity), ("cohort_sensitivity_transitions", cohort_sensitivity_transitions),
                              ("paired_comparisons", all_paired)):
            write_csv(stage / "tables" / (name + ".csv"), records)
        if (identical_files(destination / "tables", stage / "tables", ["scene_stage_summary.csv", "cohort_summary.csv"])
                and reusable_figures(previous_manifest, destination, "figures/", code_sha)):
            retained_figure_paths.extend(name for name in previous_manifest["output_files_sha256"] if name.startswith("figures/"))
        else:
            make_cohort_figure(cohort, protocol, stage / "figures")
        write_json(stage / "analysis_definition.json", analysis_definition(protocol))
        summary = {"analysis_version": VERSION, "requested_scenes": requested, "analyzed_scenes": available,
                   "pending_requested_scenes": [s for s in requested if s not in available],
                   "primary_scene_execution_status": roster,
                   "requested_scope_complete": len(available) == len(requested) and all(v["complete"] for v in info.values()),
                   "full_primary_cohort_execution_complete": all(roster[s] == "complete" for s in protocol["primary_scenes"]),
                   "full_primary_cohort_audited_complete": set(protocol["primary_scenes"]) <= set(info)
                       and all(info[s]["complete"] for s in protocol["primary_scenes"]),
                   "scene_info": info, "scene_stage_summary": all_stages, "cohort_summary": cohort,
                   "scene_transitions": all_transitions, "cohort_transitions": cohort_transitions,
                   "cohort_sensitivity": cohort_sensitivity, "cohort_sensitivity_transitions": cohort_sensitivity_transitions,
                   "paired_comparisons": all_paired}
        write_json(stage / "summary.json", summary)
        (stage / "report.md").write_text(cohort_report(data_by_scene, cohort, protocol, requested, roster, VERSION))
        for path in code_files:
            relative = path.relative_to(ROOT / "experiments")
            target = stage / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        write_json(stage / "protocol.json", protocol)
        sources.assert_unchanged()
        for path, digest in preserved_report_bases.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Authored report changed during analysis; preserving previous report: {path}")
        outputs = {str(path.relative_to(stage)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(stage.rglob("*")) if path.is_file()}
        write_json(stage / "analysis_manifest.json", {
            "analysis_version": VERSION, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "requested_scenes": requested, "analyzed_scenes": available,
            "scope_complete": summary["requested_scope_complete"], "definition_sha256": outputs["analysis_definition.json"],
            "analysis_source_sha256": code_sha,
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")},
            "python": platform.python_version(), "platform": platform.platform(),
            "input_files": sources.manifest(), "output_files_sha256": outputs,
            "input_result_files_modified": False, "model_inference_executed": False,
            "qualitative_reconstruction_status": qualitative_status,
            "retained_statistical_figure_files": retained_figure_paths,
            "preserved_scene_report_base_sha256": {str(p.relative_to(destination)): digest
                                                    for p, digest in preserved_report_bases.items()},
            "preserved_unrequested_scene_reports": [p.name for p in (stage / "scenes").iterdir() if p.is_dir() and p.name not in available]
                if (stage / "scenes").exists() else [],
        })
        publish(stage, destination)
        (results_root / "analysis_failure.json").unlink(missing_ok=True)
        print(json.dumps({"analysis": str(destination), "analyzed_scenes": available,
                          "scope_complete": summary["requested_scope_complete"], "inference_executed": False}, ensure_ascii=False), flush=True)
        return summary
    except BaseException as error:
        shutil.rmtree(stage, ignore_errors=True)
        write_json(results_root / "analysis_failure.json", {"status": "failed", "error": str(error),
            "started_utc": started, "previous_successful_analysis_preserved": destination.exists()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/study_v1.json"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--qualitative-only", action="store_true",
                        help="Add point-cloud figures to existing verified reports while preserving all existing statistics and figures")
    args = parser.parse_args()
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    results_root = args.results_root if args.results_root.is_absolute() else ROOT / args.results_root
    protocol = json.loads(protocol_path.read_text())
    action = augment_reconstructions if args.qualitative_only else analyze
    action(protocol_path, results_root, args.scenes or protocol["primary_scenes"])


def augment_reconstructions(protocol_path, results_root, requested):
    """Backfill qualitative panels without republishing statistical artifacts."""
    with FileLock(str(results_root / ".analysis.lock"), timeout=0):
        sources = Sources(ROOT)
        protocol = sources.json(protocol_path)
        if protocol.get("status") != "approved" or not protocol.get("implementation_ready"):
            raise ValueError("Qualitative augmentation requires the approved implemented protocol")
        if not requested or len(requested) != len(set(requested)) or set(requested) - set(protocol["primary_scenes"]):
            raise ValueError("Requested scenes must be unique approved primary scenes")
        sources.check(ROOT / protocol["plan"], protocol["approved_plan_sha256"])
        study = sources.json(results_root / "study_configuration.json")
        if study["protocol_sha256"] != sources.check(protocol_path) or study["plan_sha256"] != protocol["approved_plan_sha256"]:
            raise ValueError("Protocol/plan differ from frozen inference configuration")
        for filename, digest in study["source_sha256"].items():
            sources.check(ROOT / filename, digest)
        destination = results_root / "analysis"
        previous_bytes = (destination / "analysis_manifest.json").read_bytes()
        previous = json.loads(previous_bytes)
        summary = json.loads((destination / "summary.json").read_text())
        available = [s for s in requested if s in summary["analyzed_scenes"]]
        if not available:
            raise ValueError("No existing analyzed scene in the requested scope; run normal analyze first")
        code_files = [ROOT / "experiments/analyze_study.py", ROOT / "experiments/metrics.py",
                      *sorted((ROOT / "experiments/analysis").glob("*.py"))]
        code_sha = {str(p.relative_to(ROOT)): sources.check(p) for p in code_files}
        stage = Path(tempfile.mkdtemp(prefix=".analysis-qualitative-", dir=results_root))
        started = datetime.now(timezone.utc).isoformat()
        bases, statuses = {}, {}
        try:
            shutil.copytree(destination, stage, dirs_exist_ok=True)
            for scene in available:
                print(f"QUALITATIVE {scene}: audit existing outputs; preserve statistics and figures", flush=True)
                data = load_scene(ROOT, results_root, scene, protocol, study, sources)
                scene_folder = stage / "scenes" / scene
                check_path = stage / ".metrics-check.csv"
                write_csv(check_path, data["rows"])
                if check_path.read_bytes() != (scene_folder / "metrics.csv").read_bytes():
                    raise ValueError(f"Existing analysis no longer matches source scores: {scene}; normal analyze required")
                check_path.unlink()
                report_path = destination / "scenes" / scene / "report.md"
                original = report_path.read_bytes()
                bases[report_path] = hashlib.sha256(original).hexdigest()
                info = make_reconstruction_figures(ROOT, results_root / "executions" / scene,
                    data, protocol, scene_folder / "figures", sources)
                write_json(scene_folder / "reconstruction_visualization.json", info)
                (scene_folder / "report.md").write_text(insert_reconstruction_section(original.decode(), info))
                statuses[scene] = info["status"]
            definition = json.loads((stage / "analysis_definition.json").read_text())
            definition["qualitative_reconstruction"] = RENDER_POLICY
            write_json(stage / "analysis_definition.json", definition)
            for path in code_files:
                target = stage / "source_qualitative" / path.relative_to(ROOT / "experiments")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            sources.assert_unchanged()
            for path, digest in bases.items():
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Authored report changed during augmentation: {path}")
            outputs = {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(stage.rglob("*")) if p.is_file() and p != stage / "analysis_manifest.json"}
            allowed_changes = {"analysis_definition.json", "analysis_manifest.json",
                               *(str(p.relative_to(destination)) for p in bases)}
            preserved = []
            for path in destination.rglob("*"):
                if not path.is_file():
                    continue
                relative = str(path.relative_to(destination))
                managed_qualitative = relative.startswith("source_qualitative/") or any(
                    relative == f"scenes/{scene}/reconstruction_visualization.json"
                    or relative.startswith(f"scenes/{scene}/figures/reconstruction_") for scene in available)
                if relative not in allowed_changes and not managed_qualitative:
                    if outputs.get(relative) != hashlib.sha256(path.read_bytes()).hexdigest():
                        raise ValueError(f"Existing artifact changed during additive visualization: {relative}")
                    preserved.append(relative)
            manifest = dict(previous)
            manifest["definition_sha256"] = outputs["analysis_definition.json"]
            manifest["output_files_sha256"] = outputs
            manifest["qualitative_augmentation"] = {
                "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "scenes": available, "status_by_scene": statuses, "analysis_source_sha256": code_sha,
                "input_files": sources.manifest(), "previous_analysis_manifest_sha256": hashlib.sha256(previous_bytes).hexdigest(),
                "scene_report_base_sha256": {str(p.relative_to(destination)): h for p, h in bases.items()},
                "preserved_existing_files": preserved, "model_inference_executed": False,
                "statistical_outputs_recomputed": False,
            }
            write_json(stage / "analysis_manifest.json", manifest)
            publish(stage, destination)
            (results_root / "analysis_failure.json").unlink(missing_ok=True)
            print(json.dumps({"qualitative_scenes": statuses, "existing_files_preserved": len(preserved),
                              "inference_executed": False}, ensure_ascii=False), flush=True)
            return statuses
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
