"""One user interface on Mac and CUDA servers: prepare, run, status, analyze."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from types import SimpleNamespace

from filelock import FileLock
from run import ROOT, atomic_json, resolve_device
from run_parallel import cohort_configuration, require_approval, run as run_worker_group, validate_protocol


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="run", choices=("prepare", "run", "status", "analyze"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--scene", "--scenes", nargs="+", dest="scenes")
    selection.add_argument("--remaining", action="store_true", help="The planned expansion scenes after the initial office run")
    parser.add_argument("--protocol", default="experiments/study_v1.json")
    parser.add_argument("--seed", type=int, help="Default: all seeds in the approved protocol")
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda"),
                        help="Same default everywhere: CUDA if available, otherwise MPS; never implicit CPU")
    parser.add_argument("--input-root", help="Default: data/experiments/<protocol_id>")
    parser.add_argument("--results-root", help="Default: results/<protocol_id>")
    return parser


def select_scenes(args, protocol):
    if args.remaining:
        scenes = protocol["expansion_scenes"]
    elif args.scenes:
        scenes = args.scenes
    elif args.action == "analyze":
        scenes = protocol["primary_scenes"]
    else:
        scenes = protocol["initial_scenes"]
    if len(scenes) != len(set(scenes)) or any(s not in protocol["primary_scenes"] for s in scenes):
        raise ValueError("Scene selection is not in the approved primary scene list")
    return scenes


def runtime_identifier(device):
    host = hashlib.sha256(platform.node().encode()).hexdigest()[:8]
    return f"{device}-{host}"


def scientific_configuration(runtime):
    keys = ("protocol_sha256", "plan_sha256", "precision", "workers", "threads_per_worker",
            "seeds", "vendor_versions", "mast3r_model_config_sha256", "source_sha256")
    return {key: runtime[key] for key in keys}


def run(args):
    protocol_path = ROOT/args.protocol
    protocol = json.loads(protocol_path.read_text())
    scenes = select_scenes(args, protocol)
    input_root = ROOT/args.input_root if args.input_root else ROOT/"data/experiments"/protocol["protocol_id"]
    results_root = ROOT/args.results_root if args.results_root else ROOT/"results"/protocol["protocol_id"]
    if args.action == "status":
        rows = []
        for scene in scenes:
            found = sorted(results_root.glob(f"executions/*/{scene}/seed_*/batch_status.json"))
            if not found:
                rows.append({"scene": scene, "status": "not_started"})
            for path in found:
                state = json.loads(path.read_text())
                rows.append({**state, "runtime_group": path.parents[2].name})
        result = {"protocol_id": protocol["protocol_id"], "approval_status": protocol["status"],
                  "sizes": protocol["sizes"], "seeds": protocol["seeds"], "runs": rows}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    require_approval(protocol, require_implementation=args.action != "prepare")
    prepare_script = ROOT/"experiments/prepare_study.py"
    analysis_script = ROOT/"experiments/analyze_study.py"
    if args.action == "prepare":
        if not prepare_script.exists():
            raise RuntimeError("The approved common preparation is not implemented yet")
        subprocess.run([sys.executable, str(prepare_script), "--protocol", str(protocol_path),
                        "--input-root", str(input_root), "--scenes", *scenes], check=True, cwd=ROOT)
        return
    if not analysis_script.exists():
        raise RuntimeError("The approved common analysis is not implemented yet")
    if args.action == "analyze":
        subprocess.run([sys.executable, str(analysis_script), "--protocol", str(protocol_path),
                        "--results-root", str(results_root), "--scenes", *scenes], check=True, cwd=ROOT)
        return
    device = resolve_device(args.device)
    expansion = any(scene in protocol["expansion_scenes"] for scene in scenes)
    if expansion and device != "cuda":
        raise RuntimeError("Expansion scenes await the CUDA server; they were not assigned to local MPS")
    if expansion:
        # The local study records are transferred with the prepared inputs;
        # an expansion invocation never silently starts a different study.
        assignment_file = results_root/"scene_assignments.json"
        if not (results_root/"study_configuration.json").exists() or not assignment_file.exists():
            raise RuntimeError("The office scientific lock and scene assignments were not transferred")
        assignments = json.loads(assignment_file.read_text())
        for initial in protocol["initial_scenes"]:
            if initial not in assignments:
                raise RuntimeError("The office primary runtime assignment is missing")
            for seed in protocol["seeds"]:
                path = results_root/"executions"/assignments[initial]/initial/f"seed_{seed}/batch_status.json"
                if not path.exists() or json.loads(path.read_text())["status"] not in ("complete", "complete_with_failures"):
                    raise RuntimeError("Initial office run is incomplete or its study records were not transferred")
    seeds = protocol["seeds"] if args.seed is None else [args.seed]
    manifests = {}
    for scene in scenes:
        path = input_root/scene/"manifest.json"
        manifest = json.loads(path.read_text())
        for seed in seeds:
            validate_protocol(protocol, manifest, seed)
        manifests[scene] = path
    runtime = cohort_configuration(protocol_path, device, protocol)
    study = scientific_configuration(runtime)
    results_root.mkdir(parents=True, exist_ok=True)
    study_path = results_root/"study_configuration.json"
    with FileLock(str(study_path)+".lock"):
        if study_path.exists() and json.loads(study_path.read_text()) != study:
            raise ValueError("Scientific procedure changed; use a new approved study version")
        atomic_json(study_path, study)
    runtime_id = runtime_identifier(device)
    assignment_path = results_root/"scene_assignments.json"
    with FileLock(str(assignment_path)+".lock"):
        assignments = json.loads(assignment_path.read_text()) if assignment_path.exists() else {}
        for scene in scenes:
            if scene in assignments and assignments[scene] != runtime_id:
                raise ValueError("Scene already belongs to a different runtime group; cross-device resume is forbidden")
            assignments[scene] = runtime_id
        atomic_json(assignment_path, assignments)
    cohort = results_root/"executions"/runtime_id
    for scene in scenes:
        for seed in seeds:
            run_worker_group(SimpleNamespace(protocol=str(protocol_path), manifest=str(manifests[scene]),
                                            cohort=str(cohort), device=device, seed=seed))
    # Automatic reporting includes office and every available primary scene,
    # even when the just-finished command selected only the expansion scenes.
    subprocess.run([sys.executable, str(analysis_script), "--protocol", str(protocol_path),
                    "--results-root", str(results_root), "--scenes", *protocol["primary_scenes"]],
                   check=True, cwd=ROOT)


if __name__ == "__main__":
    run(build_parser().parse_args())
