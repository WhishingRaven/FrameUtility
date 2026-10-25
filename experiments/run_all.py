"""One user interface on Mac and CUDA servers: prepare, run, status, analyze."""
import argparse
import json
from pathlib import PurePosixPath
import shlex
import subprocess
import sys
from types import SimpleNamespace

from filelock import FileLock
from run import ROOT, atomic_json, resolve_device
from run_parallel import cohort_configuration, require_approval, run as run_worker_group, runtime_identifier, validate_protocol


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="run", choices=("prepare", "run", "status", "analyze"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--scene", "--scenes", nargs="+", dest="scenes")
    selection.add_argument("--remaining", action="store_true", help="The planned expansion scene list; independent of execution target")
    parser.add_argument("--target", default="local", choices=("local", "cuda-server"),
                        help="Execution location for every scene and action (default: local)")
    parser.add_argument("--server-config", default="experiments/cuda_server.json",
                        help="SSH profile used only by --target cuda-server")
    parser.add_argument("--protocol", default="experiments/study_v1.json")
    parser.add_argument("--seed", type=int, help="Default: all seeds in the approved protocol")
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda"),
                        help="GPU override on the execution host; auto prefers CUDA then MPS, never implicit CPU")
    parser.add_argument("--input-root", help="Default: data/experiments/eth3d_multi_view/")
    parser.add_argument("--results-root", help="Default: results/ under the project root")
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


def scientific_configuration(runtime):
    keys = ("protocol_sha256", "plan_sha256", "precision", "workers", "threads_per_worker",
            "seeds", "vendor_versions", "mast3r_model_config_sha256", "source_sha256")
    return {key: runtime[key] for key in keys}


def run_remote(args):
    if args.device not in ("auto", "cuda"):
        raise ValueError("--target cuda-server requires CUDA; --device mps is incompatible")
    profile_path = ROOT/args.server_config
    if not profile_path.exists():
        raise RuntimeError("CUDA server config is missing; copy experiments/cuda_server.example.json "
                           "to experiments/cuda_server.json and configure the SSH host and project path")
    profile = json.loads(profile_path.read_text())
    host, directory, python = (profile.get(key) for key in ("host", "project_dir", "python"))
    if (not isinstance(host, str) or not host or host.startswith("-")
            or any(c.isspace() or c in "\x00'\";`$" for c in host)):
        raise ValueError("Server profile host must be an SSH host alias or user@host")
    if not isinstance(directory, str) or "\x00" in directory or not PurePosixPath(directory).is_absolute():
        raise ValueError("Server profile project_dir must be an absolute server path")
    if not isinstance(python, str) or not python or python.startswith("-") or "\x00" in python:
        raise ValueError("Server profile python must name the server Python executable")
    port = profile.get("port", 22)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Server profile port must be an integer from 1 to 65535")
    child = [python, "experiments/run_all.py", args.action, "--target", "local", "--device", "cuda",
             "--protocol", args.protocol]
    if args.remaining:
        child.append("--remaining")
    elif args.scenes:
        child.extend(["--scenes", *args.scenes])
    for name in ("seed", "input_root", "results_root"):
        value = getattr(args, name)
        if value is not None:
            child.extend(["--"+name.replace("_", "-"), str(value)])
    # Quote remote arguments once for the remote shell; never invoke a local shell.
    command = shlex.join(["cd", "--", directory])+" && exec "+shlex.join(child)
    return subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                           "-p", str(port), host, command], check=True, cwd=ROOT)


def run(args):
    protocol_path = ROOT/args.protocol
    protocol = json.loads(protocol_path.read_text())
    scenes = select_scenes(args, protocol)
    if args.target == "cuda-server":
        if args.action != "status":
            require_approval(protocol, require_implementation=args.action != "prepare")
        return run_remote(args)
    input_root = ROOT/args.input_root if args.input_root else ROOT/"data/experiments/eth3d_multi_view"
    results_root = ROOT/args.results_root if args.results_root else ROOT/"results"
    if args.action == "status":
        rows = []
        for scene in scenes:
            path = results_root/"executions"/scene/"batch_status.json"
            if not path.exists():
                rows.append({"scene": scene, "status": "not_started"})
            else:
                rows.append(json.loads(path.read_text()))
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
    if args.action == "analyze":
        if not analysis_script.exists():
            raise RuntimeError("The analysis module will be implemented after experiment results are available")
        subprocess.run([sys.executable, str(analysis_script), "--protocol", str(protocol_path),
                        "--results-root", str(results_root), "--scenes", *scenes], check=True, cwd=ROOT)
        return
    device = resolve_device(args.device)
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
    cohort = results_root
    for scene in scenes:
        for seed in seeds:
            run_worker_group(SimpleNamespace(protocol=str(protocol_path), manifest=str(manifests[scene]),
                                            cohort=str(cohort), device=device, seed=seed))


if __name__ == "__main__":
    run(build_parser().parse_args())
