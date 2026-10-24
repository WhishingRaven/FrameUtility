# FrameUtility experiment runner

Implements `실험 순서.txt` and `Proposal.pdf` using fixed inputs and ADD/DROP reconstruction.

- Dataset: one ETH3D high-resolution training scene, `office` (exploratory run).
- Select 20 equally spaced filenames from the 26 calibrated views. Build nested 10/5-view subsets by removing the lowest mean-overlap view while requiring the GT laser-visibility graph (edge threshold 0.10) to remain connected after every possible single-view DROP. Selection precedes all model inference and uses no model outputs. Each size includes the target.
- Every view in each subset is a target: 35 target/context tuples per backend.
- Exposure yaw paths: 0, 0.25, 0.5, 1, 2, 4 degrees. A symmetric pure rotation is integrated with 33 samples in linear sRGB. The same fixed overscan crop is applied to clean and blurred images; invalid warp borders are rejected.
- Only one view changes per intervention. Models get all explicitly listed inputs with no adaptive keyframe selection.
- Evaluate both reconstructions on the same remaining context cameras and the same reference pixels. The target is excluded from evaluation because it does not exist in DROP.
- Pose: Sim(3)-aligned ATE divided by the GT camera-center RMS radius, mean pairwise translation error divided by each GT baseline, and mean pairwise rotation error in degrees. For each pose metric use dimensionless `U = (E_drop - E_add) / E_drop`. Preserve raw differences as well. Near-zero denominators produce a null relative utility.
- Geometry: one median depth scale over the entire common context, then depth AbsRel and RMSE. Use `U = E_drop - E_add`, separately for each metric. This measures scale-aligned depth accuracy, not reconstruction completeness.
- A nonpositive or nonfinite predicted depth at any fixed evaluation pixel makes that condition's geometry metrics fail. Keep the fixed mask; never filter or clamp those predictions to obtain a score. Pose metrics are evaluated independently. A failure in clean repeats, DROP, or ADD is recorded for its metric family and does not terminate other conditions.
- Geometry reference: the two provided ETH3D laser scans, transformed using `scan_alignment.mlp` and nearest-depth rasterized into calibrated views. No mesh filling, image masks, or official ETH3D occlusion files are applied. This is an exploratory depth reference, not an official ETH3D benchmark score. The official depth download was unavailable during initial setup.
- VGGT uses its official 518px preprocessing; MASt3R uses its official 512px preprocessing. The same prepared images and angular perturbations are supplied, with each backend's documented resize/crop applied to its reference depth as well. Intrinsics and GT are used only for synthesis/evaluation, never as reconstruction inputs.
- MASt3R uses official sparse global alignment, a complete symmetric graph, 300 coarse + 300 fine iterations. Only independent pairwise inference is cached across conditions; canonical views and optimization are recomputed. Inference and alignment use the selected device by default. `--optimizer-device cpu` reproduces CPU alignment; `--optimizer-device mps` or `cuda` selects accelerator alignment explicitly.

## Environment

Use the project `.venv` (Python 3.11). Packages are recorded in `requirements-lock.txt` after installation. Source versions:

```
VGGT   a288dd0f14786c93483e45524328726ab7b1b4ce
MASt3R f5209afc300cec36239a7ac992263f36847bbba0
DUSt3R 3cc8c88c413bb9e34c41db0e0eef99c2ee010b12
CroCo  d7de0705845239092414480bd829228723bf20de
```

Official weights are stored in `checkpoints/vggt-1b.pt` and `checkpoints/mast3r.safetensors`, with MASt3R's `config.json` saved as `mast3r-config.json`. The runner records weight/input/code SHA-256 checksums, versions, device, precision and seed. Keep `vendor`, `checkpoints`, prepared data and prediction arrays outside Git.

Pinned checkpoint revisions, official download URLs, byte counts and SHA-256 digests are in `weights.json`. Incomplete or mismatched downloads are rejected before inference. Float32 matrix multiplication uses `highest` precision and CUDA TF32 convolutions are disabled. Runtime records include elapsed inference/alignment time (excluding model loading) and cumulative process peak RAM.

## Execute

From the repository root, after extracting `office` from the existing undistorted-image and scan-evaluation archives into `data/prepared/eth3d`:

```sh
.venv/bin/python experiments/prepare.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v

# Initial 5-view pilot; repeat the clean condition to measure numerical variation.
.venv/bin/python experiments/run.py --backend vggt --sizes 5 --targets 1 --repeat-clean --output results/office_sequential
.venv/bin/python experiments/run.py --backend mast3r --sizes 5 --targets 1 --repeat-clean --output results/office_sequential

# Resume and extend to all targets and nested sets.
.venv/bin/python experiments/run.py --backend vggt --repeat-clean --output results/office_sequential
.venv/bin/python experiments/run.py --backend mast3r --repeat-clean --output results/office_sequential
.venv/bin/python experiments/summarize.py results/office_sequential
```

To execute both backends sequentially with periodic CSV/figure/report updates:

```sh
.venv/bin/python experiments/run_all.py --device mps --output results/office_sequential
```

The current accelerated local run is `results/office_fast`, using three MASt3R workers with Metal inference and alignment:

```sh
.venv/bin/python experiments/run_parallel.py --device mps --optimizer-device mps --workers 3 --threads 8 --pair-cache-source results/office_sweep --output results/office_fast
```

It reuses the completed VGGT results and compatible neural pair cache from the original run. All MASt3R clean, DROP and ADD alignments are recomputed together on MPS, in a separate result root, because CPU and GPU floating-point results can differ. Previous CPU results remain in `results/office_sweep`.

Workers own disjoint target frames across all context sizes. Baseline prediction locks avoid duplicate clean runs; locked read/modify/write operations preserve every worker's metric rows and repeatability records. The shared-pair source is accepted only when its inputs, pretrained weights, inference device, precision, package/source versions and CPU thread setting match. Alignment state is never shared across conditions.

`run_parallel.py` reports each worker PID and active condition in `batch_status.json`. SIGTERM to its controller requests a stop after the current reconstructions finish; the same command resumes saved predictions. `--threads` is per worker. Benchmark evidence is in `results/performance_tuning_20261024/report.md`.

`batch_status.json` records the driver/worker PIDs, active phase, evaluated conditions, fully valid/partial/failed counts and phase return codes. Per-phase logs are saved under each backend directory. Interrupting the driver terminates its worker; rerunning resumes saved predictions. `complete` means all requested conditions have valid results; `complete_with_failures` means every condition was evaluated but some metric families failed. `partial` means conditions remain unevaluated.

`--max-new-runs N` limits new reconstructions. `--device cpu|mps|cuda|auto` selects inference hardware. Use separate `--output` roots for different environments, source changes or hardware. Existing predictions are reused only when the recorded fingerprint matches. Failed conditions remain explicit and are retried on resume.

There are 420 planned utility conditions (210 per backend), each with five separate metric rows. Clean reconstruction is shared within an image subset; each target has its own DROP. Thus the sweep needs 426 unique reconstructions, plus six clean repeats when enabled. Saving a clean reconstruction once does not equate to saving one utility: evaluation still excludes each target separately.

The local run's 2026-10-24 failure-handling repair is documented in `results/office_sweep/recovery_20261024/verification.json`. Before resuming, all 242 previously successful conditions (1,210 metric rows) were checked for exact numeric equality, the original source/environments were archived, and completed prediction checksums were recorded. Model/input/inference settings were unchanged. This documented continuation updates the recorded control/evaluation source versions while reusing the original predictions; normal runs continue to reject unverified fingerprint changes.

## CUDA handoff

Copy the source, `vendor` at the commits above, `checkpoints`, and `data/experiments/office` to the server, preserving their relative paths. The prepared manifest uses no machine-specific absolute paths. Do not copy the Mac `.venv` to Linux. Create a Python 3.11 virtual environment on the server and install PyTorch for that server's CUDA runtime, followed by the other dependencies. CUDA RoPE kernels are optional; the upstream PyTorch fallback works without compilation.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install torch torchvision numpy scipy pillow opencv-python huggingface_hub einops matplotlib tqdm roma plyfile safetensors trimesh
.venv/bin/python experiments/run.py --backend vggt --device cuda --output results/office_cuda --repeat-clean
.venv/bin/python experiments/run.py --backend mast3r --device cuda --output results/office_cuda --repeat-clean
.venv/bin/python experiments/summarize.py results/office_cuda
```

The single-command equivalent is `.venv/bin/python experiments/run_all.py --device cuda --output results/office_cuda`.

For the parallel MASt3R runner on a fresh CUDA environment, omit the MPS cache source and start with `.venv/bin/python experiments/run_parallel.py --device cuda --workers 1 --output results/office_cuda`. Increase worker count according to the server's measured throughput and available memory.

Keep CPU/MPS/CUDA outputs in separate roots. Within one root, rerunning the same command resumes completed reconstructions; a different device/software/code fingerprint is rejected to prevent mixing measurements. The reference images, severities, nested image sets, evaluation rules and pretrained weights remain identical across hardware.

Only complete observed curves support crossing summaries. Report already-negative clean conditions separately; do not infer a transition for them. Repeated clean inference measures representative numerical variation, not confidence intervals over scenes. Multiple scenes/contexts and variance checks are required before general claims.
