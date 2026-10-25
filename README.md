# FrameUtility

**When Does an Additional View Become Harmful?**

FrameUtility is an empirical study of the **marginal utility of degraded observations in multi-view 3D reconstruction**.

The project no longer treats the keyframe-selection heuristic of VGGT-SLAM or MASt3R-SLAM as the object of evaluation. Modern reconstruction systems are complete pipelines, and a keyframe decision is meaningful only in the context of the state produced by that pipeline. Instead, FrameUtility fixes the observation set and asks a controlled downstream question:

> Given a fixed set of valid views, when does adding one geometrically valid but degraded observation become worse than excluding it?

The initial reconstruction backends are:

- VGGT / VGGT-based reconstruction
- MASt3R-based reconstruction

The initial degradation is **motion blur**. Additional degradations may be added only after the core experiment is validated.

---

## 1. Motivation

Recent feed-forward and foundation-model-based 3D reconstruction methods can consume many images directly. Existing work already shows that:

- blur and other real-world degradations reduce pose and geometry accuracy;
- low-quality video frames can hurt classical SfM/VO pipelines;
- outlier or non-overlapping views can damage feed-forward reconstruction;
- view pruning can improve robustness.

Therefore, the novelty of this project is **not** to show that blur is harmful, nor to propose another blur detector or keyframe selector.

The narrower question is whether a degraded observation has a **negative marginal value** relative to withholding that observation altogether.

---

## 2. Core Quantity: Conditional Marginal Utility

Let the fixed context of clean observations be

```text
K_-i = {F1, ..., F(i-1), F(i+1), ..., Fn}
```

and let `Fi(b)` denote the same target observation at the same nominal viewpoint, with blur severity `b`.

Let `E(S)` be a reconstruction error measured from image set `S`.

Define the marginal utility of the degraded observation as

```text
U_i(b | K_-i) = E(K_-i) - E(K_-i ∪ {Fi(b)})
```

Interpretation:

```text
U_i(b | K_-i) > 0
    adding Fi(b) improves reconstruction

U_i(b | K_-i) ≈ 0
    adding Fi(b) provides negligible benefit

U_i(b | K_-i) < 0
    reconstruction is better when Fi(b) is withheld
    → negative marginal utility
```

The zero crossing is the central event:

```text
b* : U_i(b* | K_-i) = 0
```

`b*` is not an intrinsic property of a frame. It depends on scene, context, viewpoint, reconstruction backend, degradation type, and metric.

---

## 3. Why Context Dependence Is Not a Bug

Frame utility is conditional on the other available observations. Therefore the project does **not** claim that a frame has one universal contribution score.

The experimental unit is a tuple:

```text
(scene, context K_-i, target frame Fi, degradation severity b, backend)
```

The study repeats the intervention across many tuples and analyzes distributions such as:

```text
P(U < 0 | blur severity = b)
```

and, when a zero crossing exists,

```text
distribution of b*
```

This turns context dependence into a factor to sample over rather than a hidden assumption.

---

## 4. Research Questions

**RQ1.** How does progressive motion blur change the conditional marginal utility of an otherwise geometrically valid observation?

**RQ2.** Does a finite blur severity exist at which including the observation becomes worse than excluding it?

**RQ3.** How often does negative marginal utility occur across scenes, viewpoints, and observation contexts?

**RQ4.** Are the transition severity and failure pattern different between VGGT-based and MASt3R-based reconstruction?

**RQ5.** Which geometric/context variables shift the transition, such as baseline, overlap, redundancy, texture, or depth variation?

Optional extension:

**RQ6.** Do different degradation types produce qualitatively different utility curves?

---

## 5. Main Experiment

For each selected experimental tuple:

1. Construct a clean context `K_-i`.
2. Run the reconstruction backend on `K_-i` and measure `E_drop = E(K_-i)`.
3. Create controlled variants `Fi(b0), Fi(b1), ..., Fi(bm)` from the same target view.
4. For each blur severity, reconstruct from `K_-i ∪ {Fi(b)}`.
5. Compute `U_i(b | K_-i)`.
6. Repeat across target frames, contexts, scenes, and backends.

The important comparison is always:

```text
use the degraded observation
            VS
withhold the observation
```

not:

```text
selected by one SLAM heuristic
            VS
rejected by another heuristic
```

---

## 6. Controlled Motion Blur

Motion blur should approximate exposure integration along a camera path rather than a generic 2D blur kernel whenever possible.

For target pose `T(t0)`, sample camera poses over the exposure interval and render/warp intermediate observations:

```text
Fi(b) ≈ (1/M) Σ_m Warp(I, T_m)
```

For pure rotational perturbations, image warping can be represented by a homography without depth. For translation-dominated motion, blur is depth-dependent; a renderer or known scene geometry is preferable.

Blur severity must be recorded in a physically interpretable or reproducible unit, e.g. path length, angular displacement, exposure interval, or measured pixel blur.

---

## 7. Evaluation

Do not collapse all reconstruction quality into one scalar prematurely.

### Camera pose

- Absolute Trajectory Error (ATE)
- Relative Pose Error (RPE)
- translation error
- rotation error

### Depth / point maps

- scale-aligned depth error
- absolute relative error
- RMSE

### 3D geometry

- point-cloud precision / recall / F1
- Chamfer distance where appropriate
- completeness
- point-to-surface distance where GT geometry exists

For every metric, define the sign of utility consistently so that positive `U` means the added view helps.

---

## 8. Relation to Existing Work

The project must be positioned carefully.

- **RealX3D (2026)** benchmarks reconstruction under controlled real-world degradations, including motion blur. Our difference is the **add-vs-drop marginal-utility intervention** on one fixed geometrically valid observation.
- **Emergent Outlier View Rejection in VGGT (CVPR 2026)** studies irrelevant/non-overlapping distractor views and emergent rejection behavior. Our target is a **soft corruption of an otherwise valid view**.
- **G-MASt3R-SfM (2026)** removes geometrically unreliable/outlier views. Our experiment does not propose a pruning algorithm; it measures how utility changes continuously with degradation severity.
- Classical SfM/VO work has already shown that blur-aware frame filtering can improve reconstruction. Therefore “blurred frames are bad” is not a contribution.

See `literature-review.md` for the reading list and detailed summaries.

---

## 9. Expected Contribution

A successful project should contribute an empirical characterization, not merely another robustness score.

Desired outputs:

- marginal-utility curves `U(b)`;
- probability of negative marginal utility as a function of blur severity;
- distribution of zero-crossing severity `b*`;
- comparison between reconstruction backends;
- analysis of which scene/context factors shift the transition;
- examples where a view remains geometrically valid but becomes actively harmful.

The main question is:

> **When is more visual information worse than less?**

---

## Running the Experiment

The restart plan is in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). The initial primary experiment is **office**, using **5/10/15 views**, with one execution per condition at seed 2026. Office remains part of the main analysis. The other eleven eligible-size scene candidates use the same procedure. Every scene supports `--target local|cuda-server`, with `local` as the default; the server target forwards commands over SSH using a configured profile. The plan contains 726 reconstructions for office and at most 8,712 across all scenes, excluding recovery attempts and recomputations after fixes.

[experiments/run_all.py](experiments/run_all.py) provides the same prepare/run/status/analyze interface on Mac and CUDA servers. Execution location is selected with `--target` independently of scene names, and GPU selection and runtime paths are automatic; VGGT uses one worker and MASt3R uses three, with inference and alignment on the same GPU. See [the interface and readiness documentation](experiments/README.md).

For a teammate preparing a TA-provided Amazon GPU server, follow the [CUDA server setup guide](CUDA_SERVER_SETUP.md).

The plan is **awaiting user approval**. Previous results and inference caches have been deleted from the working tree. The new experiment will not start until the plan is approved and its preparation, evaluation and analysis implementation is verified.
