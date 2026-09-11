# Experimental Design

## 1. Objective

FrameUtility measures how the value of one observation changes as its image quality degrades while its nominal geometry and viewpoint are held fixed.

The study does **not** audit the keyframe-selection heuristics of VGGT-SLAM or MASt3R-SLAM. The reconstruction backend receives an explicitly controlled image set.

The central estimand is

```text
U_i(b | K_-i) = E(K_-i) - E(K_-i ∪ {Fi(b)})
```

where positive values mean that the additional observation improves reconstruction and negative values mean that it is better to omit the observation.

---

## 2. Experimental Unit

Each sample is defined by:

```text
scene
context image set K_-i
target observation Fi
degradation type
severity b
reconstruction backend
```

The study must sample multiple contexts because marginal utility is context-dependent.

---

## 3. Independent Variables

### Primary independent variable

Motion-blur severity `b`.

Recommended levels:

```text
b0 = clean
b1 = very low
b2 = low
b3 = medium
b4 = high
b5 = severe
```

Use enough levels to identify a nonlinear transition. Three coarse levels are insufficient if the goal is to estimate a zero crossing.

### Backend

- VGGT / VGGT-based reconstruction
- MASt3R-based reconstruction

### Context factors

Record rather than artificially fix all of the following:

- number of context views;
- target-view overlap with context;
- camera baseline;
- relative translation and rotation;
- target-frame redundancy;
- texture / feature density;
- depth variation;
- scene type.

---

## 4. Dependent Variables

The primary dependent variable is conditional marginal utility.

For error metric `E_m`:

```text
U_i,m(b | K_-i) = E_m(K_-i) - E_m(K_-i ∪ {Fi(b)})
```

Do this separately for pose and geometry metrics.

Secondary outputs:

- absolute reconstruction error at every severity;
- change relative to clean target view;
- whether `U < 0`;
- estimated zero-crossing severity `b*`;
- model confidence where accessible.

---

## 5. Main Controlled Experiment

For each `(scene, K_-i, Fi)`:

1. Reconstruct from `K_-i` only. This is the **drop baseline**.
2. Reconstruct from `K_-i ∪ {Fi(clean)}`.
3. Generate blur variants of the same observation.
4. Reconstruct independently for every blur level.
5. Compute utility for every evaluation metric.
6. Fit or interpolate the utility curve only if justified by observed behavior.

The clean target should normally have positive marginal utility; otherwise that tuple is not suitable for studying a useful-to-harmful transition, though it can remain in the broader dataset as a redundant-view case.

---

## 6. Motion-Blur Synthesis

### Preferred method

Physics-consistent exposure integration using known camera trajectory and scene geometry.

```text
I_blur(x) = (1/M) Σ I(T_m, x)
```

where `T_m` samples the camera pose during exposure.

For translational camera motion, do not use a single spatially invariant 2D kernel: the blur field depends on scene depth.

### Simpler first-stage method

If the target perturbation is restricted to pure rotation, the view change can be warped by homographies and averaged without depth.

### Severity representation

Record both the generative parameter and a measured image-domain quantity when possible:

- exposure-path length or angular displacement;
- approximate mean / percentile pixel displacement;
- sharpness score for diagnostics only.

The sharpness score is not the treatment variable.

---

## 7. Sampling Contexts

A single fixed image set cannot support a general claim. Use multiple contexts.

Recommended strategy:

1. choose several scenes;
2. sample target observations with valid GT and adequate overlap;
3. construct several context sizes around each target;
4. include both redundant and non-redundant contexts;
5. repeat the same blur sweep.

This permits analysis of whether degradation becomes harmful earlier when the target view is redundant or later when it provides unique geometric coverage.

---

## 8. Hypotheses

**H1 — Monotonic utility loss.** Increasing blur will reduce the median marginal utility of a geometrically valid observation.

**H2 — Negative marginal utility.** At sufficiently strong blur, a nonzero subset of observations will satisfy `U < 0`.

**H3 — Context dependence.** The transition severity will vary substantially with overlap, redundancy, texture, and baseline.

**H4 — Model dependence.** VGGT-based and MASt3R-based reconstruction will show different utility curves and failure frequencies.

**H5 — Soft-corruption distinction.** A valid but degraded view will produce failure behavior distinct from fully unrelated/no-overlap distractors studied in outlier-view rejection work.

---

## 9. Statistical Analysis

Primary plots:

- individual `U(b)` trajectories;
- mean/median utility with confidence intervals;
- `P(U < 0 | b)`;
- distribution of `b*` for curves that cross zero.

Model context effects with mixed-effects or hierarchical regression if sample size permits:

```text
U ~ blur + overlap + baseline + redundancy + texture + backend
    + blur:backend + blur:overlap
    + random effect(scene)
```

Avoid claiming a universal blur threshold unless the data support one.

---

## 10. Robustness Checks

- repeat deterministic inference to verify numerical stability;
- fix preprocessing, resolution, and model versions;
- verify that the clean-view benefit exceeds normal run variance;
- repeat with multiple context sizes;
- repeat with several error metrics;
- inspect catastrophic-failure cases separately;
- report missing reconstruction / solver failure as outcomes, not silently discard them.

---

## 11. Optional Extensions

Only after the motion-blur experiment works:

- defocus blur;
- exposure / low-light degradation;
- sensor noise;
- partial occlusion;
- reflection;
- rolling shutter.

The useful extension is not “more corruptions” by itself. The value is comparing whether different corruptions produce different **marginal-utility transition patterns**.

---

## 12. Scientific Claim Boundary

The project can support:

> Under controlled contexts, progressively degrading a geometrically valid observation changes its marginal contribution to reconstruction, and under some conditions including the observation is worse than withholding it.

It cannot support, without additional experiments:

> A particular SLAM keyframe selector is wrong.

or

> A frame has an intrinsic, context-independent contribution value.
