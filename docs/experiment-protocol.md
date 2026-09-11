# Experimental Protocol

This document defines the executable protocol for FrameUtility.

---

# Phase 0 — Freeze the Environment

## Step 0.1 — Backends

Set up and record exact commits / versions for:

- VGGT or the selected VGGT reconstruction path;
- MASt3R or the selected MASt3R reconstruction path.

The experiment must bypass adaptive keyframe selection. The backend should receive an explicitly specified image set.

## Step 0.2 — Reproducibility

Fix:

- model weights;
- image resolution;
- preprocessing;
- camera-intrinsic handling;
- inference options;
- random seeds where relevant;
- hardware and software versions.

Run identical input sets repeatedly and measure normal numerical variance.

---

# Phase 1 — Dataset Preparation

Select sequences/scenes with:

- accurate camera poses;
- camera intrinsics;
- preferably GT depth, point cloud, or mesh;
- enough neighboring viewpoints to construct multiple contexts.

Store a manifest:

| scene | frame | timestamp | image | intrinsics | GT pose | GT depth/geometry |
|---|---:|---:|---|---|---|---|

Keep original images immutable.

---

# Phase 2 — Choose Target Observations and Contexts

For each scene:

1. choose candidate target frame `Fi`;
2. choose a context `K_-i` that excludes `Fi`;
3. verify that all context views and the target belong to the same static scene;
4. record overlap and relative-pose statistics;
5. optionally construct several context sizes for the same target.

Do not derive `K_-i` from one system's keyframe selection if the goal is backend-neutral comparison.

---

# Phase 3 — Drop Baseline

For each `(scene, K_-i, Fi)`:

1. run reconstruction using only `K_-i`;
2. compute all evaluation metrics;
3. store output as `DROP`.

```text
E_drop,m = E_m(K_-i)
```

This is the reference against which every target-view variant is evaluated.

---

# Phase 4 — Clean Add Baseline

Add the clean target view:

```text
K_clean = K_-i ∪ {Fi(clean)}
```

Run reconstruction and compute:

```text
U_clean,m = E_drop,m - E_m(K_clean)
```

Classify the tuple:

- clearly positive clean utility: suitable for useful-to-harmful transition analysis;
- near-zero clean utility: redundant-view case;
- negative clean utility: already-harmful clean-view case; retain separately and inspect.

“Near zero” must be defined relative to measured run variance or a predeclared practical tolerance.

---

# Phase 5 — Generate Motion-Blur Variants

For every target frame, generate a reproducible severity ladder.

Example manifest:

| variant | path length | angular displacement | measured pixel blur |
|---|---:|---:|---:|
| B0 | 0 | 0 | 0 |
| B1 | ... | ... | ... |
| B2 | ... | ... | ... |
| B3 | ... | ... | ... |
| B4 | ... | ... | ... |
| B5 | ... | ... | ... |

Preferred synthesis:

1. sample camera poses during exposure;
2. render or depth-aware warp each sample;
3. average radiance / image samples;
4. keep nominal target timestamp and central/end pose definition consistent across severities.

For a first implementation, pure rotational blur may be used because homographic warping avoids depth-dependent errors.

---

# Phase 6 — Add-vs-Drop Sweep

For each severity `b`:

1. construct `K_b = K_-i ∪ {Fi(b)}`;
2. run the backend on exactly `K_b`;
3. compute metrics;
4. calculate marginal utility.

```text
U_i,m(b | K_-i) = E_drop,m - E_m(K_b)
```

Suggested result table:

| scene | target | context_id | backend | blur | metric | E_drop | E_add | utility |
|---|---:|---|---|---:|---|---:|---:|---:|

---

# Phase 7 — Detect Negative-Utility Transitions

For each curve:

```text
b -> U(b)
```

record:

- maximum utility;
- utility loss from clean;
- first observed severity with `U < 0`;
- interpolated `b*` only when neighboring samples straddle zero and interpolation is defensible;
- whether reconstruction fails entirely.

Do not force every curve to have a crossing.

---

# Phase 8 — Repeat Across Contexts

Repeat Phases 3–7 for:

- multiple target frames;
- multiple scenes;
- multiple context sizes;
- low- and high-redundancy contexts;
- different baselines / relative motions;
- different texture and depth conditions.

This phase is essential. A single context only demonstrates a case study.

---

# Phase 9 — Backend Comparison

Run identical image sets through each backend.

Compare:

- median `U(b)`;
- `P(U < 0 | b)`;
- failure rate;
- distribution of `b*`;
- sensitivity to overlap and context redundancy.

Do not compare their native keyframe decisions in the main study.

---

# Phase 10 — Context-Factor Analysis

For every tuple, record:

### Relative geometry

- translation;
- rotation;
- baseline;
- approximate parallax;
- view overlap.

### Scene/context properties

- texture density;
- depth variation;
- number of context views;
- target redundancy / nearest-view similarity;
- visibility of unique scene regions where measurable.

Analyze how these variables shift utility curves or the probability of negative utility.

---

# Phase 11 — Sanity Checks

### Check A — clean utility

Verify that the clean target usually helps in the subset used for transition claims.

### Check B — blur synthesis

Visually inspect sampled variants and ensure stronger severity corresponds to a larger exposure path / image displacement.

### Check C — geometry preservation

Verify that nominal viewpoint and scene content are unchanged except for the image-formation integration.

### Check D — deterministic input set

Ensure the backend does not silently replace, reject, or augment views.

### Check E — numerical variance

Repeat representative conditions. Effects smaller than baseline variance should not be interpreted.

---

# Phase 12 — Primary Figures

**Figure 1.** Experimental schematic: fixed `K_-i`, one target view, blur sweep, add-vs-drop comparison.

**Figure 2.** Example `U(b)` curves for several targets.

**Figure 3.** Probability `P(U < 0)` versus blur severity.

**Figure 4.** Distribution of `b*`.

**Figure 5.** VGGT vs MASt3R utility curves under identical contexts.

**Figure 6.** Utility / transition severity versus overlap, baseline, redundancy, and texture.

**Figure 7.** Qualitative reconstructions for clean-add, harmful-blur-add, and drop conditions.

---

# Phase 13 — Optional Multi-Degradation Extension

If time permits, repeat the same protocol for another degradation while preserving the add-vs-drop design.

Priority candidates:

1. defocus blur;
2. exposure / low light;
3. partial occlusion;
4. sensor noise;
5. rolling shutter.

The dependent variable and drop baseline remain unchanged.

---

# Final Questions

The project should ultimately answer:

1. Does progressive degradation continuously reduce the marginal value of an otherwise valid observation?
2. Can its marginal utility become negative?
3. At what severities and under what contexts does this happen?
4. How common is the phenomenon across scenes?
5. Does the transition differ between reconstruction backends?
6. Which context variables make a degraded observation easier or harder to tolerate?

The central empirical comparison is:

```text
K_-i ∪ {degraded Fi}
          VS
K_-i
```
