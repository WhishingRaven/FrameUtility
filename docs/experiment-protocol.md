# Experimental Protocol

This document defines the initial execution procedure for FrameUtility.

The protocol is divided into:

1. SLAM keyframe selection
2. contribution measurement
3. contribution-factor analysis
4. controlled camera degradation
5. model comparison

---

# Phase 0 — Environment Setup

## Step 0.1 — Initial Systems

Set up:

    VGGT-SLAM
    MASt3R-SLAM

Use fixed repository commits and record them.

## Step 0.2 — Evaluation Tools

Prepare evaluation for:

### Pose

- ATE
- RPE
- translation error
- rotation error

### Geometry

Where ground truth is available:

- depth error,
- point-cloud error,
- reconstruction completeness.

## Step 0.3 — Reproducibility

Fix:

- random seeds,
- model weights,
- preprocessing,
- image resolution,
- inference parameters,
- hardware/software versions.

Run identical inputs multiple times before beginning the main experiment.

Estimate normal run-to-run variance.

Contribution changes smaller than baseline variance should not be treated as meaningful.

---

# Phase 1 — Dataset Preparation

Select sequences with ground-truth camera trajectories and, preferably, ground-truth geometry/depth.

The initial dataset should contain different geometric conditions such as:

- straight corridors,
- corners,
- texture-rich rooms,
- texture-poor regions,
- repeated structures,
- depth-rich scenes.

Keep original sequences unchanged.

Record:

- frame IDs,
- timestamps,
- FPS,
- resolution,
- intrinsics,
- GT poses,
- GT depth/geometry where available.

---

# Phase 2 — Run Original Keyframe Selection

For each sequence:

## Step 2.1

Run VGGT-SLAM normally.

Do not modify its keyframe-selection heuristic.

Record:

    selected frames
    rejected frames

## Step 2.2

Run MASt3R-SLAM on the identical sequence.

Record the same information.

## Step 2.3

Save selector-specific internal values when accessible.

Examples may include:

- disparity,
- correspondence count,
- overlap,
- tracking score,
- keyframe threshold.

Suggested output:

| sequence | model | frame | timestamp | selected | selector_score |
|---|---|---:|---:|---|---:|

---

# Phase 3 — Baseline Reconstruction

For each system, let its original selected keyframe set be:

    K

Run or preserve the reconstruction produced using `K`.

Calculate:

    E(K)

Store:

- camera poses,
- depth,
- reconstructed geometry,
- ATE,
- RPE,
- geometry metrics,
- runtime.

This is the baseline for contribution experiments.

---

# Phase 4 — Selected-Frame Contribution

For every selected frame:

    Fi ∈ K

perform a leave-one-out experiment.

## Step 4.1

Construct:

    K_-i = K \ {Fi}

## Step 4.2

Run reconstruction using `K_-i`.

The keyframe set must be explicitly controlled.

Do not allow the original selector to simply replace `Fi` with another frame, since this would measure selector adaptation rather than the contribution of `Fi`.

## Step 4.3

Calculate:

    C_remove(i) = E(K_-i) - E(K)

## Step 4.4

Classify provisionally:

    positive
        useful

    approximately zero
        redundant

    negative
        potentially harmful

The threshold for "approximately zero" should be determined from experimental variance rather than chosen arbitrarily.

---

# Phase 5 — Rejected-Frame Contribution

For every rejected frame:

    Fj ∉ K

or for a representative sampled subset if the sequence is too large:

## Step 5.1

Construct:

    K_+j = K ∪ {Fj}

Insert the frame at its correct temporal position.

## Step 5.2

Run reconstruction.

## Step 5.3

Calculate:

    C_add(j) = E(K) - E(K_+j)

Interpretation:

    positive
        missed useful frame

    approximately zero
        redundant frame

    negative
        harmful frame correctly rejected

---

# Phase 6 — Construct the Selection–Contribution Dataset

Create one unified table.

Suggested schema:

| sequence | model | frame | selected | contribution | contribution_type |
|---|---|---:|---|---:|---|
| seq01 | VGGT-SLAM | 14 | true | 0.42 | remove |
| seq01 | VGGT-SLAM | 15 | false | 0.03 | add |
| ... | ... | ... | ... | ... | ... |

Add reconstruction metrics separately rather than immediately collapsing all metrics into one score.

---

# Phase 7 — Evaluate Keyframe Selection

Analyze:

## Selected Frames

What fraction are:

- strongly useful?
- weakly useful?
- redundant?
- harmful?

## Rejected Frames

What fraction are:

- useful but missed?
- redundant?
- harmful?

## Ranking Analysis

Compare:

    selector score / selection decision

against:

    measured contribution

Possible metrics:

- Spearman rank correlation,
- precision/recall after defining a useful-frame threshold,
- ROC/AUC where appropriate,
- contribution distribution of selected vs rejected frames.

The threshold-based analysis should be secondary to the continuous contribution analysis.

---

# Phase 8 — Extract Geometric Factors

For every frame calculate:

## Camera Motion

- relative translation,
- relative rotation,
- baseline.

## Geometric Novelty

- parallax,
- viewpoint change,
- feature/correspondence overlap,
- newly visible geometry if measurable.

## Scene Characteristics

- feature density,
- texture density,
- depth mean,
- depth variance.

Append these measurements to the contribution dataset.

---

# Phase 9 — Analyze Contribution Factors

Test relationships such as:

    contribution vs parallax
    contribution vs overlap
    contribution vs translation
    contribution vs rotation
    contribution vs depth variation

Calculate:

- Pearson correlation,
- Spearman correlation,
- partial correlations.

Fit multivariate models where necessary to account for correlated variables.

Do not interpret observational correlations as causal effects.

---

# Phase 10 — Identify Frames for Controlled Experiments

Select representative frames from different contribution classes:

    strongly useful
    moderately useful
    redundant
    harmful

Prefer frames from multiple:

- scenes,
- trajectories,
- viewpoint changes,
- texture conditions.

These frames become the basis of degradation experiments.

---

# Phase 11 — Controlled Motion Blur

For each selected clean frame:

    Fi(B0)
    Fi(B1)
    Fi(B2)
    Fi(B3)

where:

    B0 = clean
    B1 = low blur
    B2 = medium blur
    B3 = high blur

Only degradation severity should change.

Replace the original frame with each degraded version while keeping the rest of the sequence fixed.

Measure:

    C_i(B0)
    C_i(B1)
    C_i(B2)
    C_i(B3)

Calculate:

    ΔC_i(B) = C_i(B) - C_i(B0)

Plot:

    blur severity
          ↓
    frame contribution

Look for transitions:

    useful
      ↓
    weakly useful
      ↓
    redundant
      ↓
    harmful

---

# Phase 12 — Controlled Rolling Shutter

Use a physically meaningful rolling-shutter simulation or controlled renderer.

Keep fixed:

- scene,
- nominal camera trajectory,
- frame timestamps,
- intrinsics,
- illumination.

Vary:

    sensor readout time

Example:

    RS0 = global shutter
    RS1 = low
    RS2 = medium
    RS3 = high

Measure:

    C_i(RS0)
    C_i(RS1)
    C_i(RS2)
    C_i(RS3)

Calculate contribution loss relative to the global-shutter condition.

---

# Phase 13 — Blur × Rolling-Shutter Experiment

Generate:

| RS / Blur | B0 | B1 | B2 | B3 |
|---|---:|---:|---:|---:|
| RS0 | ✓ | ✓ | ✓ | ✓ |
| RS1 | ✓ | ✓ | ✓ | ✓ |
| RS2 | ✓ | ✓ | ✓ | ✓ |
| RS3 | ✓ | ✓ | ✓ | ✓ |

For every condition measure:

- frame contribution,
- pose error,
- depth error,
- reconstruction error.

Produce heatmaps:

    x-axis = blur severity
    y-axis = rolling-shutter severity
    value  = frame contribution

and:

    value = reconstruction error

Test for an interaction between blur and rolling shutter.

---

# Phase 14 — Context-Dependence Test

Leave-one-out contribution may underestimate frames with redundant neighbors.

Test this explicitly.

## Pair Removal

For neighboring frames:

    remove Fi
    remove Fj
    remove Fi and Fj

Compare:

    C_i
    C_j
    C_{i,j}

A strong joint effect despite weak individual effects indicates redundancy/interdependence.

## Random Subsets

If computationally feasible:

1. sample many frame subsets,
2. evaluate reconstruction,
3. compare performance when frame `i` is present versus absent.

Estimate expected marginal contribution:

    C_i ≈ E[Q(S ∪ {Fi}) - Q(S)]

This can later be extended to approximate Shapley values.

---

# Phase 15 — Compare VGGT-SLAM and MASt3R-SLAM

Use identical source sequences.

Compare:

### Selection

- keyframe density,
- selected-frame locations,
- agreement between selectors.

### Contribution

- average selected-frame contribution,
- redundant-frame rate,
- harmful-frame rate,
- missed-useful-frame rate.

### Degradation

- sensitivity to blur,
- sensitivity to rolling shutter,
- useful-to-harmful transition points.

### Failure Modes

Determine whether the two systems fail for the same frames and under the same visual conditions.

---

# Phase 16 — Robustness Checks

Repeat important experiments across:

- multiple sequences,
- multiple scenes,
- different frame rates,
- different motion patterns,
- different keyframe densities.

Use multiple evaluation metrics.

Check that conclusions are not caused by:

- one scene,
- one metric,
- one selector threshold,
- one model,
- stochastic inference variation.

---

# Expected Figures

The final study should ideally contain:

## Figure 1

Timeline showing:

    all frames
    VGGT-SLAM keyframes
    MASt3R-SLAM keyframes

## Figure 2

Distribution of measured contribution for:

    selected vs rejected frames

## Figure 3

Selection–Contribution matrix:

    useful selected
    redundant selected
    harmful selected
    useful rejected

## Figure 4

Frame contribution along the camera trajectory.

## Figure 5

Contribution vs:

    viewpoint change
    parallax
    overlap

## Figure 6

Contribution vs motion-blur severity.

## Figure 7

Contribution vs rolling-shutter severity.

## Figure 8

Blur × rolling-shutter contribution heatmap.

## Figure 9

Examples of:

    useful → redundant → harmful

transitions.

## Figure 10

VGGT-SLAM vs MASt3R-SLAM comparison.

---

# Final Questions

At the end of the project, the experiments should allow us to answer:

1. Do neural SLAM systems select frames with high actual contribution?
2. How many selected keyframes are redundant?
3. Are any selected frames actively harmful?
4. Do the systems miss useful frames?
5. What geometric properties characterize high-contribution frames?
6. How does realistic camera degradation change frame contribution?
7. Can degradation turn an otherwise useful frame into a harmful one?
8. Are these effects different between VGGT-SLAM and MASt3R-SLAM?

The central empirical comparison is:

    SLAM thinks this frame matters.

                VS

    Does it actually improve the reconstruction?