# FrameUtility

**Do Modern Neural SLAM Systems Select the Frames That Actually Matter?**

FrameUtility is an empirical study of frame contribution and keyframe selection in modern neural SLAM systems.

The central questions are:

> How much does each frame actually contribute to geometry estimation?

and

> Do modern SLAM systems actually select the frames with the highest contribution?

The initial targets are:

- VGGT-SLAM
- MASt3R-SLAM

---

## 1. Motivation

A video used for SLAM may contain thousands of frames, but processing every frame is unnecessary and computationally expensive.

SLAM systems therefore perform **keyframe selection**: only a subset of frames is retained for mapping and geometry estimation.

Existing systems typically select keyframes using heuristics based on quantities such as:

- viewpoint change,
- feature/correspondence overlap,
- optical flow or disparity,
- camera motion,
- tracking quality.

However, a frame satisfying a keyframe-selection heuristic is not necessarily a frame that actually improves the final reconstruction.

This motivates a different question:

> Are the frames selected by a SLAM system actually the frames that contribute most to its reconstruction?

We therefore distinguish between:

    Predicted Importance
        = importance implied by the SLAM keyframe-selection heuristic

and

    Measured Contribution
        = actual change in reconstruction quality caused by a frame

The goal of FrameUtility is to compare the two.

---

## 2. Core Concept: Frame Contribution

Suppose a SLAM system selects the keyframe set:

    K = {F1, F3, F6, F8, F10}

We first run reconstruction using the complete selected set and obtain:

    E(K)

where `E` represents reconstruction error.

### Selected Frames: Leave-One-Out

For a selected frame `Fi`, remove it:

    K_-i = K \ {Fi}

and run reconstruction again.

Define:

    C_remove(i) = E(K_-i) - E(K)

Interpretation:

    C_remove(i) > 0
    → removing the frame makes reconstruction worse
    → useful selected frame

    C_remove(i) ≈ 0
    → removing the frame changes little
    → redundant selected frame

    C_remove(i) < 0
    → removing the frame improves reconstruction
    → potentially harmful selected frame

---

## 3. Rejected Frames: Add-One-In

Selected frames alone are not sufficient to evaluate a keyframe selector.

A system may reject a frame that would actually have improved reconstruction.

For a rejected frame `Fj`, add it to the selected set:

    K_+j = K ∪ {Fj}

and measure:

    C_add(j) = E(K) - E(K_+j)

Interpretation:

    C_add(j) > 0
    → adding the rejected frame improves reconstruction
    → potentially missed useful frame

    C_add(j) ≈ 0
    → rejected frame provides little additional information
    → appropriate rejection

    C_add(j) < 0
    → adding the frame makes reconstruction worse
    → appropriate rejection of a harmful frame

This produces four important cases:

| SLAM Decision | Measured Contribution | Interpretation |
|---|---|---|
| Selected | High positive | Good selection |
| Selected | Near zero | Redundant selection |
| Selected | Negative | Harmful selection |
| Rejected | High positive | Missed useful frame |
| Rejected | Near zero / negative | Good rejection |

---

## 4. Frame Contribution Is Context-Dependent

A frame does not necessarily have one intrinsic contribution value.

Its contribution depends on which other frames are already present.

For example, two neighboring frames may contain nearly identical information.

Removing either one individually may have little effect, while removing both may significantly damage reconstruction.

Therefore, contribution should formally be considered:

    C_i(K)

rather than simply:

    C_i

The initial experiments use leave-one-out and add-one-in measurements because they are simple and interpretable.

Later experiments may estimate more general contribution using random subsets or approximate Shapley values.

---

## 5. What Determines Frame Contribution?

After measuring contribution, we investigate why some frames are useful while others are redundant or harmful.

We hypothesize that frame utility depends on the interaction between:

    Geometric Information

and

    Image Degradation

Rather than assuming a fixed linear relationship, we write:

    Frame Utility = f(Geometric Information, Degradation)

### Geometric Factors

Candidate variables include:

- viewpoint change,
- parallax,
- camera translation,
- camera rotation,
- feature/correspondence overlap,
- texture / feature density,
- depth variation,
- visibility of previously unseen geometry.

### Degradation Factors

Initial focus:

- motion blur,
- rolling-shutter distortion.

---

## 6. Why Degradation Requires Controlled Experiments

Geometric information and image degradation are naturally confounded in real video.

For example, fast camera rotation may simultaneously produce:

    viewpoint change ↑
    optical flow ↑
    feature overlap ↓
    motion blur ↑
    rolling-shutter distortion ↑

Therefore, simple correlations from smartphone video cannot reliably isolate the effect of degradation.

FrameUtility separates the study into:

1. observational keyframe/contribution analysis,
2. controlled degradation experiments.

---

## 7. Research Questions

### Keyframe Selection

**RQ1.**
How much does the measured contribution of individual frames vary?

**RQ2.**
Do VGGT-SLAM and MASt3R-SLAM select frames with high measured contribution?

**RQ3.**
How often do the systems select redundant or harmful frames?

**RQ4.**
How often do they reject frames that would have significantly improved reconstruction?

### Contribution Factors

**RQ5.**
Which geometric properties best explain measured frame contribution?

### Camera Degradation

**RQ6.**
How does motion blur change the contribution of an otherwise identical frame?

**RQ7.**
How does rolling-shutter distortion change frame contribution?

**RQ8.**
Can a geometrically valuable frame become redundant or harmful under sufficiently severe degradation?

**RQ9.**
Do motion blur and rolling shutter interact nonlinearly?

### Model Difference

**RQ10.**
Do VGGT-SLAM and MASt3R-SLAM exhibit different frame-selection and failure patterns?

---

## 8. Experimental Structure

### Experiment 1 — Keyframe Selection Audit

Run:

- VGGT-SLAM
- MASt3R-SLAM

on the same sequences.

Record:

- selected frames,
- rejected frames,
- selector statistics,
- camera trajectory,
- reconstruction output.

Goal:

> Observe what each system chooses.

---

### Experiment 2 — Contribution Measurement

For selected frames:

    leave-one-out

For rejected frames:

    add-one-in

Measure the resulting change in reconstruction quality.

Goal:

> Determine whether the system's selection decisions correspond to actual reconstruction contribution.

---

### Experiment 3 — Contribution Factor Analysis

Measure geometric characteristics for each frame and compare them with measured contribution.

Goal:

> Determine what actually makes a frame useful.

---

### Experiment 4 — Controlled Degradation

Take geometrically identical frames or controlled synthetic renderings and systematically vary:

- motion blur,
- rolling shutter.

Goal:

> Determine how realistic camera degradation destroys otherwise useful geometric information.

---

### Experiment 5 — Blur × Rolling-Shutter Interaction

Use a factorial design:

| Rolling Shutter / Blur | B0 | B1 | B2 | B3 |
|---|---:|---:|---:|---:|
| RS0 | ✓ | ✓ | ✓ | ✓ |
| RS1 | ✓ | ✓ | ✓ | ✓ |
| RS2 | ✓ | ✓ | ✓ | ✓ |
| RS3 | ✓ | ✓ | ✓ | ✓ |

Goal:

> Determine whether the two artifacts produce additive or nonlinear degradation.

---

## 9. Evaluation

Candidate metrics include:

### Camera Pose

- Absolute Trajectory Error (ATE)
- Relative Pose Error (RPE)
- translation error
- rotation error

### Depth

- absolute relative error
- RMSE
- scale-aligned depth error

### 3D Geometry

- point-to-point error
- point-to-surface error
- Chamfer distance
- reconstruction completeness
- reconstruction consistency

Multiple metrics should be retained rather than forcing all geometry quality into one scalar too early.

---

## 10. Expected Contribution

This project is not primarily intended to propose another keyframe-selection heuristic.

Instead, it aims to evaluate the assumptions behind existing keyframe selection.

The central comparison is:

    What the SLAM system selects
                VS
    What actually improves reconstruction

The desired outcome is to identify:

- useful selected frames,
- redundant selected frames,
- harmful selected frames,
- useful frames that existing selectors miss.

The second objective is to explain these outcomes using geometric information and realistic camera degradation.

Ultimately, the project asks:

> Do modern neural SLAM systems know which frames actually matter?

and, more fundamentally:

> What makes a frame useful—or harmful—to geometry estimation?