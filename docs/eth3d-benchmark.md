# ETH3D Benchmark Plan

## 1. Purpose

FrameUtility uses ETH3D as the primary benchmark for evaluating frame contribution and keyframe selection.

ETH3D is well suited to this project because it provides:

- monocular RGB sequences,
- accurate camera calibration,
- ground-truth camera trajectories,
- synchronized global-shutter imagery,
- scenes suitable for SLAM evaluation,
- additional synthetic data involving rolling-shutter distortion.

The benchmark will be used to answer two main questions:

1. Do VGGT-SLAM and MASt3R-SLAM select frames that actually improve reconstruction?
2. What geometric and visual properties determine the measured contribution of each frame?

---

# 2. Role of ETH3D

ETH3D will serve as the controlled benchmark dataset before experiments on real smartphone video.

The overall workflow is:

    ETH3D RGB sequence
            ↓
    VGGT-SLAM / MASt3R-SLAM
            ↓
      Keyframe Selection
            ↓
    Selected / Rejected Frames
            ↓
    Contribution Measurement
            ↓
      GT Pose Evaluation

Only monocular RGB images are provided to the SLAM systems.

Ground-truth trajectory and geometry are used only for evaluation.

This preserves the monocular setting of the experiment.

---

# 3. Why ETH3D

## 3.1 Ground-Truth Camera Pose

ETH3D provides accurate ground-truth camera trajectories for its SLAM benchmark sequences.

This allows estimated camera trajectories to be evaluated using metrics such as:

- Absolute Trajectory Error (ATE)
- Relative Pose Error (RPE)
- translation error
- rotation error

Without ground truth, it would be difficult to determine whether removing or adding a frame actually improves the SLAM result.

---

## 3.2 Global-Shutter Baseline

Many ETH3D sequences were captured using synchronized global-shutter cameras.

This is important because the first stage of the project aims to study frame contribution while minimizing rolling-shutter distortion.

The global-shutter sequences therefore provide a relatively clean baseline for analyzing geometric frame contribution.

---

## 3.3 Monocular Evaluation

Although ETH3D contains data for multiple sensor configurations, this project will use only a single RGB camera stream as input.

The experiment therefore remains:

    Monocular RGB
        ↓
    Neural SLAM

Additional sensor information is not provided to the SLAM system.

---

## 3.4 Rolling-Shutter Experiments

ETH3D also provides synthetic SLAM data containing controlled rolling-shutter effects.

Relevant conditions include variants such as:

    clean
    rolling shutter
    asynchronous sensing
    asynchronous + rolling shutter

These data may be useful for validating the controlled degradation experiments without requiring a physical rolling-shutter simulation pipeline in the initial stage.

However, the exact distortion parameters and suitability for the planned factorial experiments must be checked before using them as the sole rolling-shutter benchmark.

---

# 4. Initial Benchmark Systems

The same ETH3D RGB sequence will be processed independently by:

    VGGT-SLAM

and:

    MASt3R-SLAM

For each system, record:

- selected keyframes,
- rejected frames,
- estimated trajectory,
- reconstruction output,
- keyframe-selection score or internal decision variable when accessible,
- runtime.

The original keyframe-selection logic should remain unchanged during the initial audit.

---

# 5. Baseline Run

For each sequence and SLAM system:

## Step 1

Provide the full monocular RGB sequence.

## Step 2

Run SLAM normally.

Let the selected keyframe set be:

    K = {F_k1, F_k2, ..., F_kn}

## Step 3

Save:

- selected frame indices,
- rejected frame indices,
- estimated camera poses,
- GT-aligned trajectory,
- ATE,
- RPE,
- available reconstruction outputs.

## Step 4

Define the baseline reconstruction error:

    E(K)

This baseline is used for subsequent contribution measurements.

---

# 6. Contribution of Selected Frames

For every selected frame:

    Fi ∈ K

remove the frame from the selected set:

    K_-i = K \ {Fi}

Run the reconstruction again while keeping the remaining keyframe set fixed.

Do not allow the SLAM keyframe selector to automatically replace the removed frame.

Measure:

    E(K_-i)

Define:

    C_remove(i) = E(K_-i) - E(K)

Interpretation:

    C_remove(i) > 0
        Removing Fi makes the result worse.
        Fi is beneficial.

    C_remove(i) ≈ 0
        Removing Fi changes little.
        Fi is redundant.

    C_remove(i) < 0
        Removing Fi improves the result.
        Fi may be harmful.

---

# 7. Contribution of Rejected Frames

For a rejected frame:

    Fj ∉ K

insert it into the selected frame set at its original temporal position:

    K_+j = K ∪ {Fj}

Run reconstruction again.

Measure:

    E(K_+j)

Define:

    C_add(j) = E(K) - E(K_+j)

Interpretation:

    C_add(j) > 0
        The rejected frame improves reconstruction.
        The selector missed a useful frame.

    C_add(j) ≈ 0
        The rejected frame is redundant.

    C_add(j) < 0
        Adding the frame makes reconstruction worse.
        Rejecting it was appropriate.

---

# 8. Primary Keyframe-Selection Evaluation

Each frame will be classified using two pieces of information:

    SLAM decision
        +
    measured contribution

This creates the following categories:

| SLAM Decision | Contribution | Interpretation |
|---|---|---|
| Selected | Strong positive | Useful selection |
| Selected | Near zero | Redundant selection |
| Selected | Negative | Harmful selection |
| Rejected | Strong positive | Missed useful frame |
| Rejected | Near zero | Correct rejection |
| Rejected | Negative | Correct rejection of harmful frame |

The purpose is not initially to replace the keyframe selector.

The purpose is to determine whether the current selector's decisions correspond to actual reconstruction contribution.

---

# 9. Geometric Factor Extraction

For every ETH3D frame, calculate or obtain geometric descriptors such as:

## Camera Motion

- relative translation
- relative rotation
- camera baseline

Ground-truth poses should be used where possible so that these measurements are not contaminated by SLAM estimation error.

## Geometric Information

- viewpoint change
- parallax
- overlap with neighboring frames
- feature / correspondence count
- visibility of new scene regions
- depth variation

## Image Characteristics

- texture density
- image entropy
- sharpness / blur score

These values are stored together with measured frame contribution.

---

# 10. Why Ground-Truth Pose Should Be Used for Feature Analysis

Some candidate frame properties can be estimated from the images themselves.

However, image-derived measurements may introduce confounding.

For example:

    large camera motion
        ↓
    optical flow increases
        +
    motion blur may increase

Using ETH3D ground-truth camera poses allows camera translation, rotation, and viewpoint change to be measured independently of the performance of the SLAM system.

Therefore:

    GT pose
        ↓
    physical camera motion

should be preferred over:

    estimated pose
        ↓
    possibly model-dependent motion estimate

for the geometric-factor analysis.

---

# 11. Contribution vs Geometric Information

After measuring frame contribution, analyze relationships such as:

    Contribution vs Translation

    Contribution vs Rotation

    Contribution vs Parallax

    Contribution vs Overlap

    Contribution vs Depth Variation

The objective is to determine which geometric properties are associated with frames that actually improve reconstruction.

Initial analysis may use:

- Pearson correlation
- Spearman rank correlation
- partial correlation
- multiple regression

The analysis should account for correlations between explanatory variables.

---

# 12. Scene Diversity

Experiments should include multiple ETH3D sequences rather than relying on a single trajectory.

The selected sequences should ideally contain different conditions such as:

- strong camera translation,
- strong camera rotation,
- small-motion sequences,
- texture-rich scenes,
- texture-poor scenes,
- repeated structures,
- large depth variation,
- narrow spaces,
- open spaces.

This allows investigation of whether frame contribution is scene-dependent.

---

# 13. Controlled Degradation Extension

The clean ETH3D frame serves as a reference:

    Fi(clean)

Controlled degraded variants can then be generated:

    Fi(blur_1)
    Fi(blur_2)
    Fi(blur_3)

while preserving the same nominal geometric content.

Contribution can then be measured as:

    C_i(clean)

    C_i(blur_1)

    C_i(blur_2)

    C_i(blur_3)

The change:

    ΔC_i(blur)
        =
    C_i(blur) - C_i(clean)

measures how much frame contribution is lost as degradation increases.

---

# 14. Rolling-Shutter Extension

ETH3D synthetic rolling-shutter data will be investigated as a first controlled RS benchmark.

The desired comparison is:

    same scene
    same nominal trajectory
    same camera viewpoints
            ↓
    different rolling-shutter severity

If the existing ETH3D synthetic data provide sufficient control, they can be used directly.

Otherwise, ETH3D geometry and trajectories may be used as the basis for a custom rolling-shutter rendering experiment.

The key requirement is that geometric information remains controlled while the rolling-shutter image-formation process changes.

---

# 15. Model Comparison

VGGT-SLAM and MASt3R-SLAM must receive identical ETH3D RGB sequences.

Compare:

## Keyframe Density

    selected frames / total frames

## Selection Agreement

How often do both systems select the same regions of the sequence?

## Contribution Quality

Compare:

- average selected-frame contribution,
- redundant selected-frame fraction,
- harmful selected-frame fraction,
- missed-useful-frame fraction.

## Contribution Ranking

Determine whether frames selected by one system have systematically larger contribution than those selected by the other.

---

# 16. Expected Benchmark Outputs

For each sequence, produce:

## Keyframe Timeline

    Frame number →

    All frames:
    ─────────────────────────────────

    VGGT-SLAM:
    ●────●───●──────●────●──────────

    MASt3R-SLAM:
    ●──●──────●───●───────●────────

## Contribution Timeline

Plot measured contribution for each frame along the sequence.

## Camera Trajectory

Visualize the GT trajectory and mark:

- useful selected frames,
- redundant selected frames,
- harmful selected frames,
- useful rejected frames.

## Contribution Distribution

Compare contribution distributions for:

    selected
    vs
    rejected

frames.

## Model Comparison

Compare VGGT-SLAM and MASt3R-SLAM on identical sequences.

---

# 17. Initial Experimental Scope

The first milestone should be intentionally small.

Recommended initial experiment:

    1 ETH3D sequence
        ↓
    VGGT-SLAM
        ↓
    Record selected keyframes
        ↓
    Compute baseline ATE
        ↓
    Leave-one-out on selected keyframes
        ↓
    Compute frame contribution
        ↓
    Plot contribution along the trajectory

Once this pipeline works correctly, extend to:

    rejected-frame add-one-in
        ↓
    MASt3R-SLAM
        ↓
    multiple ETH3D sequences
        ↓
    geometric factor analysis
        ↓
    degradation experiments

---

# 18. Primary Benchmark Questions

The ETH3D experiments should ultimately answer:

1. Which frames do VGGT-SLAM and MASt3R-SLAM select?

2. Are the selected frames actually important for trajectory and reconstruction accuracy?

3. How many selected keyframes are effectively redundant?

4. Are there selected frames whose removal improves the result?

5. Do the systems reject frames that would have improved reconstruction?

6. Which geometric properties characterize frames with high contribution?

7. Do VGGT-SLAM and MASt3R-SLAM value the same frames?

8. How does frame contribution change when controlled camera degradation is introduced?

The central purpose of ETH3D is to provide the ground truth needed to distinguish:

    "The SLAM system selected this frame."

from:

    "This frame actually improved the reconstruction."