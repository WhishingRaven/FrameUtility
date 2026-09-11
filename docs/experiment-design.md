# Experimental Design

## 1. Objective

FrameUtility studies two related questions:

1. Do modern neural SLAM systems select frames that actually contribute to reconstruction?
2. What determines the actual contribution of a frame?

The initial systems are:

- VGGT-SLAM
- MASt3R-SLAM

The study deliberately separates:

    keyframe selection decision

from:

    measured reconstruction contribution

A frame is not considered important simply because a SLAM system selected it.

Its contribution must be measured experimentally.

---

# 2. Keyframe Selection as the Object of Study

For an input video:

    V = {F1, F2, ..., FN}

a SLAM system produces a selected keyframe set:

    K ⊂ V

and rejected set:

    R = V \ K

We record this decision without modifying the original selector.

The first question is:

> Does membership in K correspond to high actual contribution?

---

# 3. Contribution of Selected Frames

Let:

    E(K)

be the reconstruction error obtained using the selected keyframe set.

For selected frame `Fi`:

    Fi ∈ K

remove it:

    K_-i = K \ {Fi}

and measure:

    E(K_-i)

Define leave-one-out contribution:

    C_remove(i) = E(K_-i) - E(K)

Interpretation:

    C_remove(i) > 0
        useful selected frame

    C_remove(i) ≈ 0
        redundant selected frame

    C_remove(i) < 0
        potentially harmful selected frame

---

# 4. Contribution of Rejected Frames

A keyframe selector can also fail by rejecting a useful frame.

For:

    Fj ∈ R

construct:

    K_+j = K ∪ {Fj}

and measure:

    E(K_+j)

Define add-one-in contribution:

    C_add(j) = E(K) - E(K_+j)

Interpretation:

    C_add(j) > 0
        rejected frame would improve reconstruction

    C_add(j) ≈ 0
        rejected frame is redundant

    C_add(j) < 0
        rejected frame would harm reconstruction

---

# 5. Selection–Contribution Matrix

Each frame can therefore be categorized according to:

1. selector decision,
2. measured contribution.

| Selector | Contribution | Category |
|---|---|---|
| Selected | Strong positive | Useful selection |
| Selected | Near zero | Redundant selection |
| Selected | Negative | Harmful selection |
| Rejected | Strong positive | Missed useful frame |
| Rejected | Near zero | Correct rejection |
| Rejected | Negative | Correct rejection of harmful frame |

This matrix is one of the primary outputs of the study.

---

# 6. Context Dependence

Frame contribution is conditional on the other available frames.

Formally:

    C_i = C_i(K)

Two nearly identical neighboring frames may each have low leave-one-out contribution because the other frame provides the same information.

Therefore:

    low leave-one-out contribution
        ≠
    intrinsically useless frame

This limitation must be explicitly acknowledged.

---

# 7. Extended Contribution Measurement

If computational resources permit, leave-one-out analysis will be extended using random frame subsets.

For many subsets:

    S ⊂ V

compare reconstruction with and without frame `i`:

    Δ_i(S) = Q(S ∪ {Fi}) - Q(S)

where `Q` represents reconstruction quality.

Average over many contexts:

    C_i ≈ E_S[Δ_i(S)]

This estimates the expected marginal contribution of the frame.

A more rigorous extension is an approximate Shapley value:

    φ_i = E_S[Q(S ∪ {Fi}) - Q(S)]

with appropriate weighting over subsets.

This provides a more context-independent notion of frame contribution, but has significantly greater computational cost.

The initial project therefore begins with leave-one-out and add-one-in analysis.

---

# 8. Experiment A — Keyframe Selection Audit

## Question

What frames do existing neural SLAM systems select?

## Systems

- VGGT-SLAM
- MASt3R-SLAM

## Procedure

Run both systems on identical input sequences.

Record for every frame:

- frame ID,
- timestamp,
- selected / rejected,
- selector-specific score if accessible,
- estimated pose,
- tracking status.

## Output

For each system:

    sequence
        ↓
    selected/rejected decision
        ↓
    keyframe set K

No contribution analysis is performed yet.

This experiment only records the systems' decisions.

---

# 9. Experiment B — Contribution Audit

## Question

Are selected frames actually useful?

For each selected frame:

    remove frame
        ↓
    rerun reconstruction
        ↓
    measure change in error

Then evaluate rejected frames:

    add rejected frame
        ↓
    rerun reconstruction
        ↓
    measure change in error

## Main Outputs

- distribution of selected-frame contribution,
- fraction of redundant selected frames,
- fraction of harmful selected frames,
- fraction of rejected frames with positive contribution,
- contribution ranking versus selector decision.

---

# 10. Experiment C — Geometric Factor Analysis

## Question

What geometric properties explain measured contribution?

Candidate variables:

### Camera Motion

- relative translation,
- relative rotation,
- baseline.

### Geometric Novelty

- parallax,
- viewpoint change,
- feature/correspondence overlap,
- visibility of previously unseen geometry.

### Scene Properties

- texture density,
- feature density,
- depth variation.

## Dependent Variable

Measured frame contribution.

## Analysis

Candidate analyses:

- Pearson correlation,
- Spearman correlation,
- partial correlation,
- multiple regression,
- nonlinear regression,
- feature-importance analysis.

The objective is explanatory rather than predictive.

---

# 11. Confounding Problem

Real video does not provide clean control over geometric information and image degradation.

For example:

    fast camera rotation
        ↓
    viewpoint change ↑
    optical flow ↑
    overlap may decrease
    motion blur ↑
    rolling shutter ↑

Therefore, observing that blurry frames have low contribution does not prove that blur caused the low contribution.

Controlled experiments are required.

---

# 12. Experiment D — Controlled Motion Blur

Start from the same clean frame:

    Fi

Generate:

    Fi(B0)
    Fi(B1)
    Fi(B2)
    Fi(B3)

where only blur severity changes.

The nominal:

- scene,
- viewpoint,
- geometry,
- texture,

remain fixed.

Measure:

    C_i(B)

and contribution loss:

    ΔC_i(B) = C_i(B) - C_i(B0)

Main question:

> How much geometric contribution is destroyed by motion blur?

---

# 13. Experiment E — Controlled Rolling Shutter

Rolling shutter is a temporal image-formation effect.

Different image rows correspond to different camera poses.

Therefore, arbitrary 2D image warping should not be treated as a physically accurate rolling-shutter simulation.

Prefer a controlled 3D environment with known:

- scene geometry,
- camera trajectory,
- timestamps,
- camera intrinsics.

Render:

    RS0 = global shutter
    RS1 = low readout time
    RS2 = medium readout time
    RS3 = high readout time

while keeping the nominal trajectory identical.

Measure:

    C_i(RS)

and:

    ΔC_i(RS) = C_i(RS) - C_i(RS0)

---

# 14. Experiment F — Blur × Rolling-Shutter Interaction

Use a factorial experiment:

| RS / Blur | B0 | B1 | B2 | B3 |
|---|---:|---:|---:|---:|
| RS0 | ✓ | ✓ | ✓ | ✓ |
| RS1 | ✓ | ✓ | ✓ | ✓ |
| RS2 | ✓ | ✓ | ✓ | ✓ |
| RS3 | ✓ | ✓ | ✓ | ✓ |

Measure:

    C_i(B, RS)

The null/additive expectation is approximately:

    D(B, RS)
    ≈ D(B, 0) + D(0, RS)

where `D` represents degradation in reconstruction quality or contribution.

A significant interaction occurs when the combined effect cannot be explained by the individual effects alone.

---

# 15. Main Hypotheses

## H1 — Unequal Contribution

Selected keyframes will exhibit substantially different measured contributions.

## H2 — Selection Imperfection

Existing keyframe selectors will select some frames with little or negative measured contribution.

## H3 — Missed Frames

Some rejected frames will exhibit positive add-one-in contribution.

## H4 — Geometric Information

Non-redundant geometric information such as viewpoint novelty and parallax will be positively associated with contribution.

## H5 — Motion Blur

Increasing motion blur will reduce contribution while geometric content is controlled.

## H6 — Rolling Shutter

Increasing rolling-shutter distortion will reduce contribution while geometric content is controlled.

## H7 — Useful-to-Harmful Transition

A geometrically useful frame may become redundant or harmful under sufficiently severe degradation.

## H8 — Interaction

Motion blur and rolling shutter may exhibit a nonlinear interaction.

## H9 — Model Dependence

VGGT-SLAM and MASt3R-SLAM will exhibit different selection and contribution patterns.

---

# 16. Scientific Goal

The study does not initially attempt to design a better keyframe-selection algorithm.

Instead, it audits the relationship:

    Keyframe Heuristic
           ↓
    Selection Decision
           ↓
    Actual Contribution
           ↓
    Reconstruction Quality

and asks:

> Does the heuristic identify the frames that actually matter?

After measuring contribution, the study investigates why a frame is useful, redundant, or harmful.

This provides the basis for understanding — rather than immediately replacing — modern neural SLAM keyframe selection.