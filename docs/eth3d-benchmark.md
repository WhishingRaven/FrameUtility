# ETH3D Benchmark Plan

## 1. Purpose

ETH3D is used as a source of calibrated scenes, camera trajectories, and ground truth for measuring the **marginal utility of controlled degraded observations**.

It is no longer used primarily to audit which keyframes VGGT-SLAM or MASt3R-SLAM select.

The benchmark question is:

> For a fixed context of ETH3D views, how does progressively degrading one additional valid view change reconstruction, relative to dropping that view entirely?

---

## 2. Why ETH3D

ETH3D provides useful properties for this experiment:

- calibrated camera imagery;
- accurate ground-truth camera trajectories for SLAM sequences;
- scenes with substantial geometric diversity;
- global-shutter data suitable as a clean source;
- ground-truth geometry/depth in parts of the benchmark ecosystem.

Before finalizing a sequence, verify exactly which GT products are available for that subset and do not assume every ETH3D sequence contains all modalities.

---

## 3. Role in the Experiment

The intended workflow is:

```text
ETH3D clean observations
        ↓
choose context K_-i + target Fi
        ↓
construct drop baseline K_-i
        ↓
generate Fi(b) at controlled blur levels
        ↓
run fixed-set reconstruction
        ↓
measure U_i(b | K_-i)
```

Ground truth is used only for evaluation and for deriving context descriptors such as relative pose.

---

## 4. Input Policy

Both reconstruction backends must receive the **same explicitly specified image sets**.

Do not let each SLAM system independently choose its own keyframes in the primary experiment, because that changes the context and makes marginal utility incomparable.

If a backend has an internal image-rejection mechanism that cannot be disabled, record this behavior explicitly and treat it as part of the backend response.

---

## 5. Scene Selection

Choose several sequences with varied conditions:

- translation-dominant motion;
- rotation-dominant motion;
- corners / viewpoint changes;
- texture-rich surfaces;
- texture-poor surfaces;
- repeated structures;
- large and small depth variation.

Prefer scenes where enough neighboring frames are available to build multiple valid contexts around each target.

---

## 6. Target and Context Construction

For each target frame `Fi`, construct one or more contexts `K_-i`.

Recommended context families:

### Local temporal context

A fixed number of neighboring views around `Fi` excluding the target.

### Sparse context

A lower-density subset with wider baselines.

### Redundant context

Several views close to the target pose.

### Low-redundancy context

Views that leave the target with more unique geometric coverage.

This permits direct testing of context dependence.

---

## 7. Baselines

For every context:

### Drop baseline

```text
E_drop = E(K_-i)
```

### Clean-add baseline

```text
E_clean = E(K_-i ∪ {Fi(clean)})
U_clean = E_drop - E_clean
```

Only after these two runs are validated should degraded variants be introduced.

---

## 8. Motion-Blur Generation

### Preferred

Use GT or otherwise known camera motion to synthesize exposure integration.

For translation, depth-aware rendering/warping is required for physically consistent blur because pixel motion is depth-dependent.

### Minimal viable first experiment

Select target frames where the synthetic exposure perturbation is restricted to rotation and generate blur from multiple homographic warps.

This preserves a clean causal intervention:

```text
same scene
same nominal viewpoint
same context
only exposure-time camera motion changes
```

---

## 9. Evaluation Metrics

### Pose

- ATE;
- RPE;
- translation / rotation error;
- pairwise relative-pose accuracy if full trajectory output is not meaningful for a small view set.

### Depth / geometry

Where GT is available:

- depth RMSE / AbsRel;
- point-cloud precision, recall, F1;
- Chamfer distance;
- completeness;
- point-to-surface error.

For feed-forward reconstruction on sparse sets, pairwise or per-view pose/geometry metrics may be more natural than a conventional long-sequence SLAM ATE. Choose metrics to match the backend output.

---

## 10. Utility Dataset

Recommended schema:

| scene | target | context_id | n_context | backend | blur_level | E_drop | E_add | utility | overlap | baseline | texture |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|

Keep each evaluation metric in a separate column or long-form `metric/value` pair rather than collapsing them too early.

---

## 11. Initial Milestone

Start small:

```text
1 ETH3D scene
    ↓
1 backend
    ↓
3–5 target frames
    ↓
2 contexts per target
    ↓
clean + 5 blur levels
    ↓
pose/geometry evaluation
    ↓
plot U(b)
```

The milestone succeeds if the entire add-vs-drop pipeline is reproducible and at least some clean target views show positive utility.

Only then expand to:

- more targets;
- more scenes;
- second backend;
- context-factor analysis;
- additional degradation types.

---

## 12. Relation to RealX3D

RealX3D is a more direct degradation benchmark than ETH3D for realistic adverse imaging conditions and should be read carefully before finalizing the dataset strategy.

ETH3D remains useful because it provides a familiar calibrated benchmark and can support synthetic controlled interventions. The project should avoid claiming that ETH3D itself is a new degradation benchmark.

A possible final evaluation strategy is:

- **ETH3D** for controlled synthetic intervention and context sampling;
- **RealX3D** for external validation under realistic degradations, if its release and format support the required add-vs-drop construction.

---

## 13. Benchmark Questions

ETH3D experiments should answer:

1. Is the clean target observation beneficial relative to dropping it?
2. How does its marginal utility change with blur severity?
3. Does utility cross zero?
4. How does the crossing depend on context redundancy and relative pose?
5. Are the results consistent across scenes and backends?

ETH3D is therefore a controlled test bed for:

> **When does adding a degraded but geometrically valid view become worse than not adding it?**
