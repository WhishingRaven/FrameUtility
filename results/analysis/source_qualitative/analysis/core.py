"""Fixed descriptive estimands shared by every scene and execution platform."""
from collections import defaultdict
import math

import numpy as np

from metrics import METRICS

VERSION = "1.0.0"
PRIMARY = "geometry_depth_absrel"
LABELS = {
    PRIMARY: ("Depth AbsRel", "% depth error", "percentage points", 100),
    "geometry_depth_rmse_m": ("Depth RMSE", "m", "m", 1),
    "pose_ate_normalized": ("Normalized pose ATE", "scene radius", "scene radius", 1),
    "pose_pair_translation_relative": ("Relative translation error", "relative error", "relative error", 1),
    "pose_pair_rotation_deg": ("Relative rotation error", "degrees", "degrees", 1),
}
CATEGORIES = ("helpful", "harmful", "within_threshold", "unavailable")


def classify(value, epsilon):
    if value is None:
        return "unavailable"
    if value > epsilon:
        return "helpful"
    if value < -epsilon:
        return "harmful"
    return "within_threshold"


def distribution(values):
    data = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(data):
        return {k: None for k in ("mean", "median", "q25", "q75", "min", "max")}
    if not np.isfinite(data).all():
        raise ValueError("Non-finite value reached descriptive aggregation")
    return dict(zip(("mean", "median", "q25", "q75", "min", "max"), map(float,
                    (data.mean(), np.median(data), *np.quantile(data, [.25, .75]), data.min(), data.max()))))


def normalize_rows(rows, scene, backend, manifest, protocol):
    """Validate identity and recompute U from errors, retaining every planned cell."""
    expected = {(m, t, b, metric) for m in protocol["sizes"] for t in manifest["sets"][str(m)]
                for b in protocol["blur_degrees"] for metric in METRICS}
    by_key = {}
    for raw in rows:
        key = (raw["size"], raw["target"], raw["blur_deg"], raw["metric"])
        if key not in expected or key in by_key:
            raise ValueError(f"Unexpected/duplicate score identity: {scene}/{backend}/{key}")
        if raw.get("scene") != scene or raw.get("backend") != backend or raw.get("n_context") != key[0] - 1:
            raise ValueError(f"Score context identity differs: {scene}/{backend}/{key}")
        row = dict(raw)
        if row["status"] == "ok":
            values = [row.get(k) for k in ("E_drop", "E_add", "utility_raw")]
            if any(v is None or not math.isfinite(v) for v in values):
                raise ValueError(f"Invalid successful score: {key}")
            if row["E_drop"] < 0 or row["E_add"] < 0:
                raise ValueError(f"Negative error magnitude: {key}")
            utility = row["E_drop"] - row["E_add"]
            if not math.isclose(utility, row["utility_raw"], abs_tol=1e-12, rel_tol=1e-9):
                raise ValueError(f"Stored U does not equal DROP minus ADD: {key}")
            row["utility_raw"] = utility
        else:
            row["utility_raw"] = None
        by_key[key] = row
    output = []
    for size in protocol["sizes"]:
        for target in manifest["sets"][str(size)]:
            for degree in protocol["blur_degrees"]:
                for metric in METRICS:
                    row = by_key.get((size, target, degree, metric), {
                        "scene": scene, "backend": backend, "size": size, "target": target,
                        "blur_deg": degree, "metric": metric, "n_context": size - 1,
                        "status": "missing", "E_drop": None, "E_add": None,
                        "utility_raw": None, "evaluation_pixels": None, "error": "Score row not recorded",
                    }).copy()
                    eps = protocol["evaluation"]["primary_absrel_effect_floor"] if metric == PRIMARY else 0
                    row["classification"] = classify(row["utility_raw"], eps)
                    row["raw_sign"] = classify(row["utility_raw"], 0)
                    row["classification_epsilon"] = eps
                    output.append(row)
    return output


def transition_details(levels, values, epsilon):
    """Observe brackets between signed samples; never infer an interpolated root."""
    states = [classify(v, epsilon) for v in values]
    events, previous = [], None
    for index, state in enumerate(states):
        if state not in ("helpful", "harmful"):
            continue
        if previous is not None and states[previous] != state:
            interior = states[previous + 1:index]
            events.append({
                "direction": "positive_to_negative" if state == "harmful" else "negative_to_positive",
                "lower_deg": levels[previous], "upper_deg": levels[index],
                "intermediate_within_threshold": [levels[i] for i in range(previous + 1, index)
                                                  if states[i] == "within_threshold"],
                "intermediate_unavailable": [levels[i] for i in range(previous + 1, index)
                                              if states[i] == "unavailable"],
                "interior_path_unresolved": bool(interior),
            })
        previous = index
    first = None
    if states[0] == "helpful":
        first = next((e for e in events if e["direction"] == "positive_to_negative"), None)
    return {
        "epsilon": epsilon, "clean_class": states[0], "end_class": states[-1],
        "first_positive_to_negative_bracket": first,
        "observed_clean_positive_to_negative": first is not None,
        "returns_positive_after_first_negative": first is not None and any(
            e["direction"] == "negative_to_positive" and e["lower_deg"] >= first["upper_deg"] for e in events),
        "events": events, "any_harm_observed": "harmful" in states,
        "unavailable_levels": [b for b, state in zip(levels, states) if state == "unavailable"],
    }


def curve_summaries(rows, protocol):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scene"], row["backend"], row["size"], row["target"], row["metric"])].append(row)
    output = []
    for key, group in sorted(groups.items()):
        by_degree = {r["blur_deg"]: r for r in group}
        levels = protocol["blur_degrees"]
        values = [by_degree[b]["utility_raw"] for b in levels]
        eps = protocol["evaluation"]["primary_absrel_effect_floor"] if key[-1] == PRIMARY else 0
        curve = dict(zip(("scene", "backend", "size", "target", "metric"), key))
        curve.update(complete=all(v is not None for v in values), valid_levels=sum(v is not None for v in values),
                     planned_levels=len(levels), clean_utility_raw=values[0], end_utility_raw=values[-1],
                     raw=transition_details(levels, values, 0), threshold=transition_details(levels, values, eps))
        curve["sensitivity"] = ([transition_details(levels, values, e) for e in
                                  protocol["evaluation"]["sensitivity_absrel_effect_floors"]]
                                 if key[-1] == PRIMARY else [])
        curve["representative"] = key[3] in representative_ids([r["target"] for r in rows
            if r["scene"] == key[0] and r["size"] == key[2]])
        output.append(curve)
    return output


def representative_ids(ids):
    values = sorted(set(ids))
    return list(dict.fromkeys(values[i] for i in (0, len(values) // 2, len(values) - 1))) if values else []


def stage_summaries(rows, protocol, *, sensitivity=False):
    grouped = defaultdict(list)
    clean = {(r["scene"], r["backend"], r["size"], r["target"], r["metric"]): r
             for r in rows if r["blur_deg"] == 0}
    for row in rows:
        if not sensitivity or row["metric"] == PRIMARY:
            grouped[(row["scene"], row["backend"], row["size"], row["metric"], row["blur_deg"])].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        epsilons = (protocol["evaluation"]["sensitivity_absrel_effect_floors"] if sensitivity else
                    [protocol["evaluation"]["primary_absrel_effect_floor"] if key[3] == PRIMARY else 0])
        for eps in epsilons:
            result = dict(zip(("scene", "backend", "size", "metric", "blur_deg"), key))
            counts = {state: sum(classify(r["utility_raw"], eps) == state for r in group) for state in CATEGORIES}
            valid = sum(r["status"] == "ok" for r in group)
            result.update(epsilon=eps, planned=key[2], valid=valid,
                          missing=sum(r["status"] == "missing" for r in group),
                          invalid=sum(r["status"] not in ("missing", "ok") for r in group), **counts)
            for state in CATEGORIES:
                result[state + "_fraction_planned"] = counts[state] / key[2]
            result["harmful_fraction_valid"] = counts["harmful"] / valid if valid else None
            for field in ("utility_raw", "E_drop", "E_add"):
                result[field] = distribution(r[field] for r in group if r["status"] == "ok")
            result["E_clean_add"] = distribution(clean[(r["scene"], r["backend"], r["size"], r["target"], r["metric"])]["E_add"]
                                                  for r in group)
            output.append(result)
    return output


def transition_summaries(curves, *, sensitivity=False):
    grouped = defaultdict(list)
    for c in curves:
        policies = c["sensitivity"] if sensitivity else [c["raw"], c["threshold"]]
        for i, policy in enumerate(policies):
            if sensitivity and c["metric"] != PRIMARY:
                continue
            label = "sensitivity" if sensitivity else ("raw" if i == 0 else "threshold")
            grouped[(c["scene"], c["backend"], c["size"], c["metric"], label, policy["epsilon"])].append((c, policy))
    output = []
    for key, group in sorted(grouped.items()):
        record = dict(zip(("scene", "backend", "size", "metric", "policy", "epsilon"), key))
        complete = [(c, p) for c, p in group if c["complete"]]
        count = sum(p["observed_clean_positive_to_negative"] for c, p in complete)
        record.update(planned_curves=key[2], complete_curves=len(complete),
                      incomplete_curves=len(group) - len(complete),
                      clean_helpful=sum(p["clean_class"] == "helpful" for c, p in group),
                      clean_harmful=sum(p["clean_class"] == "harmful" for c, p in group),
                      clean_within_threshold=sum(p["clean_class"] == "within_threshold" for c, p in group),
                      clean_unavailable=sum(p["clean_class"] == "unavailable" for c, p in group),
                      transitions_complete=count,
                      transition_fraction_planned=count / key[2],
                      observed_transitions_incomplete=sum(p["observed_clean_positive_to_negative"] for c, p in group if not c["complete"]),
                      complete_no_harm_observed=sum(not p["any_harm_observed"] for c, p in complete),
                      returns_positive=sum(p["returns_positive_after_first_negative"] for c, p in complete))
        eligible = sum(p["clean_class"] == "helpful" for c, p in complete)
        record["eligible_complete_clean_helpful"] = eligible
        record["transition_fraction_eligible_complete"] = count / eligible if eligible else None
        output.append(record)
    return output


def cohort_summaries(stage_rows, scene_info, requested_scenes):
    """Equal scene weights first; target pooling is an explicitly separate field."""
    groups = defaultdict(list)
    for row in stage_rows:
        info = scene_info[row["scene"]]
        for scope, value in (("all", "all"), ("device", info["device"]), ("runtime_group", info["runtime_group"])):
            groups[(row["backend"], row["size"], row["metric"], row["blur_deg"], row["epsilon"], scope, value)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        result = dict(zip(("backend", "size", "metric", "blur_deg", "epsilon", "stratum", "stratum_value"), key))
        used = sorted(r["scene"] for r in group)
        result.update(scenes_used=used, n_scenes_used=len(used),
                      n_scenes_requested=len(requested_scenes), requested_scenes=list(requested_scenes))
        for category in CATEGORIES:
            result[category + "_scene_fraction"] = distribution(r[category + "_fraction_planned"] for r in group)
        for field in ("utility_raw", "E_drop", "E_add"):
            result[field + "_scene_mean"] = distribution(r[field]["mean"] for r in group)
            result[field + "_scene_median"] = distribution(r[field]["median"] for r in group)
        n = sum(r["planned"] for r in group)
        result["pooled_targets_secondary"] = {"planned": n, "valid": sum(r["valid"] for r in group),
            "harmful_fraction_planned": sum(r["harmful"] for r in group) / n}
        output.append(result)
    return output


def paired_comparisons(rows, protocol):
    if len(protocol["backends"]) != 2:
        raise ValueError("Matched comparison currently requires exactly two approved backends")
    a, b = protocol["backends"]
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["scene"], row["size"], row["metric"], row["blur_deg"])].setdefault(row["target"], {})[row["backend"]] = row
    output = []
    for key, group in sorted(groups.items()):
        matched = [x for x in group.values() if all(x[backend]["status"] == "ok" for backend in (a, b))]
        record = dict(zip(("scene", "size", "metric", "blur_deg"), key))
        record.update(planned=key[1], matched=len(matched), unmatched=key[1] - len(matched),
                      comparison=f"{a}_minus_{b}")
        for field in ("E_drop", "E_add", "utility_raw"):
            record[field + "_difference"] = distribution(x[a][field] - x[b][field] for x in matched)
            record[field + "_matched_by_model"] = {model: distribution(x[model][field] for x in matched) for model in (a, b)}
        output.append(record)
    return output


def cohort_transition_summaries(transitions, scene_info, requested_scenes):
    groups = defaultdict(list)
    for row in transitions:
        info = scene_info[row["scene"]]
        for scope, value in (("all", "all"), ("device", info["device"]), ("runtime_group", info["runtime_group"])):
            groups[(row["backend"], row["size"], row["metric"], row["policy"], row["epsilon"], scope, value)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        record = dict(zip(("backend", "size", "metric", "policy", "epsilon", "stratum", "stratum_value"), key))
        record.update(n_scenes_used=len(group), n_scenes_requested=len(requested_scenes),
                      scenes_used=sorted(r["scene"] for r in group),
                      transition_scene_fraction=distribution(r["transition_fraction_planned"] for r in group),
                      complete_curve_scene_fraction=distribution(r["complete_curves"] / r["planned_curves"] for r in group),
                      incomplete_curves=sum(r["incomplete_curves"] for r in group),
                      observed_transitions_incomplete=sum(r["observed_transitions_incomplete"] for r in group))
        output.append(record)
    return output
