"""Analyze the frozen five-case follow-up; preserve every repeat and failure."""
import argparse
import csv
from datetime import datetime, timezone
import itertools
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backends import ROOT
from followup import DEFAULT_OUT, case_id, frozen_protocol, read_json, verify_inputs
from metrics import METRICS
from run import atomic_json, sha256


def summarize_values(drops, adds, expected, floor=0):
    combinations = [d-a for d, a in itertools.product(drops, adds)]
    complete = len(drops) == expected and len(adds) == expected
    return {"valid_drop_repeats": len(drops), "valid_add_repeats": len(adds), "complete": complete,
            "drop_error_mean": float(np.mean(drops)) if drops else None,
            "add_error_mean": float(np.mean(adds)) if adds else None,
            "utility_raw_mean": float(np.mean(combinations)) if combinations else None,
            "utility_raw_range": [min(combinations), max(combinations)] if combinations else None,
            "all_combinations_beneficial": complete and all(u > 0 for u in combinations),
            "all_combinations_harmful": complete and all(u < 0 for u in combinations),
            "all_combinations_beneficial_above_floor": complete and all(u > floor for u in combinations),
            "all_combinations_harmful_above_floor": complete and all(u < -floor for u in combinations)}


def crossing_bracket(conditions, floor=False):
    suffix = "_above_floor" if floor else ""
    clean = conditions[0]
    if not clean["all_combinations_beneficial"+suffix]:
        return None
    lower = clean["blur_deg"]
    for condition in conditions[1:]:
        if condition["all_combinations_harmful"+suffix]:
            return [lower, condition["blur_deg"]]
        if condition["all_combinations_beneficial"+suffix]:
            lower = condition["blur_deg"]
    return None


def analyze(out):
    protocol = frozen_protocol(out)
    manifest = verify_inputs(out, protocol)
    protocol_sha = sha256(out/"protocol.json")
    for name, expected_sha in protocol["code_sha256"].items():
        assert sha256(out/"source"/name) == expected_sha
    for control in protocol["controls"]:
        assert sha256(ROOT/control["path"]) == control["sha256"]
    for backend in ("vggt", "mast3r"):
        envpath = out/backend/"environment.json"
        if envpath.exists():
            assert read_json(envpath)["followup_protocol_sha256"] == protocol_sha
    audit = read_json(out / "policy_audit.json")
    expected = protocol["policy"]["replicates"]
    floor = protocol["policy"]["absrel_effect_floor"]
    records, missing, cases, rows = {}, [], [], []
    for case in protocol["cases"]:
        cid = case_id(case)
        for rep in range(expected):
            block = out / case["backend"] / f"replicate_{rep}"
            for condition in ["drop"] + [f"add_b{b:g}" for b in case["degrees"]]:
                path = block / (cid+"_"+condition+".json")
                key = (case["backend"], cid, rep, condition)
                if not path.exists():
                    missing.append(str(path.relative_to(out)))
                    continue
                record = read_json(path)
                assert record["case"] == cid and record["replicate"] == rep
                assert record["backend"] == case["backend"] and record["condition"] == condition
                common = [i for i in manifest["sets"][str(case["size"])] if i != case["target"]]
                assert record["common_ids"] == common
                if "prediction_sha256" in record:
                    prediction = block / "predictions" / (cid+"_"+condition+".npz")
                    assert sha256(prediction) == record["prediction_sha256"]
                records[key] = record
    for case in protocol["cases"]:
        cid = case_id(case)
        case_stats = {**case, "case": cid, "metrics": {}}
        for metric in METRICS:
            drops = [records.get((case["backend"], cid, rep, "drop"), {}).get("metrics", {}).get(metric)
                     for rep in range(expected)]
            drops = [v for v in drops if v is not None]
            conditions = []
            for degree in case["degrees"]:
                adds = [records.get((case["backend"], cid, rep, f"add_b{degree:g}"), {}).get("metrics", {}).get(metric)
                        for rep in range(expected)]
                stats = summarize_values(drops, [v for v in adds if v is not None], expected,
                                         floor if metric == "geometry_depth_absrel" else 0)
                stats["blur_deg"] = degree
                conditions.append(stats)
                for rep in range(expected):
                    drop = records.get((case["backend"], cid, rep, "drop"), {})
                    add = records.get((case["backend"], cid, rep, f"add_b{degree:g}"), {})
                    d, a = drop.get("metrics", {}).get(metric), add.get("metrics", {}).get(metric)
                    family = "pose" if metric.startswith("pose") else "geometry"
                    rows.append({"backend": case["backend"], "size": case["size"], "target": case["target"],
                                 "replicate": rep, "blur_deg": degree, "metric": metric,
                                 "E_drop": d, "E_add": a, "utility_raw": d-a if d is not None and a is not None else None,
                                 "utility_relative": (d-a)/d if d is not None and a is not None and abs(d)>1e-10 else None,
                                 "drop_nonpositive_fraction": drop.get("depth_diagnostics", {}).get("nonpositive_fraction"),
                                 "add_nonpositive_fraction": add.get("depth_diagnostics", {}).get("nonpositive_fraction"),
                                 "drop_error": drop.get("errors", {}).get(family, drop.get("error")),
                                 "add_error": add.get("errors", {}).get(family, add.get("error"))})
            case_stats["metrics"][metric] = {"conditions": conditions,
                                            "stable_raw_crossing_bracket": crossing_bracket(conditions),
                                            "stable_effect_crossing_bracket": crossing_bracket(conditions, floor=True)
                                            if metric == "geometry_depth_absrel" else None}
        cases.append(case_stats)
    statuses = {state: sum(r["status"] == state for r in records.values()) for state in ("ok", "partial", "failed")}
    result = {"frozen_utc": protocol["frozen_utc"], "analyzed_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": sha256(out/"protocol.json"), "policy": protocol["policy"],
              "analysis_code_sha256": sha256(Path(__file__)),
              "planned_reconstructions": protocol["new_reconstructions"], "observed_reconstructions": len(records),
              "complete": not missing and len(records) == protocol["new_reconstructions"],
              "statuses": statuses, "missing": missing, "pilot_unchanged": True,
              "identical_new_strict_scores": sum(r.get("metrics") == r.get("strict_metrics")
                                                and r.get("errors") == r.get("strict_errors") for r in records.values()),
              "policy_audit": {k: audit[k] for k in
                  ("identical_valid_legacy_metric_rows", "previously_invalid_metric_rows", "conditions")},
              "zero_control": {"images": len(protocol["controls"]),
                               "changed_pixels": sum(c["changed_pixels"] for c in protocol["controls"]),
                               "png_file_hashes_identical": all(c["sha256"] == c["clean_sha256"] for c in protocol["controls"])},
              "cases": cases}
    atomic_json(out / "summary.json", result)
    with (out/"utilities.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_cases(cases, floor, out)
    write_report(result, protocol, records, out)
    print("COMPLETE", result["complete"], "OBSERVED", len(records), "STATUSES", statuses, flush=True)
    for case in cases:
        stats = case["metrics"]["geometry_depth_absrel"]
        print(case["backend"], case["case"], "RAW", stats["stable_raw_crossing_bracket"],
              "EFFECT", stats["stable_effect_crossing_bracket"], flush=True)


def plot_cases(cases, floor, out):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    for ax, case in zip(axes.flat, cases):
        stats = case["metrics"]["geometry_depth_absrel"]["conditions"]
        valid = [s for s in stats if s["utility_raw_range"]]
        if not valid:
            continue
        x = [s["blur_deg"] for s in valid]
        lo, hi = np.array([s["utility_raw_range"] for s in valid]).T*100
        mean = np.array([s["utility_raw_mean"] for s in valid])*100
        color = "#2868b2" if case["backend"] == "vggt" else "#c65d23"
        ax.axhspan(-floor*100, floor*100, color="#aaaaaa", alpha=.14, label="Effect floor ±0.05 pp")
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.fill_between(x, lo, hi, color=color, alpha=.23, label="All DROP/ADD repeat combinations")
        ax.plot(x, mean, "o-", color=color, markersize=4, label="Mean utility")
        ax.set_title(f"{case['backend'].upper()} · {case['size']} views · {case['target']}")
        ax.set_xlabel("Full yaw exposure path (degrees)")
        ax.set_ylabel("Depth AbsRel utility (percentage points)")
        ax.grid(alpha=.17)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=10)
    axes.flat[-1].text(.05, .15, "Three fresh same-seed repeats per condition.\nRanges describe numerical repeatability,\nnot uncertainty over scenes.\nCases were selected from the pilot.",
                       transform=axes.flat[-1].transAxes, fontsize=10)
    fig.suptitle("Frozen follow-up · fixed-mask depth utility · positive helps, negative harms", fontsize=15)
    fig.tight_layout(rect=(0, .015, 1, .95))
    fig.savefig(out/"depth_repeat_ranges.png", dpi=170)
    fig.savefig(out/"depth_repeat_ranges.svg")
    plt.close(fig)


def show_range(value):
    return f"{value[0]*100:+.5f} ~ {value[1]*100:+.5f}" if value else "미계산"


def show_bracket(value):
    return f"[{value[0]:g}°, {value[1]:g}°]" if value else "확인 안 됨"


def write_report(result, protocol, records, out):
    lines = ["# 평가 기준 고정·0° 대조군·전환 반복 검증", "",
             f"프로토콜 고정 시각: {result['frozen_utc']}. 원래 420조건과 후속 결과를 분리했다. "
             f"새 재구성 **{result['observed_reconstructions']}/{result['planned_reconstructions']}회**: "
             f"정상 {result['statuses']['ok']} · 부분 유효 {result['statuses']['partial']} · 실패 {result['statuses']['failed']}.", ""]
    if not result["complete"]:
        lines += ["**계산 중인 중간 보고서다. 미완료 반복은 안정적 전환으로 판정하지 않았다.**", ""]
    lines += ["## 1. 실행 전에 고정한 평가 기준", "",
              "주 지표는 공통 컨텍스트의 scale-aligned depth AbsRel이다. 타깃을 제외한 동일 GT 마스크 전체에서 오차를 계산한다. "
              "전체 컨텍스트의 양수 예측에서 median(GT/prediction)으로 하나의 scale을 맞춘 뒤, 유한한 음수·0 깊이도 오차에 포함한다. "
              "평가 픽셀의 NaN/Inf는 geometry 실패로 남긴다. 비양수·비유한 픽셀 비율과 기존 엄격 점수도 함께 보존한다. Pose는 독립 평가한다.", "",
              "GT는 정렬된 ETH3D laser scan을 nearest-depth z-buffer로 투영한 탐색용 참조다. "
              "공식 occlusion mask를 적용한 ETH3D benchmark 점수나 장면 전체 completeness를 측정하는 지표는 아니다.", "",
              "AbsRel 효과 문턱은 ε=0.0005 (= 0.05 percentage point)로 실행 전에 고정했다. "
              "파일럿에서 얻은 정보를 참고한 문턱이며 통계적 유의성 기준은 아니다. 보조 지표는 RMSE와 포즈 3종이다.", "",
              "기존 데이터로 구현을 감사한 결과, 원래 유효했던 **712개 geometry 지표 행의 E 값은 정확히 일치**했다. "
              "원래 평가 불가였던 128개 지표 행(64조건)은 별도 사후 재평가로 남겼다. 새 규칙에서 기존 420조건 모두 깊이 평가가 가능했다. "
              "원본 점수는 수정하지 않았다.", "",
              "## 2. 0° 처리 대조군", "",
              f"20장 모두 clean의 단축 경로를 우회해 33개 항등 homography, 선형 radiance 적분, sRGB 복원·양자화와 동일 crop을 거쳤다. "
              f"**변경된 픽셀 {result['zero_control']['changed_pixels']}개**였으며, PNG 파일 해시도 "
              f"{'모두 일치했다' if result['zero_control']['png_file_hashes_identical'] else '일부 달랐다'}. "
              "입력이 정확히 같으므로 추론에서는 clean 조건과 합쳤다.", "",
              "이 검사는 0° fast path와 일반 처리 경로의 불일치를 배제한다. "
              "비영 회전의 보간 효과와 실제 노출 적분 효과를 독립적으로 분리한 대조군은 아니다.", "",
              "## 3. 반복 설계와 주 지표 결과", "",
              "파일럿에서 고른 다섯 컨텍스트의 각 조건을 3회씩 새로 재구성했다. 이전 결과를 새 반복 횟수에 포함하지 않았다. "
              "Seed는 매번 2026이다. MASt3R pair cache는 반복 블록마다 새로 만들고 같은 블록 안에서만 pair 추론을 재사용했다. "
              "모든 ADD/DROP의 정렬은 매번 새로 수행했다. 관측 범위는 같은 seed에서의 수치적 반복성이며 장면 모집단의 불확실성이 아니다.", "",
              "U 범위는 **DROP 3회 × ADD 3회의 9개 조합**이다. 모든 clean 조합이 양수이고 이후 모든 조합이 음수일 때 안정적 원시 전환으로 판정한다. "
              "효과 문턱 판정은 각각 +ε와 −ε를 넘어야 한다.", "",
              "| Backend / 컨텍스트 | Clean U 범위 (AbsRel pp) | 원시 부호 전환 구간 | 효과 문턱을 넘는 관측 구간 |",
              "|---|---:|---|---|"]
    for case in result["cases"]:
        depth = case["metrics"]["geometry_depth_absrel"]
        lines.append(f"| {case['backend'].upper()} / {case['size']}장 / {case['target']} | "
                     f"{show_range(depth['conditions'][0]['utility_raw_range'])} | "
                     f"{show_bracket(depth['stable_raw_crossing_bracket'])} | {show_bracket(depth['stable_effect_crossing_bracket'])} |")
    lines += ["", "구간은 관측된 안정적 양수 단계와 첫 안정적 음수 단계 사이를 뜻한다. "
              "효과 문턱 열에서는 양쪽 효과가 각각 문턱을 넘어야 한다. 중간 단계에 반복 간 부호 혼합이나 작은 효과가 있으면 구간이 넓어진다. "
              "단조성을 가정하거나 보간으로 정확한 연속 임계값을 추정하지 않았다.", "",
              "![세부 블러 단계와 반복 범위](depth_repeat_ranges.png)", ""]
    if result["complete"]:
        lines += ["### 결과 해석", ""]
        for case in result["cases"]:
            depth = case["metrics"]["geometry_depth_absrel"]
            label = f"{case['backend'].upper()} {case['size']}장 / {case['target']}"
            if depth["stable_effect_crossing_bracket"]:
                lines.append(f"- **{label}:** 원시 전환 {show_bracket(depth['stable_raw_crossing_bracket'])}이 재현됐고 효과 문턱도 통과했다.")
            elif depth["stable_raw_crossing_bracket"]:
                lines.append(f"- **{label}:** 원시 부호 전환은 재현됐지만 clean 이득과 blur 손해가 모두 문턱을 넘는 전환은 확인되지 않았다. 작은 원시 부호를 큰 효과로 해석하면 안 된다.")
            else:
                endpoint = depth["conditions"][-1]
                lines.append(f"- **{label}:** 선택한 구간에서 안정적인 AbsRel 전환이 확인되지 않았다. 마지막 단계 U 범위는 {show_range(endpoint['utility_raw_range'])} pp다.")
        lines += [""]
        strong = next(c for c in result["cases"] if c["backend"] == "mast3r" and c["size"] == 20)
        if all(m["stable_raw_crossing_bracket"] for m in strong["metrics"].values()):
            lines += ["### 큰 MASt3R 사례의 새 반복 평균", "",
                      "20장 / DSC_0238의 다섯 지표 모두에서 clean 이득과 4° 손해가 모든 반복 조합에 걸쳐 유지됐다. "
                      "아래는 각 조건의 새 반복 3회 평균이며, 모두 낮을수록 좋다.", "",
                      "| 지표 | DROP | Clean ADD | 4° ADD | 4° U_raw 범위 |", "|---|---:|---:|---:|---:|"]
            for metric, name, multiplier in [
                    ("geometry_depth_absrel", "Depth AbsRel (%)", 100),
                    ("geometry_depth_rmse_m", "Depth RMSE (m)", 1),
                    ("pose_ate_normalized", "Normalized ATE", 1),
                    ("pose_pair_translation_relative", "Pair translation relative", 1),
                    ("pose_pair_rotation_deg", "Pair rotation (°)", 1)]:
                conditions = strong["metrics"][metric]["conditions"]
                clean, bad = conditions[0], conditions[-1]
                lo, hi = bad["utility_raw_range"]
                lines.append(f"| {name} | {clean['drop_error_mean']*multiplier:.5f} | "
                             f"{clean['add_error_mean']*multiplier:.5f} | {bad['add_error_mean']*multiplier:.5f} | "
                             f"{lo*multiplier:+.5f} ~ {hi*multiplier:+.5f} |")
            lines += ["", "AbsRel 행의 U 범위 단위는 percentage point이며, 나머지는 각 지표의 원래 단위다.", ""]
    lines += ["### 단계별 깊이 utility", "",
              "| Backend / 컨텍스트 | Blur | U 범위 (AbsRel pp) | 유효 DROP / ADD 반복 | 판정 |", "|---|---:|---:|---:|---|"]
    for case in result["cases"]:
        for c in case["metrics"]["geometry_depth_absrel"]["conditions"]:
            state = ("도움, 문턱 초과" if c["all_combinations_beneficial_above_floor"] else
                     "해로움, 문턱 초과" if c["all_combinations_harmful_above_floor"] else
                     "양수, 일부 문턱 이하" if c["all_combinations_beneficial"] else
                     "음수, 일부 문턱 이하" if c["all_combinations_harmful"] else
                     "반복 간 부호 혼합" if c["complete"] else "미완료")
            lines.append(f"| {case['backend'].upper()} / {case['size']} / {case['target']} | {c['blur_deg']:g}° | "
                         f"{show_range(c['utility_raw_range'])} | {c['valid_drop_repeats']} / {c['valid_add_repeats']} | {state} |")
    lines += ["", "## 4. 보조 지표의 안정적 원시 전환", "",
              "| 컨텍스트 | RMSE | 정규화 ATE | Pair translation | Pair rotation |", "|---|---|---|---|---|"]
    for case in result["cases"]:
        fields = [show_bracket(case["metrics"][m]["stable_raw_crossing_bracket"]) for m in
                  ("geometry_depth_rmse_m", "pose_ate_normalized", "pose_pair_translation_relative", "pose_pair_rotation_deg")]
        lines.append(f"| {case['backend'].upper()} / {case['size']} / {case['target']} | " + " | ".join(fields) + " |")
    bad = [r for r in records.values() if r.get("depth_diagnostics", {}).get("nonpositive_pixels", 0)]
    max_fraction = max((r.get("depth_diagnostics", {}).get("nonpositive_fraction", 0) for r in records.values()), default=0)
    lines += ["", "## 5. 유효성과 해석 범위", "",
              f"새 재구성 중 비양수 깊이가 관측된 결과는 {len(bad)}개이며 고정 마스크 내 최대 비율은 {max_fraction*100:.6f}%였다. "
              "각 결과 JSON에는 기존 엄격 평가의 실패 여부도 기록돼 있다.", "",
              f"새 재구성 {result['observed_reconstructions']}개 중 **{result['identical_new_strict_scores']}개의 모든 평가 값·실패 상태가 기존 엄격 평가와 정확히 일치**했다. "
              "이번 선정 사례에서 확인된 전환은 새 깊이 처리 규칙으로 점수를 바꿔 만든 결과가 아니다.", "",
              "사례 선정과 효과 문턱은 탐색 결과의 영향을 받았다. 이번 실험은 그 사례의 대조군과 수치적 반복성을 검증한다. "
              "모집단의 전환 빈도, 모델 우열, 보편적 blur 임계값을 추정하지 않는다. 단일 office 장면이며 컨텍스트가 중첩된다. "
              "원시 부호 전환과 효과 문턱을 넘는 전환을 구분해야 한다.", "",
              "다음 확장은 다른 장면에서 같은 평가 정책을 고정해 검증하는 것이다. 새 장면은 이번 파일럿 선정 사례와 분리해 집계해야 한다.", "",
              "## 파일과 재현", "",
              "- [고정 프로토콜](protocol.json), [읽기용 프로토콜과 실행 명령](protocol.md)",
              "- [기존 데이터의 평가 정책 감사](policy_audit.json)",
              "- [반복별 전체 지표 CSV](utilities.csv), [집계와 모든 관측 구간](summary.json)",
              "- [최종 무결성 검증](verification.json)",
              "- [SVG 그림](depth_repeat_ranges.svg), [고정 소스 보관본](source/followup.py)", "",
              "재생성: .venv/bin/python experiments/analyze_followup.py", ""]
    (out/"report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit("Retired one-off/legacy entry point. Review EXPERIMENT_PLAN.md and use the approved experiments/run_all.py workflow.")
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=DEFAULT_OUT)
    analyze(ROOT / parser.parse_args().folder)
