"""Write empirical tables and figures, preserving incomplete/failed conditions."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import METRICS, condition_counts

ROOT = Path(__file__).resolve().parents[1]


def summarize(folder):
    folder = ROOT / folder
    rows = []
    for backend in ("vggt", "mast3r"):
        path = folder / backend / "utilities.json"
        if path.exists():
            rows.extend(json.loads(path.read_text()))
    folder.mkdir(parents=True, exist_ok=True)
    columns = ["backend", "scene", "size", "n_context", "target", "blur_deg", "metric", "status",
               "E_drop", "E_add", "utility_raw", "utility_relative", "utility", "evaluation_pixels", "error"]
    csv_temp = folder / f".utilities-{os.getpid()}.csv"
    with csv_temp.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    csv_temp.replace(folder / "utilities.csv")
    primary = ["geometry_depth_absrel", "pose_ate_normalized", "pose_pair_rotation_deg"]
    labels = ["Geometry U (depth AbsRel)", "Pose U (relative ATE change)", "Pose U (relative rotation-error change)"]
    curve_levels = {}
    for r in rows:
        if r["status"] == "ok" and r["utility"] is not None:
            curve_levels.setdefault((r["backend"], r["size"], r["target"], r["metric"]), set()).add(r["blur_deg"])
    plot_sizes = sorted({key[1] for key, levels in curve_levels.items() if len(levels) == 6}) or [5]
    fig, axes = plt.subplots(len(plot_sizes), 3, figsize=(13, 3.5*len(plot_sizes)), sharex=True, squeeze=False)
    aggregate = []
    for row_idx, size in enumerate(plot_sizes):
        for col_idx, (metric, label) in enumerate(zip(primary, labels)):
            ax = axes[row_idx, col_idx]
            for backend, color in [("vggt", "#2868b2"), ("mast3r", "#c65d23")]:
                subset = [r for r in rows if r["backend"] == backend and r["size"] == size
                          and r["metric"] == metric and r["status"] == "ok" and r["utility"] is not None]
                complete_targets = {target for target in {r["target"] for r in subset}
                                    if len({r["blur_deg"] for r in subset if r["target"] == target}) == 6}
                subset = [r for r in subset if r["target"] in complete_targets]
                x, y = [], []
                for severity in sorted({r["blur_deg"] for r in subset}):
                    values = [r["utility"] for r in subset if r["blur_deg"] == severity]
                    x.append(severity)
                    y.append(float(np.median(values)))
                    aggregate.append({"backend": backend, "size": size, "metric": metric, "blur_deg": severity,
                                      "n": len(values), "planned_n": size, "median_utility": y[-1],
                                      "fraction_negative_observed": float(np.mean(np.array(values) < 0))})
                if x:
                    ax.plot(x, y, "o-", label=f"{backend.upper()} (n={len(complete_targets)})", color=color)
            ax.axhline(0, color="gray", linewidth=.8)
            ax.set_title(f"{size} views · {label}", fontsize=10)
            ax.set_ylabel("Median utility")
            ax.grid(alpha=.2)
            if row_idx == len(plot_sizes)-1:
                ax.set_xlabel("Exposure yaw path (degrees)")
            if ax.lines and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
    fig.suptitle("FrameUtility · ETH3D office · complete observed curves only")
    fig.tight_layout()
    figure_temp = folder / f".utility_curves-{os.getpid()}.png"
    fig.savefig(figure_temp, dpi=170)
    figure_temp.replace(folder / "utility_curves.png")
    plt.close(fig)
    (folder / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    crossings = []
    for backend in ["vggt", "mast3r"]:
        for size in [5, 10, 20]:
            for target in sorted({r["target"] for r in rows if r["backend"] == backend and r["size"] == size}):
                for metric in METRICS:
                    curve = sorted([r for r in rows if r["backend"] == backend and r["size"] == size
                                    and r["target"] == target and r["metric"] == metric
                                    and r["status"] == "ok" and r["utility"] is not None], key=lambda r: r["blur_deg"])
                    if not curve:
                        continue
                    negative = [r["blur_deg"] for r in curve if r["utility"] < 0]
                    clean = next((r["utility"] for r in curve if r["blur_deg"] == 0), None)
                    crossings.append({"backend": backend, "size": size, "target": target, "metric": metric,
                                      "observed_levels": len(curve), "planned_levels": 6,
                                      "clean_utility": clean, "first_observed_negative_deg": min(negative) if negative else None,
                                      "positive_to_negative_brackets_deg": [[a["blur_deg"], b["blur_deg"]]
                                          for a, b in zip(curve, curve[1:]) if a["utility"] > 0 and b["utility"] < 0],
                                      "complete_curve": len(curve) == 6})
    (folder / "transitions.json").write_text(json.dumps(crossings, indent=2) + "\n")
    counts = condition_counts(rows)
    report = ["# FrameUtility 실험 결과", "", f"갱신: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
              f"평가한 ADD/DROP 조건: **{counts['attempted']}/420**. 모든 지표 정상 {counts['ok']}개, 일부 지표 실패 {counts['partial']}개, 모든 지표 실패 {counts['failed']}개.", "",
              "조건마다 pose 3개와 geometry 2개 지표를 별도로 계산한다. 잘못된 깊이는 geometry 실패로 기록하며, 유효한 pose 결과는 유지한다.", "",
              "ETH3D office 한 장면의 탐색 실험이다. 입력/GT/checksum은 `inputs.json`, 장비·버전·가중치는 각 backend의 `environment.json`에 있다.", "",
              "| Backend | 이미지 수 | 평가한 조건 / 계획 | 모든 지표 정상 | 일부 실패 | 모두 실패 |",
              "|---|---:|---:|---:|---:|---:|"]
    for backend in ("vggt", "mast3r"):
        for size in (5, 10, 20):
            group = condition_counts([r for r in rows if r["backend"] == backend and r["size"] == size])
            report.append(f"| {backend} | {size} | {group['attempted']}/{size*6} | {group['ok']} | {group['partial']} | {group['failed']} |")
    report.extend(["", "![완성된 곡선의 utility](utility_curves.png)", "",
                   "그림과 aggregate 통계는 6단계가 모두 관측된 타깃 곡선만 사용한다. 진행 중인 조건도 전체 `utilities.csv`에는 보존된다.", "",
                   "## 첫 타깃의 5장 결과", "",
                   "Geometry U는 depth AbsRel의 원래 차이이다. Pose U는 DROP 대비 오차 감소율(%)이며, 음수면 해당 지표에서 추가 뷰가 해롭다.", "",
                   "| Backend | Target | Blur 경로 (°) | Geometry U | Pose ATE U (%) | Pose rotation U (%) |",
                   "|---|---|---:|---:|---:|---:|"])
    for backend in ("vggt", "mast3r"):
        successful = [r for r in rows if r["backend"] == backend and r["size"] == 5 and r["status"] == "ok"]
        if not successful:
            continue
        first_target = min(r["target"] for r in successful)
        for severity in sorted({r["blur_deg"] for r in successful if r["target"] == first_target}):
            values = {r["metric"]: r["utility"] for r in successful if r["target"] == first_target and r["blur_deg"] == severity}
            def fmt(metric, percent=False):
                value = values.get(metric)
                return "—" if value is None else f"{value*(100 if percent else 1):.6f}"
            report.append(f"| {backend} | {first_target} | {severity:g} | {fmt(primary[0])} | {fmt(primary[1], True)} | {fmt(primary[2], True)} |")
    report.extend(["", "## 검증과 해석 범위", "",
                   "- clean 반복 결과는 각 backend의 `repeatability.json`에 기록한다. 반복 오차보다 작은 변화에는 의미를 부여하지 않는다.",
                   "- Clean에서 이미 U<0인 조건은 처음부터 해로운 뷰로 분리한다. 이를 blur 때문에 양수에서 음수로 전환된 사례로 세지 않는다.",
                   "- Geometry 참조는 정렬된 laser scan의 z-buffer 투영이다. 공식 ETH3D occlusion 처리를 포함하지 않으며, 공식 benchmark 점수가 아니다.",
                   "- 두 backend는 같은 이미지 집합과 회전 노출 경로를 받는다. GT는 blur 생성·선정·평가에만 사용한다.",
                   "- 한 장면의 일부 관측으로 일반적인 임계 blur나 모델 우열을 결론 내릴 수 없다.", "",
                   "실행·CUDA 이전 방법: [실험 실행 문서](../../experiments/README.md).", ""])
    (folder / "report.md").write_text("\n".join(report))
    print(f"Evaluated conditions: {counts['attempted']}/420; fully valid {counts['ok']}, partial {counts['partial']}, failed {counts['failed']}; metric rows: {len(rows)}")
    print(folder / "utilities.csv")
    print(folder / "utility_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default="results/office_sweep")
    summarize(parser.parse_args().folder)
