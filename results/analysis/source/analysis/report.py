"""Korean explanations generated from the same audited tables as the figures."""
import re

from .core import PRIMARY, LABELS, representative_ids

KOREAN = {"helpful": "도움", "harmful": "해로움", "within_threshold": "문턱 이내", "unavailable": "평가 불가"}
RECONSTRUCTION_START = "<!-- reconstruction-visualization:start -->"
RECONSTRUCTION_END = "<!-- reconstruction-visualization:end -->"


def reconstruction_section(info):
    lines = [RECONSTRUCTION_START, "## RGB 사진에서 복원한 실제 3D 공간", ""]
    if info["status"] != "generated":
        lines += ["두 모델의 저장된 clean 재구성이 모두 준비된 뒤 같은 규칙으로 3D 비교 그림을 생성한다. 이 분석에서 추가 신경망 추론은 실행하지 않는다.", ""]
    else:
        size, reference = info["size"], info["reference_image_id"]
        camera = info["shared_virtual_camera"]
        lines += ["![Original RGB와 두 모델의 RGB 3D 점군](figures/reconstruction_comparison.png)", "",
                  f"**그림:** 서로 다른 위치에서 찍은 RGB 사진 {size}장으로 3D 구조를 복원했다. 왼쪽은 선명한 입력 사진 {reference} 한 장이고, 가운데·오른쪽은 각각 VGGT·MASt3R가 복원한 geometry를 RGB 색의 3D 점군으로 표시한 결과다.", "",
                  f"저장된 깊이·카메라 내부 파라미터·포즈를 사용해 각 점의 3D 위치를 계산하고 실제 입력 RGB로 색을 입혔다. 두 점군은 추정 카메라 중심을 같은 GT 카메라 중심에 전역 Sim(3) 정렬한 좌표계에서 **같은 가상 시점, 수평 화각 {camera['horizontal_fov_deg']:g}°, 점 크기 {camera['point_diameter_px']}픽셀**로 렌더링했다. GT 표면으로 예측 형상을 보정하거나 채우지는 않았다. 가상 시점은 GT와 입력 사진만으로 정했고 원본 RGB 촬영 시점에서 약간 이동했으므로 왼쪽 사진과 완전히 같은 투영은 아니다.", "",
                  "점군은 두 모델에 공통인 영상 격자에서 표본화하며 GT 유효 마스크로 예측 점을 골라내지 않는다. 비유한·비양수 깊이는 그림에서만 제외하고 그 수를 별도 JSON에 기록한다. 점의 겹침·가림·표본 밀도 때문에 보이는 차이도 있으므로, 이 정성 그림을 실제 AbsRel·포즈 점수와 함께 읽는다.", ""]
        contrast = info["contrast"]
        if contrast["status"] == "generated":
            lines += ["![DROP, ADD clean, ADD blurred의 공통 컨텍스트 3D 점군](figures/reconstruction_conditions.png)", "",
                      f"**그림:** 대표 타깃 {contrast['target']}을 뺀 DROP, 선명한 타깃을 넣은 ADD clean, yaw {contrast['blur_deg']:g}°로 흐린 타깃을 넣은 ADD blurred다. 위 행은 VGGT, 아래 행은 MASt3R이며 여섯 패널 모두 위와 같은 가상 카메라·화각·점 크기를 사용한다.", "",
                      f"타깃은 사진 ID 사전순 가운데 사진으로, 블러는 사전에 설정된 가장 큰 단계로 정했다. 모든 패널에서 **동일한 나머지 {len(contrast['common_image_ids'])}장만 점의 공급원과 정렬 기준으로 사용**한다. ADD는 타깃을 포함해 계산된 예측이지만 타깃 자체의 점은 표시하지 않아, 단순히 점이 추가되는 효과와 나머지 공간의 재구성 변화를 구별한다. 결과가 나쁜 타깃을 사후 선택한 사례는 아니다.", ""]
    lines += [RECONSTRUCTION_END, "", ""]
    return "\n".join(lines)


def insert_reconstruction_section(report, info):
    """Replace only our managed block, retaining all authored text outside it."""
    block = reconstruction_section(info)
    if RECONSTRUCTION_START in report:
        pattern = re.escape(RECONSTRUCTION_START) + r"[\s\S]*?" + re.escape(RECONSTRUCTION_END) + r"\n\n"
        if report.count(RECONSTRUCTION_START) != 1 or len(re.findall(pattern, report)) != 1:
            raise ValueError("Malformed or duplicate reconstruction report block")
        return re.sub(pattern, lambda match: block, report, count=1)
    for anchor in ("## 선정 사진과 정답 표본 수", "## 원시 예측과 점수를 얼마나 확인했는가"):
        if anchor in report:
            index = report.index(anchor)
            return report[:index] + block + report[index:]
    return report.rstrip() + "\n\n" + block


def f(value, digits=4, scale=1):
    return "NA" if value is None else f"{value * scale:.{digits}f}"


def table(headers, rows):
    return ["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in row) + " |" for row in rows), ""]


def bracket(policy):
    value = policy["first_positive_to_negative_bracket"]
    if value:
        suffix = (" (미평가 중간 단계 포함)" if value["intermediate_unavailable"] else
                  " (문턱 이내 중간 단계 포함)" if value["intermediate_within_threshold"] else "")
        return f"[{value['lower_deg']:g}, {value['upper_deg']:g}]°" + suffix
    if policy["clean_class"] != "helpful":
        return "clean부터 " + KOREAN[policy["clean_class"]]
    return "이번 관측 단계에서 전환 미관측"


def methods(protocol):
    eps = protocol["evaluation"]["primary_absrel_effect_floor"]
    return [
        "## 비교 방법과 숫자를 읽는 법", "",
        "입력 사진 묶음은 5·10·15장이다. 각 묶음에서 타깃 한 장을 고르고, 그 사진을 뺀 나머지 사진으로 공간을 복원한 결과를 DROP이라 한다. 같은 나머지 사진에 타깃을 더한 결과가 ADD다. ADD에서는 타깃 한 장만 흐리게 만들며, 나머지는 항상 선명하다. 채점할 때는 타깃을 제외한 같은 나머지 카메라와 같은 GT 픽셀만 비교한다.", "",
        "```mermaid", "flowchart LR", 'C["같은 나머지 사진 m−1장"] --> D["DROP: 나머지 사진만 복원"]',
        'C --> A["ADD: 나머지 사진 + 타깃 한 장 복원"]', 'T["타깃: 선명 또는 지정한 블러"] --> A',
        'D --> E["같은 나머지 카메라·공식 GT 위치에서 오차 비교"]', 'A --> E', "```", "",
        "블러 각도는 노출 동안 카메라가 좌우로 돌아가는 전체 yaw 범위다. 원본을 왜곡 보정한 RGB에서 회전 워핑 33장을 선형 sRGB로 적분했고, 준비 영상은 1008×672 픽셀이다. 0도도 같은 처리 경로를 거쳐 선명 영상과 픽셀이 같은지 확인했다. 이는 특정 합성 흔들림 모델이며 실제 동영상의 모든 흐림을 뜻하지 않는다.", "",
        "주 지표 AbsRel은 예측 깊이와 공식 ETH3D 정답 깊이의 상대 오차를 평균한 값이다. 예를 들어 AbsRel 0.02는 표본 위치의 평균 상대 오차 2%다. 두 모델 모두 같은 252×160 격자·공식 유효 마스크에서 채점한다. 공통 컨텍스트 전체의 유효한 양수 예측으로 GT/예측 비율의 중앙값 한 개를 구해 깊이 스케일을 맞추고, 유한한 0·음수 예측도 그 마스크에서 오차에 넣는다. NaN·Inf는 geometry 평가 불가로 남긴다. 유효 GT 픽셀 수가 많은 사진의 깊이 점수 가중치가 더 크지만 같은 비교에서는 모델·ADD/DROP의 가중치가 같다.", "",
        "모든 지표의 분석 기여도는 **U = E_DROP − E_ADD**다. 양수는 타깃을 추가해 오차가 줄었다는 뜻, 음수는 오차가 늘었다는 뜻이다. 깊이 U를 표시할 때는 100을 곱해 percentage point(pp)로 쓴다. AbsRel 2.00%에서 1.80%로 줄면 U=+0.20pp다. 이는 상대적으로 10% 감소했다는 표현과 단위가 다르다. 포즈의 저장된 `utility` 열은 상대 변화율이므로 주 분석에 사용하지 않고, `E_DROP`·`E_ADD`의 원래 차이와 `utility_raw`를 대조한다.", "",
        f"깊이의 주 효과 문턱은 ε={eps:g}, 즉 {eps * 100:g}pp다. U>+ε는 도움, U<−ε는 해로움, 경계를 포함한 −ε≤U≤+ε는 문턱 이내다. 이 기준은 작은 단일 관측 차이를 분리하는 연구 운영 기준이며 보편적 중요도 기준이나 재실행 안정성의 증명이 아니다. 이전 office 탐색이 기준 선택에 영향을 준 사실을 유지한다. 원시 부호와 미리 정한 문턱 민감도도 함께 제시한다.", "",
        "포즈는 GT 카메라 중심에 반사를 허용하지 않는 Sim(3) 정렬을 적용한다. 보조 지표는 깊이 RMSE(m), 장면 반경으로 정규화한 위치 ATE, 상대 병진 오차, 상대 회전 오차(도)다. 포즈와 깊이는 독립 평가하며 보조 지표에는 원래 단위와 원시 부호를 쓴다.", "",
        "타깃별 값의 중앙값·사분위 범위(IQR)는 **서로 다른 타깃 사이의 분포**다. 신뢰구간·반복 측정 오차 막대가 아니다. seed 2026의 단일 실행이며 복구 시도는 새 표본으로 평균하지 않는다. 중첩된 컨텍스트와 같은 clean 재구성을 공유하는 비교는 서로 독립된 반복이 아니다.", "",
        "각 단계의 주 분모는 계획한 타깃 수 m이다. 도움·해로움·문턱 이내·평가 불가의 합이 100%가 되며, 누락은 평가 불가에 포함하되 원시 표에서 누락과 실패를 구별한다. 유효 타깃만을 분모로 한 유해 비율은 보조 열이다. 장면을 합칠 때는 장면별 비율을 먼저 계산한 뒤 같은 가중치로 평균하고, 장면 평균·중앙값·최솟값·최댓값과 사용 장면 수를 함께 공개한다.", "",
        "전환은 clean에서 도움을 보인 곡선이 이후 처음 해로움을 보인 **관측 구간**이다. 첫 음수 앞에서 마지막으로 관측된 양수 단계와 첫 음수 단계 사이를 기록한다. 문턱 분석에서는 각각 +ε 초과·−ε 미만을 쓴다. 중간에 문턱 이내·미해결 단계가 있으면 구간 내부를 미확인으로 표시한다. 역전환도 모두 보존하고 단조성이나 선형 보간을 가정하지 않는다. 4도까지 음수를 관측하지 않았다는 사실로 연속 임계값이 4도보다 크다고 단정하지 않는다.", "",
    ]


def completion_rows(data, protocol):
    result = []
    for backend in protocol["backends"]:
        rows = [r for r in data["rows"] if r["backend"] == backend]
        cases = {}
        for row in rows:
            cases.setdefault((row["size"], row["target"], row["blur_deg"]), {})[row["metric"]] = row["status"]
        geometry = protocol["evaluation"]["primary_metric"], "geometry_depth_rmse_m"
        pose = [m for m in LABELS if m.startswith("pose_")]
        result.append([data["info"]["scene"], backend.upper(), len(cases),
                       sum(any(v != "missing" for v in c.values()) for c in cases.values()),
                       sum(all(c[m] == "ok" for m in geometry) for c in cases.values()),
                       sum(all(c[m] == "ok" for m in pose) for c in cases.values()),
                       sum(all(v == "ok" for v in c.values()) for c in cases.values()),
                       sum(any(v == "failed" for v in c.values()) for c in cases.values()),
                       sum(any(v == "missing" for v in c.values()) for c in cases.values()), data["info"]["device"]])
    return result


def scene_report(data, stages, curves, transitions, sensitivity, contexts, protocol):
    scene = data["info"]["scene"]
    primary_curves = [c for c in curves if c["metric"] == PRIMARY]
    level_last = protocol["blur_degrees"][-1]
    index = {(r["backend"], r["size"], r["metric"], r["blur_deg"]): r for r in stages}
    lines = [f"# {scene} 실험 분석", "", "**연구 질문:** 여러 사진으로 공간을 복원할 때, 흐린 사진 한 장을 추가하면 그 사진을 뺀 것보다 언제 나빠지는가?", "",
             f"분석 상태: **{'완료된 장면의 최종 기술 통계' if data['info']['complete'] else '진행 중 결과의 중간 기술 통계'}**. 실제 장치는 {data['info']['device'].upper()}, seed는 {data['info']['seed']}다. 이 보고서는 {scene} 한 장면의 지정된 5·10·15장 집합을 설명한다.", "",
             "## 먼저 계산 완료 여부를 확인한다", ""]
    lines += table(["장면", "모델", "계획 비교", "점수 기록 있음", "깊이 2지표 유효", "포즈 3지표 유효", "전체 5지표 유효", "실패 포함", "누락 포함", "장치"], completion_rows(data, protocol))
    comparisons = sum(protocol["sizes"]) * len(protocol["blur_degrees"])
    reconstructions = comparisons + len(protocol["sizes"])
    lines += [f"한 비교는 타깃·집합 크기·블러 단계의 조합이다. 모델당 {comparisons}비교이지만 clean 예측을 여러 타깃에 재사용하므로 별도의 원시 재구성은 모델당 {reconstructions}회다. 첫 표의 유효 수는 저장된 결과를 다시 채점해 확인한 값이다.", "", "## 핵심 관측", ""]
    overview = []
    for backend in protocol["backends"]:
        selected = [c for c in primary_curves if c["backend"] == backend]
        initial_helpful = sum(c["threshold"]["clean_class"] == "helpful" for c in selected)
        initial_harmful = sum(c["threshold"]["clean_class"] == "harmful" for c in selected)
        crossed = sum(c["complete"] and c["threshold"]["observed_clean_positive_to_negative"] for c in selected)
        recovered = sum(c["complete"] and c["threshold"]["returns_positive_after_first_negative"] for c in selected)
        endharm = sum(c["threshold"]["end_class"] == "harmful" for c in selected)
        lines.append(f"- **{backend.upper()}**: {len(selected)}개 타깃·컨텍스트 곡선 중 clean에서 문턱을 넘게 도움을 보인 곡선은 {initial_helpful}개, clean부터 해로운 곡선은 {initial_harmful}개다. 완전 곡선에서 clean 도움→이후 해로움 전환은 {crossed}개, 그 뒤 다시 도움으로 돌아온 곡선은 {recovered}개다. 마지막 {level_last:g}도에서 해로운 비교는 {endharm}개다.")
        for size in protocol["sizes"]:
            a, b = index[(backend, size, PRIMARY, 0)], index[(backend, size, PRIMARY, level_last)]
            cs = [c for c in selected if c["size"] == size]
            overview.append([backend.upper(), size, f"{a['valid']}/{size}",
                             f"{a['helpful']}/{size}", f"{a['harmful']}/{size}", f"{b['harmful']}/{size}",
                             f(sum(c["complete"] and c["threshold"]["observed_clean_positive_to_negative"] for c in cs), 0),
                             f(a["utility_raw"]["median"], scale=100), f(b["utility_raw"]["median"], scale=100)])
    lines += ["", "위의 30개는 5+10+15개 타깃·컨텍스트 조합이며 같은 사진이 여러 집합에 들어갈 수 있다. 30개의 독립 장면이나 반복 표본이라는 뜻이 아니다. 전환 수는 clean부터 유익했던 곡선에서만 센다. 처음부터 해롭거나 문턱 이내였던 곡선은 별도 범주다.", ""]
    lines += table(["모델", "집합 크기", "clean 유효", "clean 도움", "clean 유해", f"{level_last:g}° 유해", "도움→유해 전환 수", "clean U 중앙값(pp)", f"{level_last:g}° U 중앙값(pp)"], overview)
    lines += ["전환이 관측되는 사례와 관측되지 않는 사례가 함께 있다. 전환 수를 비교할 때에는 clean부터 도움인 자격 곡선 수가 모델마다 다르다는 점을 함께 봐야 한다. 단순 전환 개수만으로 모델의 블러 취약성 순위를 정하지 않는다. U가 커지는 것과 모델의 절대 깊이 오차가 작다는 것도 별도로 판단한다.", ""]
    for backend in protocol["backends"]:
        for size in protocol["sizes"]:
            group = [index[(backend, size, PRIMARY, degree)] for degree in protocol["blur_degrees"]]
            available = [r for r in group if r["utility_raw"]["median"] is not None]
            if not available:
                continue
            low = min(available, key=lambda r: r["utility_raw"]["median"])
            end = group[-1]
            if low["blur_deg"] != end["blur_deg"] and end["utility_raw"]["median"] is not None and end["utility_raw"]["median"] > low["utility_raw"]["median"]:
                lines.append(f"- {backend.upper()} {size}장: 타깃 U 중앙값은 {low['blur_deg']:g}도에서 {f(low['utility_raw']['median'], scale=100)}pp였고, 마지막 {level_last:g}도에서 {f(end['utility_raw']['median'], scale=100)}pp로 올라왔다. 이는 이산 단계에서 관측한 비단조 변화이며, 중앙값을 만드는 타깃이 단계마다 같다는 뜻은 아니다.")
    lines += ["", "위 비교는 흐린 타깃 자체를 채점하지 않고 나머지 공통 컨텍스트에 미친 영향을 본다. 강한 블러에서 U가 회복되더라도 흐린 사진 자체나 전체 3D 복원이 좋아졌다고 해석하지 않는다. 모델이 타깃을 덜 사용했는지, 정렬이나 출력이 달라졌는지 등의 원인은 이번 점수만으로 확인할 수 없다. 공통 임계각이나 단조 악화도 가정하지 않는다.", ""]
    lines += methods(protocol)
    lines += ["## 선정 사진과 정답 표본 수", "",
              "사진 집합은 공식 GT의 연결·겹침 규칙으로 모델 추론 전에 고정했다. 작은 집합은 큰 집합 안에 포함된다. 아래 표의 GT 픽셀 수는 같은 공식 격자에서 유효한 표본 수이며, 해당 사진이 타깃일 때는 그 사진의 픽셀을 채점에서 전부 제외한다. 각 컨텍스트가 사용하는 정확한 픽셀 수는 `metrics.csv`에 있다.", ""]
    lines += table(["사진 ID", "CAMERA_ID", *[f"{size}장 집합" for size in protocol["sizes"]], "유효 공식 GT 픽셀"],
                   [[frame["id"], frame["camera_id"],
                     *["포함" if frame["id"] in data["manifest"]["sets"][str(size)] else "—" for size in protocol["sizes"]],
                     frame["official_gt_valid_pixels"]] for frame in data["manifest"]["frames"]])
    lines += ["## 원시 예측과 점수를 얼마나 확인했는가", ""]
    lines += table(["모델", "예정 예측", "예측 파일 해시 확인", "저장 지표 행", "CPU 재채점 행", "재채점 일치"],
                   [[r["backend"].upper(), r["expected_predictions"], r["verified_prediction_hashes"], r["stored_metric_rows"],
                     r["rescored_metric_rows"], r["all_stored_scores_match_cpu_rescoring"]] for r in data["audit"]])
    lines += ["예측 파일은 성공 이력의 SHA-256과 대조했다. 사진 ID 순서·공통 컨텍스트·GT 마스크 해시·픽셀 수·seed·장치·과학 설정을 확인하고, 저장된 깊이 공통 격자와 포즈를 공식 GT로 다시 채점했다. 저장된 E_DROP·E_ADD와 재계산 결과가 허용 오차 안에서 같아야 보고서를 생성한다. RGB 준비본도 snapshot의 파일 해시와 대조한다. 신경망 추론은 추가로 실행하지 않았다.", "",
              "## 깊이 기여도와 실제 오차", "",
              "![타깃별 깊이 기여도](figures/depth_utility.png)", "",
              "가는 선은 타깃 각각, 굵은 선은 타깃 중앙값, 띠는 타깃 사분위 범위다. 같은 집합 크기의 두 모델은 같은 y축 범위를 사용한다. 회색 가운데 띠는 ±ε다. 위는 도움·아래는 손해다. x축은 실제로 측정한 12개 단계만 같은 간격으로 나열하며 점 사이 선은 관측값을 잇는 안내선이다.", "",
              "![깊이 절대 오차](figures/depth_errors.png)", "",
              "이 그림은 U 대신 실제 오차(%)를 보여준다. 낮을수록 정답에 가깝다. DROP은 같은 타깃의 모든 블러 비교에서 같은 기준값이다. 각 점의 타깃 집합은 같은 m개이며, NA가 있으면 표의 유효 수에 표시한다.", "",
              "![범주 구성 비율](figures/depth_classification.png)", "",
              "도움·해로움·문턱 이내·평가 불가를 계획한 타깃 전체를 분모로 보여준다. 평가 불가를 해롭지 않은 값으로 취급하지 않는다.", ""]
    for backend in protocol["backends"]:
        for size in protocol["sizes"]:
            lines += [f"### {backend.upper()} · {size}장: 12개 측정 단계", ""]
            group = [index[(backend, size, PRIMARY, degree)] for degree in protocol["blur_degrees"]]
            lines += table(["yaw(°)", "유효/계획", "DROP 중앙값(%)", "ADD 중앙값(%)", "U 중앙값 [Q25, Q75](pp)", "도움", "유해", "문턱 이내", "평가 불가"],
                [[f"{r['blur_deg']:g}", f"{r['valid']}/{r['planned']}", f(r["E_drop"]["median"], scale=100), f(r["E_add"]["median"], scale=100),
                  f"{f(r['utility_raw']['median'], scale=100)} [{f(r['utility_raw']['q25'], scale=100)}, {f(r['utility_raw']['q75'], scale=100)}]",
                  r["helpful"], r["harmful"], r["within_threshold"], r["unavailable"]] for r in group])
    lines += ["## 전환 구간과 다시 양수가 된 사례", "",
              "아래 표는 모든 주 지표 곡선을 보여준다. 원시 전환은 U=0, 주 전환은 ±ε 기준이다. 구간 끝은 관측된 각도일 뿐 정확한 연속 임계값이 아니다. 역전환 여부는 첫 해로움 뒤 다시 +ε를 넘었는지를 뜻한다. 모든 방향 전환과 중간 문턱 이내·NA 목록은 `curves.json`에 보존한다.", ""]
    lines += table(["모델", "크기", "타깃", "유효 단계", "clean 판정", "원시 첫 구간", "문턱 첫 구간", "다시 도움"],
        [[c["backend"].upper(), c["size"], c["target"], f"{c['valid_levels']}/{c['planned_levels']}",
          KOREAN[c["threshold"]["clean_class"]], bracket(c["raw"]), bracket(c["threshold"]),
          "예" if c["threshold"]["returns_positive_after_first_negative"] else "관측 안 됨"] for c in primary_curves])
    lines += ["## 미리 정한 효과 문턱에 대한 민감도", "",
              "모든 사전 지정 ε를 같은 방식으로 계산했다. 원래 주 기준 ε=0.0005는 유지한다. 행별 전환 수의 분모는 계획 곡선 수 m이고, clean에서 도움인 완전 곡선 수는 별도 열이다. 문턱에 따라 사례 수가 바뀌는 정도를 보여주며 결과에 맞는 문턱을 선택하지 않는다.", ""]
    sensitivity_rows = []
    for row in sensitivity:
        backend, size, eps = row["backend"], row["size"], row["epsilon"]
        end = next(r for r in data["sensitivity_stages"] if r["backend"] == backend and r["size"] == size and r["epsilon"] == eps and r["blur_deg"] == level_last)
        sensitivity_rows.append([backend.upper(), size, f"{eps:g}", row["eligible_complete_clean_helpful"],
                                 f"{row['transitions_complete']}/{size}", f"{end['harmful']}/{size}", row["incomplete_curves"]])
    lines += table(["모델", "크기", "ε(AbsRel 원단위)", "clean 도움 완전 곡선", "전환/계획 곡선", f"{level_last:g}° 유해/계획", "미완전 곡선"], sensitivity_rows)
    lines += ["## 포즈와 다른 깊이 지표", "", "![보조 지표](figures/secondary_metrics.png)", "",
              "보조 지표도 E_DROP−E_ADD 원래 차이를 사용하므로 양수는 도움이다. 깊이 RMSE의 단위는 m, 정규화 ATE는 장면 반경 비율, 상대 병진은 기준선 대비 비율, 상대 회전은 도다. 서로 단위가 다른 지표의 U 크기를 직접 비교하지 않는다. 원시 부호를 사용하며 보조 지표의 문턱은 결과를 본 뒤 추가하지 않는다.", ""]
    secondary_rows = []
    for metric in protocol["evaluation"]["secondary_metrics"]:
        for backend in protocol["backends"]:
            for size in protocol["sizes"]:
                a, b = index[(backend, size, metric, 0)], index[(backend, size, metric, level_last)]
                secondary_rows.append([LABELS[metric][0], backend.upper(), size, f(a["E_drop"]["median"]),
                    f(a["E_add"]["median"]), f(b["E_add"]["median"]), f(a["utility_raw"]["median"]),
                    f(b["utility_raw"]["median"]), f"{a['harmful']}/{size}", f"{b['harmful']}/{size}"])
    lines += table(["지표", "모델", "크기", "DROP 중앙값", "clean ADD", f"{level_last:g}° ADD", "clean U", f"{level_last:g}° U", "clean 음수", f"{level_last:g}° 음수"], secondary_rows)
    lines += ["## 사례는 미리 정한 위치에서 고른다", "", "![고정 대표 타깃](figures/representative_targets.png)", "",
              "각 집합에서 사진 ID를 사전순으로 정렬해 첫 번째·가운데·마지막을 선택했다. 극적인 결과를 보고 고른 사례가 아니다. 사례의 빈도를 전체 타깃의 빈도처럼 해석하지 않는다.", ""]
    cases = []
    row_index = {(r["backend"], r["size"], r["target"], r["blur_deg"]): r for r in data["rows"] if r["metric"] == PRIMARY}
    for size in protocol["sizes"]:
        for target in representative_ids(data["manifest"]["sets"][str(size)]):
            for backend in protocol["backends"]:
                a, b = row_index[(backend, size, target, 0)], row_index[(backend, size, target, level_last)]
                cases.append([backend.upper(), size, target, f(a["E_drop"], scale=100), f(a["E_add"], scale=100),
                              f(b["E_add"], scale=100), f(a["utility_raw"], scale=100), f(b["utility_raw"], scale=100)])
    lines += table(["모델", "크기", "타깃", "DROP(%)", "clean ADD(%)", f"{level_last:g}° ADD(%)", "clean U(pp)", f"{level_last:g}° U(pp)"], cases)
    lines += ["## 비정상 깊이와 엄격 채점 진단", "",
              "이 표는 저장된 예측을 분석 중 다시 읽어 공식 고정 마스크에서 계산했다. 비양수는 유한한 0·음수이고, 비유한은 NaN·Inf다. 같은 DROP 또는 clean 예측이 여러 비교에 재사용되므로 아래 비교 수·픽셀 수를 독립 추론 횟수로 더하지 않는다. 비양수가 있으면 주 점수는 오차에 포함하지만 기존 엄격 채점은 평가 불가다. 엄격 점수는 진단만 하며 주 결론을 교체하지 않는다.", ""]
    diag_rows = []
    for backend in protocol["backends"]:
        group = [d for d in data["diagnostics"] if d["backend"] == backend]
        for role in ("drop", "add"):
            diag_rows.append([backend.upper(), role.upper(), len(group),
                sum(d[role + "_nonpositive_pixels"] > 0 for d in group),
                f(max((d[role + "_nonpositive_fraction"] for d in group), default=0), scale=100),
                sum(d[role + "_nonfinite_pixels"] > 0 for d in group),
                sum(not d[role + "_strict_positive_depth_valid"] for d in group)])
    lines += table(["모델", "역할", "진단 비교", "비양수 포함 비교", "최대 비양수 비율(%)", "NaN/Inf 포함 비교", "엄격 채점 불가"], diag_rows)
    lines += ["## GT 컨텍스트와의 관계: 탐색 분석", "", "![GT 컨텍스트 탐색](figures/context_exploration.png)", "",
              "사진 간 공식 GT 겹침은 추론 전에 저장된 선정 결과에서 읽었다. 기준선·상대 회전·깊이 분포는 추론 전 고정한 GT 포즈·깊이 snapshot으로 분석 중 파생했다. 이 파생 시점을 기록하며, 미리 별도 표가 저장됐다고 주장하지 않는다. 산점도는 clean와 마지막 단계의 U가 이런 컨텍스트 특성과 함께 어떻게 나타나는지 보여준다. 표본이 중첩되고 장면도 하나이므로 원인·일반 법칙·유의성을 추정하지 않는다. 지표·표본을 사후 선택한 확증 분석으로 바꾸지 않는다.", "",
              "## 실행 비용과 복구 기록", ""]
    lines += table(["모델", "시도", "성공", "실패", "추가 시도", "추론 구간 합집합(분)", "worker 시간 합(분)", "최대 단일 worker RSS(GiB)"],
        [[r["backend"].upper(), r["attempts"], r["successful_attempts"], r["failed_attempts"], r["retries"],
          f(r["inference_interval_union_seconds"] / 60, 1), f(r["summed_worker_inference_seconds"] / 60, 1),
          f(r["max_single_worker_process_rss_bytes"] / 2**30, 2)] for r in data["costs"]])
    lines += ["추론 구간 합집합은 여러 worker가 동시에 계산한 시간을 중복해서 더하지 않은 값이다. worker 시간 합은 병렬 작업 시간을 더한 값으로 실제 경과 시간과 다르다. 모델 로드·파일 검증 등은 이 추론 구간에 전부 포함되지 않는다. RSS는 프로세스 메모리이며 동시 전체 메모리나 GPU 메모리의 최대값이 아니다.", "",
              "실행 중 문서 준비 상태 설명의 해시 변경으로 controller가 한 번 중단된 이력은 `results/plan_fingerprint_reconciliation.json`에 남아 있다. 그 파일이 있는 경우 분석 입력 해시 목록에도 포함한다. 이 보고서의 점수 재검산은 현재 보존된 실행 설정과 원시 예측의 일치를 확인한다.", "",
              "## 해석 범위와 다음 장면에 적용할 규칙", "",
              f"이 보고서는 {scene}의 특정 합성 yaw 노출, 선정된 컨텍스트, 두 모델, 고정 지표와 seed 한 번에 대한 기술 통계다. 반복 안정성, 모든 환경의 공통 임계각, 사진 수만의 인과 효과, 전체 3D 완전성, 실제 프레임 삭제 정책의 개선을 입증하지 않는다. 5·10·15장 집합은 사진 수와 함께 겹침·기준선·시야 구성도 달라진다.", "",
              "MPS와 CUDA의 차이가 연구 결론에 큰 영향을 주지 않는다는 것은 사용자 결정에 따른 가정이다. 별도 교차·반복 실험으로 검증한 결과가 아니다. 이후 장면에도 동일 분석 코드를 적용하고 장면별 집계를 먼저 만든 뒤 같은 장면 가중치로 합친다. 플랫폼별 표는 실행 이력의 보조 기술 통계다.", "",
              "두 모델에서 함께 유효한 동일 조건의 비교는 `paired_comparisons.csv`에 있으며 일치·제외 조건 수와 실제 오차·U를 모두 기록한다. 일부 유효 조건만으로 모델의 보편적 우열을 매기지 않는다. 전체 10장면의 진행 상태와 같은 가중치 집계는 상위 `../../report.md`에서 확인한다.", "",
              "## 재생성과 산출물", "", f"```sh\n.venv/bin/python experiments/run_all.py analyze --scene {scene}\n```", "",
              "- `metrics.csv`: 모든 예정 지표 행과 원래 오차·U·원시 부호·주 판정·누락/실패.",
              "- `stage_summary.csv`: 블러별 오차·U 분포와 네 범주의 계획/유효 분모.",
              "- `curves.json`, `curves.csv`, `transitions.csv`: 완전/미완전 곡선, 원시/문턱 전환, 모든 역전환.",
              "- `sensitivity_stages.csv`, `sensitivity_transitions.csv`: 사전 지정 효과 문턱 전체.",
              "- `depth_diagnostics.csv`, `context_descriptors.csv`, `paired_comparisons.csv`, `audit.json`, `costs.json`.",
              "- `figures/`: PNG와 벡터 PDF. 상위 `analysis_manifest.json`에는 입력·분석 코드·출력 해시가 있다.", ""]
    return "\n".join(lines)


def cohort_report(data_by_scene, cohort, protocol, requested, roster, version):
    available = list(data_by_scene)
    lines = ["# FrameUtility 공통 분석 보고서", "", f"분석 버전 **{version}**. 요청 범위: **{', '.join(requested)}**. 숫자 집계에 사용한 장면: **{', '.join(available) or '없음'}**.", "",
             "**질문:** 흐린 사진 한 장을 추가하는 것이 그 사진을 뺀 것보다 언제 나빠지는가? 주 지표는 공식 GT 공통 격자의 깊이 AbsRel이고 U=E_DROP−E_ADD다. 양수는 도움, 음수는 손해다.", "",
             "## 주 분석 장면 전체의 진행 상황", ""]
    lines += table(["장면", "이번 요청에 포함", "실행 상태", "이번 분석에 포함", "상세 보고서"],
        [[s, "예" if s in requested else "아니오", roster[s], "예" if s in available else "아니오",
          f"[보고서](scenes/{s}/report.md)" if s in available else "대기"] for s in protocol["primary_scenes"]])
    lines += ["계획된 주 분석 장면은 office를 포함한 10개다. 다른 장면이 없으면 office의 결과만 기술하고, 미계산 장면을 유해율 0으로 넣지 않는다. 요청 범위의 일부만 준비됐을 때도 사용/계획 장면 수를 보이며 전체 완료로 표시하지 않는다.", "",
              "사전 제외 장면: " + "; ".join(f"{s}: {reason}" for s, reason in protocol["excluded_scenes"].items()), "",
              "## 완료율과 결과의 의미", ""]
    completion = []
    for data in data_by_scene.values():
        completion.extend(completion_rows(data, protocol))
    lines += table(["장면", "모델", "계획 비교", "점수 기록", "깊이 유효", "포즈 유효", "5지표 유효", "실패 포함", "누락 포함", "장치"], completion)
    lines += ["각 장면의 상세 보고서는 모든 타깃 곡선·실제 오차·문턱 민감도·비정상 깊이·사례·복구 기록을 같은 형식으로 보여준다. 전체 기록이 없어 분석할 수 없는 장면은 상태 표에 남기고, 없는 결과를 추정하거나 보간하지 않는다.", "",
              "## 장면 동일 가중치 집계", "",
              "모델·집합 크기·각도별로 각 장면의 계획 타깃 분모에서 유해 비율을 계산한 다음 장면들을 같은 가중치로 평균한다. 아래는 원래 주 문턱을 적용한 clean/마지막 단계다. 표의 중앙값·최솟값·최댓값도 **장면별 유해 비율 사이의 분포**다. 여러 타깃을 한데 모은 값은 CSV의 보조 열로 구분한다.", ""]
    if len(available) < protocol["minimum_scenes_for_multi_scene_summary"]:
        lines += [f"현재 사용할 수 있는 장면은 {len(available)}개로, 다중 장면 요약의 최소 {protocol['minimum_scenes_for_multi_scene_summary']}개에 못 미친다. 아래 숫자는 확보된 요청 장면의 기술 통계이며 다중 장면 최종 결론으로 해석하지 않는다.", ""]
    selected = [r for r in cohort if r["stratum"] == "all" and r["metric"] == PRIMARY and r["blur_deg"] in (0, protocol["blur_degrees"][-1])]
    lines += table(["모델", "크기", "각도(°)", "사용/요청 장면", "유해율 장면 평균(%)", "중앙값(%)", "최소–최대(%)", "미평가율 평균(%)"],
        [[r["backend"].upper(), r["size"], r["blur_deg"], f"{r['n_scenes_used']}/{r['n_scenes_requested']}",
          f(r["harmful_scene_fraction"]["mean"], 1, 100), f(r["harmful_scene_fraction"]["median"], 1, 100),
          f"{f(r['harmful_scene_fraction']['min'], 1, 100)}–{f(r['harmful_scene_fraction']['max'], 1, 100)}",
          f(r["unavailable_scene_fraction"]["mean"], 1, 100)] for r in selected])
    if available:
        lines += ["![장면 동일 가중치 유해 비율](figures/cohort_harmful_fraction.png)", "",
                  "점은 장면별 유해 비율을 같은 가중치로 평균한 값이고 띠는 사용 장면 사이의 최솟값–최댓값이다. 신뢰구간이 아니다. 회색 점선은 평가 불가 비율의 장면 평균이다. 미계산 장면을 0으로 넣지 않으며 사용/요청 장면 수를 각 패널에 표시한다.", ""]
    lines += ["`tables/cohort_summary.csv`에는 모든 각도와 다섯 지표, 전체·장치별·실행 그룹별 집계가 있다. 장치별 집계는 장치 효과 실험이 아니며 장면 구성 차이를 함께 가진다. MPS/CUDA의 결론상 동등성은 가정이고 별도 교차 검증하지 않았다. 같은 장면의 두 모델과 ADD/DROP은 같은 플랫폼에서 실행됐는지 입력 검증에서 확인한다.", ""]
    lines += methods(protocol)
    lines += ["## 이후 장면에도 같은 분석을 적용한다", "", "```sh",
              ".venv/bin/python experiments/run_all.py analyze --scene office",
              ".venv/bin/python experiments/run_all.py analyze --scene courtyard",
              "# 준비된 모든 주 분석 장면을 같은 규칙으로 다시 집계",
              ".venv/bin/python experiments/run_all.py analyze",
              "# 등록된 서버에서도 같은 인터페이스",
              ".venv/bin/python experiments/run_all.py analyze --target cuda-server", "```", "",
              "`--results-root`로 선택한 실행 위치의 결과 루트를 바꿀 수 있다. 분석은 저장된 예측과 공식 GT/RGB snapshot을 읽는 CPU 작업이다. 신경망·CUDA/MPS 추론을 추가하지 않는다. 실제 파일·사진 ID·GT 마스크·설정·점수 재검산이 실패하면 기존 정상 보고서를 교체하지 않고 오류를 낸다.", "",
              "상위 보고서와 공통 표는 **이번에 요청한 장면 집합**으로 새로 만들어진다. 장면별 보고서는 `scenes/<scene>/`에 보존되므로 다른 장면 하나를 분석해도 앞서 생성한 장면 보고서가 사라지지 않는다. 전체 통합은 scene 옵션 없이 다시 실행한다. 집계는 이전 보고서 숫자를 합치는 대신 매번 원시 결과에서 계산해 분석 버전이 섞이는 것을 막는다.", "",
              "`analysis_definition.json`은 분석 정의·버전·문턱·분모·전환 규칙을 보존한다. `analysis_manifest.json`은 모든 사용 입력·코드·출력 해시와 라이브러리 버전을 보존하고, `source/`에는 실제 분석 코드를 복사한다. 분석 버그를 고치면 같은 코드로 대상 장면 전체를 다시 분석하고 새로운 분석 해시를 비교한다. 모델 실행 결과는 수정하지 않는다.", ""]
    return "\n".join(lines)
