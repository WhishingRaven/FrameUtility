# 후속 실험: 실행 전 고정 사항

평가 기준 정리와 작은 로컬 검증을 수행한다. 원래 420조건과 후속 결과는 분리한다. 단일 장면의 파일럿에서 고른 사례 검증이며 새로운 장면에서의 확인 실험은 아니다.

정의·입력/코드 해시·고정 시각은 [protocol.json](protocol.json)에 있다. 첫 후속 추론 전에 생성했다. 실행 시 정책·소스·원본 결과·입력의 변경을 검사한다. 정책 구현은 [followup_metrics.py](source/followup_metrics.py)에 보관했다.

## 평가

- 주 지표: 타깃을 제외한 동일 공통 컨텍스트의 scale-aligned depth AbsRel.
- 모든 유한한 양수 GT 픽셀로 평가 마스크를 고정한다. 프레임당 최소 100픽셀.
- 전체 컨텍스트의 양수 예측에서 median(GT/prediction)으로 하나의 양수 scale을 구한다. 최소 양수 예측 100개.
- 평가 마스크의 모든 픽셀에 오차를 계산한다. 유한한 음수·0 깊이를 제거하거나 clamp하지 않는다.
- 고정 마스크에 NaN/Inf가 있으면 geometry 실패로 기록한다. Pose는 독립 평가한다.
- 비양수·비유한 예측 개수와 비율, 기존 엄격 점수/실패도 함께 저장한다.
- 보조 지표: depth RMSE, 정규화 ATE, pair translation relative error, pair rotation error.
- U_raw = E_DROP − E_ADD. Pose는 무차원 상대 utility도 보존한다.
- AbsRel 효과 문턱 ε=0.0005 (= 0.05 percentage point). 파일럿을 참고해 실행 전에 고정한 문턱이며 통계적 유의성 기준은 아니다.

## 0° 대조군

20장 모두 clean의 단축 경로를 우회해 33개 항등 homography, 선형 radiance 적분, sRGB 복원·양자화와 동일 crop을 거친다. 디코딩된 픽셀이 clean과 정확히 일치해야 실행한다. 일치하는 대조군은 추론에서 clean과 같은 입력으로 처리한다. 비영 회전의 보간 효과까지 독립적으로 분리하는 대조군은 아니다.

## 선정과 실행

| Backend | 집합 | 타깃 | 블러 단계 (°) | 파일럿에 따른 선정 이유 |
|---|---:|---|---|---|
| VGGT | 5 | DSC_0250 | 0, .025, .05, .075, .1, .15, .2, .25 | clean 효과가 문턱보다 큰 첫 구간 전환 |
| VGGT | 10 | DSC_0242 | 동일 | 10장 집합의 첫 구간 전환 |
| VGGT | 20 | DSC_0252 | 동일 | 양수 clean 효과가 가장 작은 사례 |
| MASt3R | 5 | DSC_0251 | 0, .5, 1 | 작은 clean 효과와 [0.5,1] 전환 |
| MASt3R | 20 | DSC_0238 | 0, 2, 4 | 이미 제시한 강한 5지표 전환 |

각 사례의 DROP과 모든 ADD를 3회씩 새로 실행한다. VGGT 81회, MASt3R 24회로 총 105회다. 이전 실행을 반복 횟수에 포함하지 않는다. Seed는 매번 2026으로 고정하며 같은 seed의 수치적 반복성을 측정한다.

MASt3R의 신경망 pair cache는 반복 블록별로 새로 생성하며 블록 사이에는 공유하지 않는다. 같은 블록 안에서는 신경망 pair 추론만 재사용한다. 모든 ADD/DROP의 정렬은 새로 수행한다. Float32, native preprocessing, complete symmetric graph, 300+300 정렬 반복을 유지한다. 로컬 MPS에서 VGGT는 한 프로세스, MASt3R은 최대 세 반복 프로세스를 병렬 실행한다.

## 판정과 종료 조건

- 3개 DROP × 3개 ADD의 모든 조합을 보고한다. 최소·최대는 관측 범위이며 신뢰구간이 아니다.
- 안정적인 원시 전환: 모든 clean 조합의 U>0이고, 이후 단계 모든 조합의 U<0.
- 효과 문턱을 넘는 전환: 모든 clean 조합의 U>ε이고, 이후 단계 모든 조합의 U<−ε.
- 모든 5개 사례와 실패를 보고한다. 결과를 보고 사례를 추가·제외하거나 문턱을 변경하지 않는다.
- 첫 전환은 관측된 양수 단계와 음수 단계 사이의 구간으로 기록한다. 단조성이나 보간으로 정확한 임계값을 가정하지 않는다.
- 105회와 무결성 검증·분석을 완료하면 종료한다. 다른 장면 확장은 이번 범위 밖이다.

## 실행

~~~sh
.venv/bin/python experiments/followup.py prepare
.venv/bin/python experiments/followup.py audit
.venv/bin/python experiments/followup.py run --backend vggt --device mps
.venv/bin/python experiments/followup.py run --backend mast3r --device mps --replicate 0
.venv/bin/python experiments/followup.py run --backend mast3r --device mps --replicate 1
.venv/bin/python experiments/followup.py run --backend mast3r --device mps --replicate 2
.venv/bin/python experiments/analyze_followup.py
~~~

MASt3R의 세 명령은 독립적인 반복 블록으로 병렬 실행할 수 있다. CUDA에서는 같은 코드에 --device cuda와 새 --output을 사용한다. 원본 office 입력·파일럿 manifest/결과 metadata를 동일한 상대 경로에 두고 새 출력에서 prepare를 먼저 실행한다. 평가 정책 감사 결과도 새 출력에 보관한다. 하드웨어별 결과는 섞지 않는다.
