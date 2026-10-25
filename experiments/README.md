# 로컬과 CUDA 서버의 공통 실험 인터페이스

실험 계획과 분석 절차는 승인됐다. **사전 검증을 모두 통과해야 모델 추론을 시작한다.**

## 디렉토리 구조

~~~text
experiments/                         실행 코드와 잠금된 프로토콜
├── run_all.py                       로컬/CUDA 공통 명령
├── run_parallel.py → run.py         GPU 작업자와 실행 기록
├── backends.py + metrics.py         모델 어댑터와 승인된 점수 계산
├── prepare_study.py                 선택된 장면 입력 manifest와 사전 검증
├── prepare_source_data.py           확장 장면 원본 자료·블러 생성
├── prepare_official_gt.py           공식 GT snapshot 생성
├── select_official_scenes.py        공식 GT 기반 사진 선정
├── study_v1.json + weights.json     프로토콜·가중치 잠금 자료
├── analyze_study.py                공통 analyze 진입점
├── analysis/                       검산·집계·그림·한국어 보고서 모듈
└── test_*.py                        검증 코드

data/experiments/eth3d_multi_view/<scene>/
├── manifest.json                    장면 입력과 5·10·15장 집합
├── official_ground_truth_manifest.json
└── <image-id>/                       사진 변형과 사진별 공식 GT

results/
├── study_configuration.json         과학적 절차 잠금
├── runtime_configurations.json      실행 장치·환경 기록
└── executions/<scene>/              원시 실행 결과·상태
~~~

실행 모듈은 현재 CLI 경로와 소스 해시 잠금을 유지하도록 한 곳에 둔다. 생성 데이터와 결과는 각각 `data/`와 `results/` 아래에 분리한다. 승인된 단일 실행에서 호출되지 않는 이전 반복 후속 분석 코드는 제거했다.

첫 대상은 **office 초기 주실험**이다. office도 전체 주 분석에 포함한다. 5·10·15장, 타깃 전부, 블러 12단계, 단일 seed 2026, 두 모델을 적용한다. 공식 사진 선정에 통과한 확장 9개 장면도 같은 방법을 적용한다. 모든 장면의 실행 위치는 --target local|cuda-server로 선택하며 기본값은 local이다.

## 장면과 실행 위치를 각각 선택한다

~~~sh
# 기본 local: 현재 컴퓨터에서 실행
.venv/bin/python experiments/run_all.py prepare --scene office
.venv/bin/python experiments/run_all.py run --scene office
.venv/bin/python experiments/run_all.py status --scene office
.venv/bin/python experiments/run_all.py analyze --scene office

# 등록한 CUDA 서버에서 같은 동작 실행
.venv/bin/python experiments/run_all.py prepare --scene office --target cuda-server
.venv/bin/python experiments/run_all.py run --scene office --target cuda-server
.venv/bin/python experiments/run_all.py status --scene office --target cuda-server
.venv/bin/python experiments/run_all.py analyze --scene office --target cuda-server

# 모든 장면에 동일한 옵션
.venv/bin/python experiments/run_all.py run --scene courtyard --target local
.venv/bin/python experiments/run_all.py run --scene courtyard --target cuda-server
.venv/bin/python experiments/run_all.py run --remaining --target cuda-server
.venv/bin/python experiments/run_all.py analyze --target cuda-server
~~~

--target local이 기본값이며 현재 호스트에서 계산한다. GPU는 CUDA가 있으면 CUDA, 아니면 MPS를 사용한다. --target cuda-server는 설정된 서버에 SSH로 같은 프로그램을 실행하고 서버 CUDA를 사용한다. office와 다른 장면의 위치를 별도로 지정하거나 office 완료를 요구하는 분기는 없다. 두 target 모두 CPU로 조용히 전환하지 않는다.

--remaining은 공식 선정 통과 office 외 9개 장면을 고르는 옵션이며 실행 위치와 무관하다. --device는 실행 호스트의 GPU를 지정하는 전문가용 옵션이고, 실행 위치는 --target으로 고른다.

prepare/run/analyze는 승인·구현 준비 검사를 통과해야 하며 서버에서도 다시 검사한다. status는 읽기 전용으로 선택한 위치의 기록을 읽고 GPU나 결과 폴더를 만들지 않는다. 서버 연결 실패 시 local로 변경하지 않는다.

## 공식 깊이 GT 스냅샷

공식 ETH3D 깊이 아카이브에서 주 분석 10개 장면의 공식 선정 15장 사진별 깊이와 유효 마스크를 만들려면 다음 명령을 실행한다.

~~~sh
.venv/bin/python experiments/prepare_official_gt.py
~~~

결과는 `data/experiments/eth3d_multi_view/<scene>/<image-id>/` 안의 `official_depth_m.npy`, `official_valid_mask.npy`, `official_gt_metadata.json`이다. 깊이 배열은 계획에서 고정한 252×160 평가 격자이며, 마스크는 공식 GT의 유효한 양수 깊이 위치만 표시한다. 격자·마스크는 공식 선정 기록과 대조된다. 장면별 `official_ground_truth_manifest.json`은 파일 해시와 출처를 기록한다. `laser_depth.npy`는 기존 레이저 투영 참고 자료로 그대로 둔다.

## 구현된 가속 실행부

run_all.py는 사용자 인터페이스, run_parallel.py는 공통 GPU 실행부, run.py는 내부 worker다.

- VGGT 한 worker, MASt3R 세 worker.
- worker마다 CPU thread 8개.
- 추론과 정렬은 같은 GPU.
- float32, highest 행렬곱 정밀도, TF32 비활성화.
- MASt3R complete symmetric graph, coarse 300회 + fine 300회.
- clean baseline과 결과 저장의 파일 잠금·원자적 쓰기.
- 같은 장면·seed 안의 신경망 pair cache만 공유.
- canonical map·초기화·정렬은 각 정확한 입력 집합마다 새로 계산.
- 한 GPU 실행 그룹에서 동시에 활성화되는 장면·seed는 하나로 제한.
- seed를 worker와 환경 기록에 전달.

가속된 기본 코드를 유지하면서 연구 방법은 승인 문서로 고정한다. 이전 실행의 예측값과 cache는 재사용하지 않는다.

## 과학적 잠금과 실행 환경 잠금

연구 전체의 study_configuration.json은 승인 문서·프로토콜·모델 설정·과학적 코드의 해시를 잠근다. MPS와 CUDA에서 이 부분은 같다.

runtime_configurations.json은 실행 그룹별 GPU·호스트·플랫폼·설치 패키지·환경을 기록하고 잠근다. 장면 폴더의 cohort_configuration.json에도 해당 환경을 보존한다. 장치·호스트·seed는 파일 내용에 기록하고 디렉토리 이름에는 넣지 않는다. Mac과 CUDA 서버의 환경이 bit 단위로 같다고 주장하지 않는다. scene_assignments.json은 각 장면이 처음 배정된 실행 그룹을 기록해, 결과를 본 뒤 장치가 다른 값으로 덮어쓰거나 한 장면의 ADD/DROP을 플랫폼 사이에서 섞지 못하게 한다.

기본 결과 루트는 프로젝트 바로 아래 results/이며 protocol_id 이름의 중간 폴더를 추가하지 않는다. --results-root로 변경할 수 있다. 기본 입력 루트는 이미 준비 중인 data/experiments/eth3d_multi_view/다. 기존 자료를 보존하며 기본 구조는 양쪽에서 같다.

~~~text
data/experiments/eth3d_multi_view/<scene>/manifest.json
results/study_configuration.json
results/scene_assignments.json
results/runtime_configurations.json
results/executions/<scene>/cohort_configuration.json
results/executions/<scene>/batch_status.json
results/executions/<scene>/inputs.json
results/executions/<scene>/vggt/
results/executions/<scene>/mast3r/
results/analysis/report.md
~~~

office도 선택한 실행 위치와 관계없이 주 분석에 포함한다. 전체 보고서는 장치 표기를 유지하고 혼합 플랫폼 평균과 플랫폼별 요약을 구분한다. 같은 장면 내부의 모델·ADD/DROP은 같은 플랫폼에서 수행한다. MPS와 CUDA 차이는 연구 결론에 영향을 주지 않는다고 가정하며 별도 장치 교차 실험은 하지 않는다. 재실행 변동도 측정하지 않는다.

## 승인 후 완성하고 검증할 부분

[study_v1.json](study_v1.json)의 status는 approved다. 구현 준비 여부는 사전 검증 결과를 확인한 뒤 갱신한다.

공식 GT와 보정 파일을 이용한 정식 사진 선정은 [선정 스크립트](select_official_scenes.py)로 완료했고, 선정된 사진들의 공식 깊이·마스크 스냅샷도 준비했다. `prepare_study.py`는 정식 RGB·블러 입력 manifest를 만들고 추론 전 검증을 수행한다. `run.py`는 해당 GT를 해시·선정 명단과 대조하고, 두 모델의 깊이 예측을 동일 공통 격자로 표본화해 고정 마스크에서 채점한다. `analyze_study.py`와 `analysis/`는 저장된 결과를 검산하고 모든 장면에 같은 분석을 적용한다.

모델 실행은 승인과 구현 검증 이후에 가능하다. 가속 실행부가 준비됐다는 것과 새 실험의 전체 과학적 구현이 완료됐다는 것을 구분한다.

옛 단일 카메라·단순 laser raster 전처리와 일회성 office 분석 CLI는 퇴역 처리했다. 순수 함수는 기존 검증에 남아 있지만 이 CLI로 옛 실험을 다시 만들 수 없다. 내부 파라미터가 여러 개면 옛 파서는 첫 카메라를 복사하는 대신 오류를 낸다.

## 서버에 한 번 설정할 내용

조교가 제공한 Amazon GPU 서버를 팀원이 준비할 때는 [CUDA 서버 설정 가이드](../CUDA_SERVER_SETUP.md)를 따른다. SSH, Python/CUDA, 코드·가중치·자료 전송, 연결 확인과 인수인계를 설명한다.

[cuda_server.example.json](cuda_server.example.json)을 experiments/cuda_server.json으로 복사해 host, project_dir, python, port를 설정한다. host는 SSH 별칭 또는 user@host, project_dir는 서버 저장소 절대 경로, python은 서버 가상환경 실행 파일이다. 실제 profile은 Git에서 제외한다. SSH 키·접속 설정은 기존 SSH 구성을 사용한다.

서버에는 같은 코드·승인 문서·가중치·준비 RGB/GT snapshot을 배치하고 해당 CUDA 환경에 패키지를 설치한다. 실제 버전은 실행 그룹별로 기록한다. 서버 target은 준비된 서버 자료로 명령을 실행하며 자료의 자동 업로드·동기화와는 별도다.

--protocol은 양쪽 저장소에서 같은 상대 경로를 사용한다. --input-root와 --results-root는 선택한 실행 위치의 경로다. status/analyze는 그 위치의 결과만 읽는다. 여러 위치의 결과를 합치려면 최종 결과와 연구 잠금을 공통 결과 루트에 모아 과학적 해시와 장면 배정을 확인한다. office에만 적용되는 선행 결과나 전송 조건은 없다.

SSH 명령 전달은 가상 profile·subprocess로 검증한다. 실제 서버 접속과 CUDA 실기기 확인은 서버 준비 후 수행한다.

## 모든 장면에 같은 분석을 적용한다

run은 장면별 원시 예측·점수·입력·환경·복구 이력을 저장한 뒤 끝난다. 분석 모듈이 없어도 실행할 수 있으며 자동 분석하지 않는다. 주 지표·U·실패 처리·집계 정의는 먼저 고정한다.

`analyze --scene office`는 `results/executions/office/`의 원시 예측·점수·실행 기록과 공식 GT/RGB snapshot을 검증하고, CPU에서 점수를 다시 계산한 뒤 `results/analysis/`에 그래프·CSV·한국어 보고서를 만든다. 장면 옵션이 없으면 전체 주 분석 명단을 요청 범위로 삼고 미계산 장면도 상태 표에 남긴다. 장면별 상세 보고서는 `results/analysis/scenes/<scene>/report.md`다. 상위 `results/analysis/report.md`와 통합 표는 이번 요청 범위로 생성하며 다른 장면의 상세 보고서는 보존한다.

분석 정의·분모·전환 구간·민감도·미해결 처리·산출물·검증은 [공통 분석 설계](ANALYSIS_METHOD.md)에 설명했다. 주 지표의 문턱은 프로토콜에서 읽고, 포즈를 포함한 모든 U는 `E_DROP − E_ADD` 원래 차이로 재계산한다. 누락·실패를 분모에서 숨기지 않으며, 장면을 묶을 때 장면별 비율을 같은 가중치로 평균한다. 전체·장치별·실행 그룹별 표를 함께 만든다.

장면별 보고서에는 저장된 실제 깊이·포즈·내부 파라미터로 만든 RGB 3D 점군도 자동으로 추가한다. `Original RGB | VGGT reconstruction | MASt3R reconstruction` 비교와, 파일이 준비된 경우 같은 대표 타깃의 `DROP | ADD clean | ADD blurred` 그림을 만든다. 모델 사이의 가상 시점·화각·점 크기와 좌표 정렬 기준은 통일하며 추가 추론은 실행하지 않는다. `reconstruction_visualization.json`에 설정·출처·정렬·점 수를 보존한다.

분석은 신경망 재실행을 하지 않는다. 모든 저장 예측의 해시와 공식 GT 재채점이 일치해야 보고서를 교체한다. 분석 코드·입력·출력 해시, 적용한 프로토콜과 실제 분석 코드 사본을 결과에 보존한다. 테스트는 `.venv/bin/python -m unittest discover -s experiments -p test_analysis.py -v`로 실행한다. 예전 office 반복·20장·6단계 전용 분석 CLI는 퇴역 상태이며 새 공통 분석에는 사용하지 않는다.

## 진행 상태와 재개

batch_status.json은 각 장면·seed 아래에 저장한다. 계획 조건 수는 manifest의 실제 단계 수와 5·10·15장 집합으로 계산한다. 공식 선정 후 계획은 office 726회·전체 7,260회 재구성이며 복구 비용은 별도다.

SIGINT·SIGTERM은 현재 재구성이 끝난 뒤 새 작업을 시작하지 않도록 요청한다. worker가 실패 코드로 종료하면 다음 모델을 자동 실행하지 않는다.

complete는 계획 비교가 모두 평가되고 모든 지표가 유효한 경우, complete_with_failures는 모두 평가했지만 일부 지표가 실패한 경우다. 수정 계획은 같은 조건의 자동 시도를 최초 포함 최대 3회로 정하고, 계속 실패하면 중단·원인 수정·검증·영향 조건 재계산을 수행한다. 첫 유효 결과를 사용하며 모든 실패·복구 이력을 보존한다. 미해결 조건을 숨기거나 정상값으로 조작하지 않는다. 이 복구 정책은 승인 후 worker·재개 구현에 반영할 항목이며 현재 구현 완료를 뜻하지 않는다.

## 자료와 검증

- [전체 계획](../EXPERIMENT_PLAN.md)
- [기계 판독 설정](study_v1.json)
- [공식 archive의 장면별 metadata](dataset_inventory.json)
- [공식 15·10·5장 사진 명단과 장면 판정](official_selection_summary.md)
- [선정 계산 상세 JSON](../data/experiments/eth3d_multi_view/official_selection.json)
- [선정 계산 스크립트](select_official_scenes.py)
- [가중치 revision과 SHA-256](weights.json)

이전 결과·cache는 작업 트리에서 삭제했고 원본 데이터와 모델 가중치는 유지했다. Git 과거 이력은 재작성하지 않았다.

~~~sh
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/run_all.py --help
.venv/bin/python experiments/run_all.py status --scene office
~~~

시험은 가상의 배열·파일·worker와 로컬 보정 metadata를 사용하며 실제 사전학습 모델 추론을 하지 않는다. 모든 장면의 local/CUDA 서버 target·SSH 전달·기본 local, 승인 전 실행 차단, 단일 seed 전달, 1+3 worker, 장면·환경 잠금, 실패 보존을 검증한다.
