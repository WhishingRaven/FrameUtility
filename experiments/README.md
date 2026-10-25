# 로컬과 CUDA 서버의 공통 실험 인터페이스

승인 대상은 [실험 계획과 분석 절차](../EXPERIMENT_PLAN.md)다. 현재는 **승인 대기 / 새 과학적 방법 구현 검증 미완료**이며 새 모델 추론은 시작하지 않았다.

첫 대상은 **office 초기 주실험**이다. office도 전체 주 분석에 포함한다. 5·10·15장, 타깃 전부, 블러 12단계, 단일 seed 2026, 두 모델을 적용한다. 나머지 11개 후보는 CUDA 서버 준비 후 같은 방법으로 확장한다.

## 사용자가 쓰는 명령은 양쪽에서 같다

~~~sh
.venv/bin/python experiments/run_all.py prepare --scene office
.venv/bin/python experiments/run_all.py run --scene office
.venv/bin/python experiments/run_all.py status --scene office
.venv/bin/python experiments/run_all.py analyze --scene office
~~~

서버에서는 장면 이름만 바꾼다. 서버 준비 후 확장 전체는 다음과 같다.

~~~sh
.venv/bin/python experiments/run_all.py prepare --remaining
.venv/bin/python experiments/run_all.py run --remaining
.venv/bin/python experiments/run_all.py analyze
~~~

일반 사용자는 --device, --manifest, --cohort를 장비에 맞춰 바꿀 필요가 없다. CUDA가 있으면 CUDA, 그렇지 않으면 MPS를 자동 선택한다. 둘 다 없으면 오류를 내며 CPU로 조용히 전환하지 않는다. 확장 장면은 CUDA에서만 실행하며, 서버가 없다고 로컬 MPS에 배정하지 않는다.

--remaining은 처음 정한 office 외 11개 확장 후보다. 좋은 결과만 선택하는 기능이 아니다. 서버 확장 전에 office 초기 실행의 기록과 연구 잠금을 전달해야 한다.

현재 상태에서는 prepare/run/analyze의 승인 검사가 작업을 막는다. status는 GPU나 결과 폴더를 만들지 않고 읽기 전용으로 준비 상태를 보여준다.

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

실행 그룹별 cohort_configuration.json은 GPU·호스트·플랫폼·설치 패키지·실행 환경을 기록하고 잠근다. Mac과 CUDA 서버의 환경이 bit 단위로 같다고 주장하지 않는다. scene_assignments.json은 각 장면이 처음 배정된 실행 그룹을 기록해, 결과를 본 뒤 장치가 다른 값으로 덮어쓰거나 한 장면의 ADD/DROP을 플랫폼 사이에서 섞지 못하게 한다.

기본 구조는 양쪽에서 같다. runtime ID는 내부에서 자동 결정한다.

~~~text
data/experiments/eth3d_restart_v1/<scene>/manifest.json
results/eth3d_restart_v1/study_configuration.json
results/eth3d_restart_v1/scene_assignments.json
results/eth3d_restart_v1/executions/<runtime ID>/<scene>/seed_2026/
results/eth3d_restart_v1/analysis/report.md
~~~

office의 로컬 결과도 주 분석에 포함한다. 전체 보고서는 장치 표기를 유지하고 혼합 플랫폼 평균과 플랫폼별 요약을 구분한다. 같은 장면 내부의 모델·ADD/DROP은 같은 플랫폼에서 수행한다. MPS와 CUDA 차이는 연구 결론에 영향을 주지 않는다고 가정하며 별도 장치 교차 실험은 하지 않는다. 재실행 변동도 측정하지 않는다.

## 승인 후 완성하고 검증할 부분

[study_v1.json](study_v1.json)의 status는 awaiting_approval, implementation_ready는 false다.

공식 깊이 GT의 왜곡 좌표 연결, 사진별 CAMERA_ID 처리, 두 모델의 공통 평가 격자, GT 기반 사진 선정과 최종 다중 장면 분석은 승인 문서대로 구현·검증한 뒤 준비 상태를 바꾼다. prepare_study.py와 analyze_study.py가 그 공통 구현이 될 예정이며 아직 구현 완료를 주장하지 않는다.

데이터 준비는 승인 후에, 모델 실행은 승인과 구현 검증 이후에 가능하다. 가속 실행부가 준비됐다는 것과 새 실험의 전체 과학적 구현이 완료됐다는 것을 구분한다.

옛 단일 카메라·단순 laser raster 전처리와 일회성 office 분석 CLI는 퇴역 처리했다. 순수 함수는 기존 검증에 남아 있지만 이 CLI로 옛 실험을 다시 만들 수 없다. 내부 파라미터가 여러 개면 옛 파서는 첫 카메라를 복사하는 대신 오류를 낸다.

## 서버에 한 번 설정할 내용

동일 코드, 승인 문서, 가중치, 준비 RGB/GT snapshot, office의 결과·연구 잠금을 전달한다. Mac 가상환경을 Linux에 복사하지 않고 해당 CUDA 환경에 패키지를 설치한다. 실제 사용 버전은 실행 그룹별로 기록한다.

기본 경로는 저장소 상대 경로다. 필요한 경우 --input-root와 --results-root를 양쪽에서 같은 방식으로 지정할 수 있다. --seed와 --device도 전문가용 옵션으로 동일하게 제공하지만 일반 실행에 필수는 아니다.

현재 통일한 것은 Mac과 CUDA 서버에서 같은 프로그램을 사용하는 인터페이스다. 실제 서버 주소·전송 설정과 CUDA 실기기 검증은 서버 준비 후 연결한다. SSH 연결이 이미 구성됐다고 주장하지 않는다.

## 진행 상태와 재개

batch_status.json은 각 장면·seed 아래에 저장한다. 계획 조건 수는 manifest의 실제 단계 수와 5·10·15장 집합으로 계산한다. 최초 계획은 office 726회·전체 최대 8,712회 재구성이며 복구 비용은 별도다.

SIGINT·SIGTERM은 현재 재구성이 끝난 뒤 새 작업을 시작하지 않도록 요청한다. worker가 실패 코드로 종료하면 다음 모델을 자동 실행하지 않는다.

complete는 계획 비교가 모두 평가되고 모든 지표가 유효한 경우, complete_with_failures는 모두 평가했지만 일부 지표가 실패한 경우다. 수정 계획은 같은 조건의 자동 시도를 최초 포함 최대 3회로 정하고, 계속 실패하면 중단·원인 수정·검증·영향 조건 재계산을 수행한다. 첫 유효 결과를 사용하며 모든 실패·복구 이력을 보존한다. 미해결 조건을 숨기거나 정상값으로 조작하지 않는다. 이 복구 정책은 승인 후 worker·재개 구현에 반영할 항목이며 현재 구현 완료를 뜻하지 않는다.

## 자료와 검증

- [전체 계획](../EXPERIMENT_PLAN.md)
- [기계 판독 설정](study_v1.json)
- [공식 archive의 장면별 metadata](dataset_inventory.json)
- [가중치 revision과 SHA-256](weights.json)

이전 결과·cache는 작업 트리에서 삭제했고 원본 데이터와 모델 가중치는 유지했다. Git 과거 이력은 재작성하지 않았다.

~~~sh
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/run_all.py --help
.venv/bin/python experiments/run_all.py status --scene office
~~~

시험은 가상의 배열·파일·worker와 로컬 보정 metadata를 사용하며 실제 사전학습 모델 추론을 하지 않는다. 같은 사용자 명령의 MPS/CUDA dispatch, 승인 전 실행 차단, 단일 seed 전달, 1+3 worker, 장면·환경 잠금, 실패 보존을 검증한다.
