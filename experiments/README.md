# 통일된 실험 실행 코드

새 실험의 승인 대상은 [실험 계획과 분석 절차](../EXPERIMENT_PLAN.md)다. 장면 목록, 공통 전처리, 블러, 반복, 채점, 실패 처리와 최종 보고 형식을 이 문서에서 고정한다.

현재 상태는 **승인 대기 / 새 방법 구현 검증 미완료**다. 새 모델 추론은 시작하지 않았다. 이전 results 하위의 결과와 cache는 작업 트리에서 삭제했다.

## 기본 실행 경로

공식 진입점은 experiments/run_all.py다. run_parallel.py와 같은 parser·실행 함수를 사용하므로 순차 CPU 정렬 실행부와 가속 실행부가 서로 다른 기본값으로 갈라지지 않는다.

- VGGT: 한 worker.
- MASt3R: 세 worker, worker마다 CPU thread 8개.
- 추론과 정렬: 같은 MPS 또는 CUDA GPU.
- float32, highest 행렬곱 정밀도, TF32 비활성화.
- MASt3R complete symmetric graph, 300+300회 정렬.
- checkpoint와 native preprocessing은 고정된 공식 버전 사용.
- clean baseline과 결과 저장에 파일 잠금과 원자적 쓰기 적용.
- 신경망 pair cache는 같은 장면·같은 seed의 출력 폴더 안에서 공유.
- 기본 진입점에서 다른 실행의 cache를 가져오는 인수를 제공하지 않음.

모델 로더와 가속 정렬은 backends.py에 있다. GPU를 사용할 수 없다고 CPU로 자동 전환하지 않는다. run.py는 내부 worker와 명시적인 진단용 저수준 실행부이며 정식 실험을 수동 호출 여러 개로 대체하지 않는다.

## 승인과 준비 검사

[study_v1.json](study_v1.json)이 기계 판독 설정이다.

현재 status는 awaiting_approval, implementation_ready는 false다. 기본 진입점은 승인·준비 검사가 끝나기 전에 입력을 읽거나 GPU를 선택하거나 결과 폴더를 만들지 않는다.

승인 뒤에도 공식 깊이 GT 연결, 사진별 CAMERA_ID, 공통 평가 격자, 세 seed의 전체 반복, 다중 장면 분석을 구현·검증해야 한다. **가속 실행부가 정리됐다는 것과 새 과학적 프로토콜의 구현이 완료됐다는 것을 구분한다.** 계획의 방법을 바꾸지 않는 구현이 완료되면 문서 해시와 준비 상태를 고정한다. 방법 변경은 계획서 재승인 대상이다.

단일 카메라·단순 laser raster 참조에 맞춘 옛 prepare.py CLI와 과거 일회성 분석·후속 실행 CLI는 퇴역 처리했다. 순수 함수는 기존 검증에 남아 있지만 해당 CLI로 이전 실험을 다시 생성할 수 없다. 사진별 보정이 여러 개면 옛 단일 카메라 파서는 잘못 첫 카메라를 적용하는 대신 오류를 낸다.

## 승인 후 사용할 실행 형식

다음은 문법 예시다. 현재의 승인 대기 설정으로 실행하면 모델을 시작하지 않고 거부한다.

~~~sh
.venv/bin/python experiments/run_all.py \
  --protocol experiments/study_v1.json \
  --manifest data/experiments/eth3d_restart_v1/courtyard/manifest.json \
  --cohort results/eth3d_restart_v1_cuda \
  --device cuda \
  --seed 2026
~~~

MPS에서도 같은 형식을 사용한다. --device는 필수이며 mps 또는 cuda를 명시한다. 출력은 cohort/scene/seed_<seed>로 정해져 장면과 반복을 섞지 않는다. 다른 seed는 같은 cohort를 사용하면서 각각 별도 결과·cache를 만든다.

전체 장면·seed의 일정과 완료 검사는 승인 후 구현할 순회 실행부가 담당한다. 본실험에서는 사전에 잠근 후보·통과 목록 전체를 사용한다.

## 같은 실행 묶음의 잠금

cohort_configuration.json에 다음을 기록한다.

- 승인한 protocol과 MD의 해시
- 실제 코드 파일 해시
- 공식 vendor commit과 tracked 변경분 해시
- MASt3R 모델 설정 해시
- 장치, 호스트, 플랫폼, Python과 핵심 패키지 버전
- worker·thread·정밀도·seed 목록과 관련 실행 환경

다른 장면이나 seed를 시작할 때 이 잠금과 다르면 거부한다. 모델별 worker도 가중치·입력·환경 fingerprint를 검증한다. GPU 플랫폼을 바꾸려면 모든 대상 장면을 새 묶음에서 계산하며 이전 묶음과 합쳐 하나의 결과로 취급하지 않는다.

## 중단과 진행 상태

batch_status.json은 각 scene/seed 아래에 저장한다. 계획 조건 수는 manifest의 실제 블러 단계 수로 계산하며 office나 여섯 단계에 고정하지 않는다.

SIGINT와 SIGTERM은 중단 파일을 남겨 진행 중 재구성이 끝난 뒤 새 작업을 시작하지 않도록 요청한다. 완료한 결과는 보존한다. worker가 실패 코드로 종료하면 다음 모델 단계를 자동 실행하지 않는다.

complete는 모든 계획 비교가 평가됐고 모든 지표가 유효한 경우다. complete_with_failures는 모든 비교를 평가했지만 일부 지표는 실패한 경우다. 실패를 삭제하거나 성공할 때까지 다시 실행하는 정책은 새 계획에서 금지하며, 해당 보존·재개 규칙도 구현 준비 검사에 포함한다.

## 모델과 자료

모델 가중치는 [weights.json](weights.json)의 checkpoint revision·SHA-256으로 검증한다. 패키지의 기존 로컬 목록은 requirements-lock.txt에 있다. CUDA 서버에는 Mac 가상환경을 복사하지 않고 해당 플랫폼 환경을 별도로 만든 뒤 실행 묶음 전체의 버전을 고정한다.

공식 학습 자료의 목록은 [dataset_inventory.json](dataset_inventory.json)에 있다. 보정 자료의 scene별 사진 수와 여러 내부 파라미터 묶음을 직접 확인한 기록이다. 새 실험에는 공식 occlusion-aware DSLR depth 자료를 추가로 확보하고, 왜곡 좌표와 undistorted RGB를 맞추는 공통 전처리가 필요하다.

## 코드 검증

아래 명령은 가상의 배열·파일·worker와 로컬 보정 metadata를 이용한다. 사전학습 모델 추론을 하지 않는다.

~~~sh
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/run_all.py --help
~~~

시험에는 승인 전 실행 차단, 명시적 GPU 선택, 1+3 worker 실행, seed 전달, cohort 변경 거부, manifest 단계 수에 맞는 집계, 동시 쓰기, 실패 보존과 잘못된 다중 카메라 처리를 막는 검증이 포함된다.
