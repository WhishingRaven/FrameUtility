# 조교 제공 Amazon CUDA 서버 준비 가이드

작성 기준: 2026-10-25. 대상은 **서버를 준비하는 팀원**이다. 이 문서는 서버 접속, Python/CUDA 환경, 코드·가중치·자료 배치와 연결 확인까지 안내한다. 실험 방법은 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md), 실행 인터페이스는 [experiments/README.md](experiments/README.md)를 따른다.

현재 서버의 종류·주소·GPU·운영체제는 아직 확인되지 않았다. 아래는 **조교가 제공한 Linux + NVIDIA GPU 서버**를 사용하는 절차다. Amazon EC2일 가능성을 기준으로 설명하되, 실제로 SageMaker나 브라우저 전용 실습 환경이면 접속 방법부터 확인한다. 인스턴스를 새로 구매하거나 새 GPU 서버를 생성하는 절차는 포함하지 않는다.

**현재 프로젝트 상태:** 실험 계획은 승인 대기이며 `implementation_ready`는 `false`다. 준비 자료와 공식 사진 선정 결과는 이미 존재하지만 최종 입력·GT 채점 통합은 작업 중이다. 인프라 준비와 작은 CUDA 연산 검증은 먼저 할 수 있다. 본실험은 담당자의 승인·구현 검증 이후에 시작한다. `analyze` 구현은 결과 확보 뒤 진행하며 서버 준비의 선행 조건이 아니다.

## 0. run에 필요한 최소 구성

이 문서는 **이미지는 별도의 분할 묶음으로 전송한다는 계획**을 전제로 한다. 그 이미지 묶음과 별도로 보내는 **실행 코드 묶음은 `experiments/`와 `EXPERIMENT_PLAN.md` 두 항목**이다. 모델 소스·가중치·Python/CUDA 환경은 서버 최초 설정 때 한 번 준비한다.

아래 표는 최종적으로 서버에 있어야 할 구성이다. 이미지 묶음마다 이 전체 구성을 다시 보내라는 뜻이 아니다.

| 구분 | 전송·설치 시점 |
|---|---|
| 실행 코드: experiments/ + EXPERIMENT_PLAN.md | 본실험 시작 전 한 번 동기화 |
| 모델 소스·가중치·Python/CUDA | 서버 최초 설정 때 한 번 준비 |
| 이미지·GT·manifest 분할 묶음 | 별도 계획대로 나눠 전송. 서버에서 원래 준비 경로로 복원 |
| 실제 run | 해당 장면의 모든 입력이 수신·검증된 뒤 실행. 다른 장면의 전송 완료를 기다릴 필요는 없음 |

| 항목 | run에서 필요한 이유 | 준비 방법 |
|---|---|---|
| `experiments/` | 실행 코드, 프로토콜 JSON, 가중치 목록, 추론 코드 해시 | 로컬과 같은 디렉토리 내용 전송 |
| `EXPERIMENT_PLAN.md` | 승인한 문서의 SHA-256 대조 | 로컬과 같은 파일 전송 |
| `vendor/vggt/`, `vendor/mast3r/`와 DUSt3R·CroCo 하위 모듈 | 모델·전처리·정렬 코드, 버전·변경분 확인 | 서버에서 고정 커밋으로 Git clone하거나 완전한 checkout 전송 |
| `checkpoints/` | `vggt-1b.pt`, `mast3r.safetensors`, `mast3r-config.json` | 전송 후 가중치 크기·SHA-256 확인 |
| `data/experiments/eth3d_multi_view/`의 실행할 장면과 공통 메타데이터 | 최종 manifest, 모든 입력 RGB 변형, 채점용 깊이·GT·마스크 등 참조 파일 | 완성된 입력 snapshot 전송 |
| 서버의 `.venv/`와 NVIDIA 드라이버 | Python 의존성과 CUDA 실행 환경 | Linux 서버에서 새로 설치. Mac의 .venv는 복사하지 않음 |

기본 배치는 다음과 같다. 상위 폴더 이름은 달라도 되지만 이 상대 구조는 유지한다.

```text
FrameUtility/
├── EXPERIMENT_PLAN.md
├── experiments/
├── vendor/
│   ├── vggt/
│   └── mast3r/                 # dust3r와 croco 하위 모듈 포함
├── checkpoints/
│   ├── vggt-1b.pt
│   ├── mast3r.safetensors
│   └── mast3r-config.json
├── data/experiments/eth3d_multi_view/
│   ├── official_selection.json
│   └── office/                 # 또는 실행할 다른 장면
│       ├── manifest.json
│       └── ...                 # manifest가 참조하는 모든 입력·GT 파일
└── .venv/                      # 서버에서 생성
```

FrameUtility 자체의 루트 `.git`, `Proposal.pdf`, 일반 README, 과거 실험 결과와 분석 모듈은 `run`의 필수 입력이 아니다. `results/`도 신규 실행 때 자동 생성한다. 같은 작업을 재개할 때는 해당 작업의 기존 결과·실행 메타데이터를 보존해야 한다.

반면 **vendor의 Git 정보는 현재 코드에 필수**다. 실행기가 `git rev-parse`와 `git diff`로 모델 버전을 확인하므로 모델 소스를 ZIP으로만 풀거나 `.git`과 하위 모듈 메타데이터를 제거하면 안 된다. 서버에서 5절대로 clone하는 방법을 권장한다.

실행 전용 서버에는 입력 변환·선정에 썼던 `data/eth3d/high_res_multi_view/`의 원본·보정·GT archive 전체를 기본 전송하지 않는다. 이 생략은 **최종 manifest의 모든 실행 참조 파일이 준비 snapshot에 포함돼 있을 때** 가능하다. 외부 파일을 가리키는 경로나 심볼릭 링크가 남아 있으면 그 대상도 필요하다. 현재 worker는 manifest의 모든 frame에 선언된 depth와 RGB variant의 파일·해시를 확인하므로 사용하지 않을 것 같은 frame을 임의로 빼서 전송하지 않는다.

현재 정리본의 12개 장면 manifest에서 RGB·깊이 참조 2,340건은 모두 준비 루트 내부에 있고 누락·심볼릭 링크가 없는 것을 확인했다. 이는 경로·존재 확인이다. **현재 정리본에는 최종 프로토콜 필드·GT 통합이 남아 있어 곧바로 run 가능한 완성 입력이라는 뜻은 아니다.** 최종 입력을 확정한 뒤 추가 GT·마스크를 포함해 다시 확인한다.

## 1. 조교에게 먼저 받을 정보

아래 표를 채운 뒤 시작하면 접속 오류와 환경 재설치를 줄일 수 있다.

| 항목 | 확인할 내용 |
|---|---|
| 제공 방식 | EC2 Linux 서버인지, SageMaker인지, 수업 실습 계정인지 |
| 접속 주소 | public DNS/IP 또는 VPN·bastion을 통한 내부 주소 |
| 로그인 사용자 | 조교가 지정한 계정. Ubuntu의 `ubuntu`, Amazon Linux의 `ec2-user` 등은 예시일 뿐 확정값이 아님 |
| 인증 | 개인 SSH 공개키 등록 또는 제공된 키 파일, 필요한 MFA·AWS 로그인 방식 |
| 접속 경로 | SSH 포트, 교내 VPN, bastion, SSM 필요 여부 |
| GPU | NVIDIA GPU 모델·개수·각 GPU의 VRAM, 팀에 할당된 GPU 번호 |
| CPU·RAM | vCPU 수, 시스템 메모리, 다른 팀과 자원 공유 여부 |
| OS·기존 환경 | Linux 배포판, NVIDIA 드라이버, DLAMI 여부, 제공된 Python/Conda 환경 |
| 저장소 | 사용 가능한 프로젝트 경로·용량, EBS 등 영구 저장소인지 임시 디스크인지 |
| 설치 권한 | sudo 사용 가능 여부, 필요한 시스템 패키지의 설치 담당자 |
| 네트워크 | GitHub·PyTorch wheel 저장소 접근과 로컬에서의 자료 전송 가능 여부 |
| 운영 제한 | 사용 기간·예산, 자동 중지·세션 만료·Spot 중단 가능성, 종료 담당자 |

우리 실행기는 일반 SSH 클라이언트로 서버에 명령을 전달한다. AWS 콘솔의 브라우저 터미널만 접속되는 상태라면 로컬의 `--target cuda-server`를 바로 쓸 수 없다. SSH 별칭으로 접속할 경로를 마련해야 한다. SSM 전용 환경에서는 조교가 허용한 AWS CLI·Session Manager 구성에 따라 SSH ProxyCommand를 설정할 수 있다. [AWS 접속 방식](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html), [SSM을 통한 SSH](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-getting-started-enable-ssh-connections.html).

## 2. 로컬 컴퓨터에서 SSH 연결을 만든다

이 문서의 **로컬**은 명령을 입력하는 개인 컴퓨터, **서버**는 조교가 제공한 Amazon 컴퓨터다. 이후 코드 블록마다 실행 위치를 구분한다.

로컬 `~/.ssh/config`에 다음 형태로 등록한다. `HostName`, `User`, `IdentityFile`은 실제 제공값으로 바꾼다. 예시 별칭 `frameutility-gpu`는 이후 명령과 JSON에서 일관되게 사용한다.

```sshconfig
Host frameutility-gpu
    HostName EC2_PUBLIC_DNS_OR_IP
    User ubuntu
    IdentityFile ~/.ssh/frameutility-course.pem
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

키 파일을 제공받은 경우의 로컬 명령이다. 개인 키는 로컬에 두고 서버·Git·결과 폴더로 복사하지 않는다. 각 팀원이 별도 키를 쓰면 자기 컴퓨터의 SSH 설정만 달라지며 프로젝트 명령은 같다.

```bash
# 로컬: 실제 키 파일명으로 바꾼 뒤 실행
chmod 400 ~/.ssh/frameutility-course.pem
ssh frameutility-gpu
```

첫 연결 때 서버의 호스트 키를 확인한다. 조교가 지문을 제공했다면 대조한다. 접속 후 `whoami`와 `pwd`로 사용자·시작 경로를 확인하고 `exit`로 돌아온다. SSH 접속에는 올바른 사용자·키와 허용된 네트워크 경로가 필요하며, 직접 접속 방식에서는 연결하는 컴퓨터의 IP에 대한 보안 그룹 규칙도 확인한다. [AWS SSH 연결 안내](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-linux-inst-ssh.html), [SSH 연결 전제 조건](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-to-linux-instance.html).

그다음 **입력 창을 띄우지 않는 접속**이 되는지 확인한다. 우리 프로그램도 `BatchMode=yes`를 사용하므로 이 검사가 중요하다.

```bash
# 로컬
ssh -T -o BatchMode=yes -o ConnectTimeout=10 frameutility-gpu 'whoami'
```

암호나 인증 입력이 없어서 실패한다면 SSH 키·agent 또는 조교가 제공한 인증 절차를 먼저 정리한다. 브라우저에서 로그인된 것만으로 이 검사를 통과했다고 판단하지 않는다.

## 3. 서버의 GPU와 저장 경로를 확인한다

이후 경로 예시는 `/home/ubuntu/projects/FrameUtility`다. 실제로는 **조교가 지정한 쓰기 가능한 영구 저장 경로**로 모두 바꾼다. 다른 경로를 사용했다면 9절 JSON의 `project_dir`도 같은 절대 경로로 설정한다.

```bash
# 서버
mkdir -p /home/ubuntu/projects/FrameUtility
cd /home/ubuntu/projects/FrameUtility
mkdir -p .cache/server_setup

cat /etc/os-release
uname -m
nproc
free -h
df -h .
nvidia-smi
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
nvidia-smi > .cache/server_setup/nvidia-smi.txt
```

`nvidia-smi`에서 실제 NVIDIA GPU와 드라이버가 보여야 한다. 명령이 없거나 GPU가 없으면 조교에게 서버·드라이버 상태를 확인한다. AWS Inferentia/Trainium 등 다른 가속기만 있는 환경을 CUDA 서버로 간주하지 않는다.

DLAMI는 종류에 따라 드라이버·CUDA 또는 PyTorch 환경이 미리 제공될 수 있다. 먼저 확인하고 기존 공유 시스템 드라이버를 임의로 재설치하지 않는다. 서버에 CUDA 관련 파일이 있다는 것만으로 GPU가 실제 제공됐다고 판단하지 않는다. [AWS DLAMI 구성](https://docs.aws.amazon.com/dlami/latest/devguide/features.html).

현재 실행 설정은 VGGT 1 worker, MASt3R 3 worker이며 worker마다 CPU thread 8개다. MASt3R의 세 worker는 **한 GPU를 공유**하며 세 GPU로 자동 분산되지 않는다. 다중 GPU 서버에서는 조교가 배정한 GPU를 사용하도록 실행 환경을 정한다. 모델·활성값·정렬 상태의 VRAM 사용량은 실제 서버에서 아직 측정하지 않았으므로 특정 VRAM 용량이면 반드시 된다고 보장하지 않는다.

Ubuntu에서 필요한 도구가 빠져 있고 sudo 설치가 허용된 경우에만 다음을 사용한다. 다른 Linux 배포판은 제공 환경에 맞는 패키지 관리자를 사용한다.

```bash
# 서버 / Ubuntu / 시스템 패키지 설치가 허용된 경우
sudo apt-get update
sudo apt-get install -y git rsync tmux build-essential python3-venv libgl1 libglib2.0-0
```

## 4. 실행 코드와 승인 문서를 서버에 배치한다

**이미지를 나눠 보내는 작업과 별개의 코드 전송 단계**다. 전송물은 `experiments/`와 `EXPERIMENT_PLAN.md`로 한정한다. 가중치·모델 소스·Python 환경이 이미 준비됐다면 다시 전송하지 않는다. 아래 절차는 현재 로컬 작업 트리의 코드를 전송하므로 미커밋 수정도 포함한다. 팀이 사용할 코드 snapshot을 먼저 정하고 본실험 시작 전에 동기화한다.

```bash
# 로컬: FrameUtility 저장소 루트에서 실행
ssh frameutility-gpu 'mkdir -p /home/ubuntu/projects/FrameUtility/experiments'

rsync -av --progress \
  --exclude='__pycache__/' --exclude='cuda_server.json' \
  experiments/ frameutility-gpu:/home/ubuntu/projects/FrameUtility/experiments/
rsync -av --progress EXPERIMENT_PLAN.md \
  frameutility-gpu:/home/ubuntu/projects/FrameUtility/
```

FrameUtility 루트의 `.git` 이력은 전송 필수가 아니다. 커밋·미커밋 변경 이력은 로컬에서 기록하고 서버에서는 8절의 추론 소스 해시를 대조한다. 처음에는 빈 전용 디렉토리를 사용하고 기존 타 프로젝트 위에 복사하지 않는다. `--delete`는 사용하지 않는다. 전체 저장소를 clone하는 방식도 가능하지만 실행에 불필요한 문서와 Git 이력까지 반드시 전송할 필요는 없다.

```bash
# 서버
cd /home/ubuntu/projects/FrameUtility
ls experiments/run_all.py experiments/study_v1.json experiments/weights.json EXPERIMENT_PLAN.md
```

## 5. 모델 소스를 같은 버전으로 준비한다

작성 시점 로컬에서 확인한 버전은 다음과 같다. 본실험 코드를 최종 확정할 때 변경됐다면 실험 담당자가 제공하는 최신 고정 목록과 맞춘다.

| 소스 | 현재 고정 커밋 |
|---|---|
| VGGT | `a288dd0f14786c93483e45524328726ab7b1b4ce` |
| MASt3R | `f5209afc300cec36239a7ac992263f36847bbba0` |
| DUSt3R | `3cc8c88c413bb9e34c41db0e0eef99c2ee010b12` |
| CroCo | `d7de0705845239092414480bd829228723bf20de` |

```bash
# 서버: 처음 설치할 때. 해당 폴더가 이미 있으면 재clone하지 않고 상태 확인
cd /home/ubuntu/projects/FrameUtility
mkdir -p vendor

git clone https://github.com/facebookresearch/vggt.git vendor/vggt
git -C vendor/vggt checkout --detach a288dd0f14786c93483e45524328726ab7b1b4ce

git clone https://github.com/naver/mast3r.git vendor/mast3r
git -C vendor/mast3r checkout --detach f5209afc300cec36239a7ac992263f36847bbba0
git -C vendor/mast3r submodule update --init --recursive

git -C vendor/vggt rev-parse HEAD
git -C vendor/mast3r rev-parse HEAD
git -C vendor/mast3r submodule status --recursive
```

상위 MASt3R 커밋이 가리키는 DUSt3R·CroCo 커밋도 표와 일치하는지 확인한다. 본실험 시작 뒤 `git pull`로 모델 소스를 갱신하지 않는다. 로컬 vendor에 별도 수정이 생기면 커밋 번호만으로는 같다고 볼 수 없으므로 변경분도 함께 전달하고 확인한다.

## 6. Python과 CUDA용 PyTorch를 설치한다

### 6.1 프로젝트 전용 환경

기준 Python은 로컬과 같은 **3.11 계열**이다. 서버에 `python3.11`이 있으면 다음과 같이 만든다. Mac의 `.venv` 디렉토리를 Linux로 복사하지 않는다.

```bash
# 서버
cd /home/ubuntu/projects/FrameUtility
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

`python3.11`이 없고 조교가 Conda를 제공했다면 위 환경 생성 대신 다음을 사용할 수 있다. 둘 중 한 방법만 선택한다. 제공되지 않은 Python이나 설치 권한 문제는 조교에게 확인한다.

```bash
# 서버 / Conda가 제공된 경우의 대안
cd /home/ubuntu/projects/FrameUtility
conda create -y --prefix "$PWD/.venv" python=3.11 pip
```

중요한 것은 **`.venv/bin/python`을 직접 실행해도 모든 의존성이 보이는 것**이다. `run_all.py`의 SSH 호출은 대화형 `conda activate`에 의존하지 않는다.

### 6.2 Torch와 일반 패키지를 나눠 설치

[requirements-lock.txt](experiments/requirements-lock.txt)는 로컬 환경의 버전 목록이다. 작성 시점에는 `torch==2.14.1`, `torchvision==0.29.1`이 기록돼 있다. **이 버전의 Linux CUDA wheel이 서버 GPU·드라이버에서 제공되고 동작하는지는 아직 검증하지 않았다.** 로컬 lock 전체를 일반 pip 명령으로 설치한 뒤 CUDA 설치가 됐다고 가정하지 않는다.

먼저 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)와 [버전별 설치 명령](https://pytorch.org/get-started/previous-versions/)에서 Linux·Pip·CUDA 조합을 확인한다. 아래 입력에는 공식 명령의 `--index-url` 뒤 URL을 넣는다. CUDA 빌드 선택에는 GPU 아키텍처와 드라이버 호환성도 확인해야 한다. [NVIDIA CUDA 호환성 설명](https://docs.nvidia.com/deploy/cuda-compatibility/why-cuda-compatibility.html).

```bash
# 서버: 프로젝트 루트. 먼저 lock을 두 설치 목록으로 나눔
mkdir -p .cache/server_setup
.venv/bin/python - <<'PY'
from pathlib import Path

lines = [x.strip() for x in Path('experiments/requirements-lock.txt').read_text().splitlines()
         if x.strip() and not x.lstrip().startswith('#')]
torch = [x for x in lines if x.split('==', 1)[0].lower() in {'torch', 'torchvision'}]
base = [x for x in lines if x not in torch]
Path('.cache/server_setup/torch-pins.txt').write_text('\n'.join(torch) + '\n')
Path('.cache/server_setup/base-pins.txt').write_text('\n'.join(base) + '\n')
print('로컬 기준 Torch 조합:', torch)
PY

read -r -p '공식 명령에서 확인한 CUDA wheel index URL: ' FRAMEUTILITY_TORCH_INDEX
.venv/bin/python -m pip install \
  -r .cache/server_setup/torch-pins.txt \
  --index-url "$FRAMEUTILITY_TORCH_INDEX"
```

고정 버전 wheel을 찾지 못하면 여기서 설치 조합을 다시 정한다. 조교의 지원 환경 또는 공식 문서에 있는 **서로 호환되는 Torch·TorchVision 조합**을 명시적으로 선택하고 실제 설치 명령·버전을 팀에 공유한다. CPU wheel이나 임의의 다른 버전으로 몰래 넘어가지 않는다. 최초 본실험 전에 조합을 확정하며 이후 서버 실행 중에는 패키지를 갱신하지 않는다.

CUDA용 Torch를 설치한 다음 나머지 패키지를 설치한다. 다음 constraints는 일반 패키지 설치가 선택한 Torch를 바꾸지 못하게 한다.

```bash
# 서버: CUDA용 Torch 설치가 성공한 뒤
.venv/bin/python - <<'PY'
from pathlib import Path
import importlib.metadata as m

text = '\n'.join(f'{name}=={m.version(name)}' for name in ('torch', 'torchvision'))
Path('.cache/server_setup/torch-constraints.txt').write_text(text + '\n')
PY

.venv/bin/python -m pip install \
  -r .cache/server_setup/base-pins.txt \
  -c .cache/server_setup/torch-constraints.txt
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > .cache/server_setup/requirements-cuda.freeze.txt
```

다른 고정 패키지의 Linux wheel도 없으면 패키지명·버전·오류를 기록해 본실험 전에 해결한다. vendor의 requirements를 추가로 무조건 설치하면 Torch 등을 다른 버전으로 바꿀 수 있으므로 이 프로젝트의 설치 목록과 충돌을 먼저 확인한다. 현재 코드는 `opencv-python` 배포 이름을 조회하므로 `opencv-python-headless`만 설치하는 것으로 대체하지 않는다.

### 6.3 작은 실제 GPU 연산으로 확인

아래는 가중치·실험 이미지를 사용하지 않는 인프라 검사다. CUDA 인식뿐 아니라 작은 행렬곱과 미분까지 동작하는지 확인한다.

```bash
# 서버
.venv/bin/python - <<'PY'
import platform
import torch
import torchvision

print('Python:', platform.python_version())
print('Torch:', torch.__version__, 'TorchVision:', torchvision.__version__)
print('Torch CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), '이 Python 환경에서 CUDA를 사용할 수 없음'
print('GPU:', torch.cuda.get_device_name(0))
print('지원 아키텍처:', torch.cuda.get_arch_list())
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
x = torch.randn(32, 32, device='cuda', dtype=torch.float32, requires_grad=True)
(x @ x).square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all().item()
print('CUDA 행렬곱·미분 확인 완료')
PY
```

가능하면 같은 검사를 로컬에서 `ssh -T -o BatchMode=yes frameutility-gpu`로 접속한 비대화형 명령에서도 확인한다. 터미널의 활성 환경에서만 성공하고 `.venv/bin/python` 직접 실행에서 실패하면 설정이 아직 끝난 것이 아니다.

### 6.4 선택 사항: MASt3R의 RoPE CUDA 확장

MASt3R 공식 문서는 RoPE CUDA 커널의 선택적 빌드를 안내한다. 사용하려면 서버의 CUDA 컴파일러·현재 Torch와 맞춰 **본실험 전에** 빌드하고 동작을 검증한다. Mac에서 만든 바이너리는 복사하지 않는다. 확장 사용 여부와 빌드 환경도 기록한다. [MASt3R 설치 안내](https://github.com/naver/mast3r#installation).

```bash
# 서버: nvcc가 제공되고 빌드할 것으로 정한 경우에만
nvcc --version
cd /home/ubuntu/projects/FrameUtility/vendor/mast3r/dust3r/croco/models/curope
/home/ubuntu/projects/FrameUtility/.venv/bin/python setup.py build_ext --inplace
cd /home/ubuntu/projects/FrameUtility
```

## 7. 가중치와 완성된 실행 입력을 전송한다

자료는 실험 담당자가 준비하고 있다. 서버 담당자는 임의로 사진을 재선정하거나 블러를 다시 생성하지 않고 **전달받은 확정 snapshot**을 옮긴다. 아직 생성 중인 파일은 완료된 뒤 동기화·해시 확인을 다시 한다.

### 7.1 준비가 끝난 자료로 run만 실행하는 경우

이미지 분할 전송과 코드 전송은 독립적으로 진행한다. 각 이미지 묶음은 서버의 같은 `data/experiments/eth3d_multi_view/` 아래에 모이며, 분할 번호를 run의 입력 경로에 추가하지 않는다. 압축파일을 여러 조각으로 나누는 방식을 쓰면 조각을 모두 수신·검증한 뒤 원래 파일과 경로를 복원한다.

분할 묶음은 RGB뿐 아니라 해당 장면의 최종 manifest와 그 manifest가 참조하는 깊이·GT·마스크도 빠짐없이 전달해야 한다. 한 장면의 필요한 파일과 해시가 모두 확인되면 그 장면의 run을 시작할 수 있다. 일부 조각만 있는 상태에서 실행하거나 같은 장면의 입력을 계산 도중 덮어쓰지 않는다.

아래는 장면별 분할 전송을 rsync로 수행하는 예시다. 이미 다른 방식으로 나눠 전달한 이미지 파일을 다시 보낼 필요는 없으며, 최종 경로와 해시를 확인하면 된다. 가중치는 최초 준비 때 한 번만 전송한다.

작성 시점 로컬 `du -sh` 참고값이다. 최종 자료가 추가되면 다시 측정한다.

| 전송 경로 | 내용 | 현재 크기 |
|---|---|---:|
| `data/experiments/eth3d_multi_view/` | 사진별 PNG·참고 깊이·manifest·공식 선정 결과 | 약 2.0G |
| `checkpoints/` | 두 모델 가중치와 MASt3R 설정 JSON | 약 7.3G |

현재 정리본과 가중치의 합은 약 9G대다. 이는 최종 입력 크기의 확정값이 아니며 공식 GT·마스크 파일이 추가되면 늘 수 있다. Python 환경·결과·pair cache 공간도 별도다. `data/` 전체 약 **396G**나 원본 고해상도 자료 약 **36G**를 run용 기본 전송에 넣지 않는다. 여유 공간은 최종 입력과 예상 최대 임시 사용량을 합쳐 확인하며, 위 수치를 VRAM이나 전체 실행 디스크의 보장값으로 해석하지 않는다.

```bash
# 서버: 수신 디렉토리 준비
mkdir -p /home/ubuntu/projects/FrameUtility/data/experiments/eth3d_multi_view
mkdir -p /home/ubuntu/projects/FrameUtility/checkpoints
```

```bash
# 로컬: 공통 메타데이터와 office 이미지 묶음의 예시
rsync -av --progress data/experiments/eth3d_multi_view/*.json \
  frameutility-gpu:/home/ubuntu/projects/FrameUtility/data/experiments/eth3d_multi_view/
rsync -av --progress --partial data/experiments/eth3d_multi_view/office/ \
  frameutility-gpu:/home/ubuntu/projects/FrameUtility/data/experiments/eth3d_multi_view/office/

# 최초 한 번. 서버에 같은 가중치가 이미 있으면 생략
rsync -av --progress --partial checkpoints/ \
  frameutility-gpu:/home/ubuntu/projects/FrameUtility/checkpoints/
```

`official_selection.json`은 `data/experiments/eth3d_multi_view/`에 있으며 manifest와 프로토콜에서 이 경로와 해시를 참조한다. 파일명·카메라 ID·사진 순서·블러 단계·GT를 서버 사정에 맞춰 변경하지 않는다.

묶음 수신 뒤 해당 장면의 checksum 차이가 있는지 읽기 전용으로 확인한다. 다른 장면도 같은 절차로 나눠 전송·확인한다. 공식 선정 결과 등 공통 메타데이터의 해시도 대조한다.

```bash
# 로컬: -n은 전송하지 않는 확인 모드, -c는 파일 내용 checksum 비교
rsync -avnc --itemize-changes data/experiments/eth3d_multi_view/office/ \
  frameutility-gpu:/home/ubuntu/projects/FrameUtility/data/experiments/eth3d_multi_view/office/
```

서버에서는 두 대형 가중치의 바이트 수와 SHA-256을 [weights.json](experiments/weights.json)에 대조한다. `mast3r-config.json`도 전송됐는지 함께 확인한다.

```bash
# 서버: 프로젝트 루트
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib
import json

for name, meta in json.loads(Path('experiments/weights.json').read_text()).items():
    p = Path(meta['file'])
    assert p.stat().st_size == meta['bytes'], f'{name}: 크기 불일치'
    digest = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == meta['sha256'], f'{name}: SHA-256 불일치'
    print(name, '가중치 확인 완료')
assert Path('checkpoints/mast3r-config.json').is_file()
print('MASt3R config 확인 완료')
PY
```

### 7.2 서버에서 prepare·입력 재생성도 수행하는 경우에만 추가

서버에서 원본 사진을 다시 변환하거나 공식 깊이 GT 대응·선정 검증을 수행한다면 준비 절차가 사용하는 원본·보정·공식 GT 자료가 추가로 필요하다. 그 경우에만 다음 전송을 추가한다. 준비된 입력만 사용하는 run의 기본 전송 명령이 아니다.

```bash
# 로컬: 서버에서도 자료 준비를 수행하기로 정한 경우에만
ssh frameutility-gpu 'mkdir -p /home/ubuntu/projects/FrameUtility/data/eth3d'
rsync -av --progress --partial data/eth3d/high_res_multi_view/ \
  frameutility-gpu:/home/ubuntu/projects/FrameUtility/data/eth3d/high_res_multi_view/
```

서버에 vendor·가중치·완성 입력·Python 환경이 이미 동일하게 준비돼 있으면 다음 실행을 위해 이들을 재전송하지 않는다. 코드가 바뀌면 본실험 시작 전에 `experiments/`를 맞추고, 승인 문서가 바뀌면 `EXPERIMENT_PLAN.md`도 함께 맞춘다. 실행 중인 코드 위에 새 코드를 덮어쓰지 않는다.

## 8. 모델을 돌리기 전 설치 상태를 확인한다

```bash
# 서버: 모델 생성·추론 없이 import 경로 확인
.venv/bin/python - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path('vendor/vggt').resolve()))
from vggt.models.vggt import VGGT
sys.path.insert(0, str(Path('vendor/mast3r').resolve()))
from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
print('VGGT·MASt3R·정렬 모듈 import 확인 완료')
PY

.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/run_all.py --help
.venv/bin/python experiments/run_all.py status --scene office
```

단위 시험은 가상 자료와 가상 worker를 사용하며 본실험 추론을 하지 않는다. 옛 `data/prepared/eth3d/office` 또는 옛 office manifest를 요구하는 일부 회귀 시험은 그 자료를 전송하지 않았다면 skip될 수 있다. skip 이름·이유는 기록하고, 새 인터페이스·경로·실행 잠금 시험의 실패를 skip으로 대신하지 않는다. 이 시험 통과가 최종 GT 채점 통합이나 CUDA 모델 전체 실행을 보증하지는 않는다.

최종 프로젝트 코드의 커밋·미커밋 변경 이력은 로컬에서 남긴다. 서버에 루트 `.git`을 전송하지 않은 최소 구성에서도 다음 추론 소스 해시 대조는 가능하다. vendor는 Git checkout 정보를 별도로 유지하고, 실험 직전에는 추론 소스 해시가 같은지 확인한다.

```bash
# 로컬과 서버에서 각각 같은 명령 실행 후 출력 비교
.venv/bin/python - <<'PY'
import json
import sys
sys.path.insert(0, 'experiments')
from run_parallel import inference_source_hashes
print(json.dumps(inference_source_hashes(), indent=2, sort_keys=True))
PY
```

기존 manifest의 `staging_sets`나 공식 선정 통과 기록만 보고 추론 입력 준비가 끝났다고 판단하지 않는다. 최종 GT·공통 평가 격자·프로토콜 검증은 실험 담당자가 완료한다. 서버 담당자는 `study_v1.json`의 승인·준비 상태를 임의로 `approved`·`true`로 바꾸지 않는다.

## 9. 로컬에 CUDA 서버 profile을 등록한다

로컬에서 [cuda_server.example.json](experiments/cuda_server.example.json)을 복사하고 실제 서버 경로로 수정한다. 이 파일은 Git에서 제외된다. host에는 2절의 SSH 별칭을 넣고 개인 키는 SSH 설정에 둔다.

```bash
# 로컬
cp experiments/cuda_server.example.json experiments/cuda_server.json
```

```json
{
  "host": "frameutility-gpu",
  "project_dir": "/home/ubuntu/projects/FrameUtility",
  "python": ".venv/bin/python",
  "port": 22
}
```

```bash
# 로컬: 실제 원격 호출 확인. GPU 작업을 시작하지 않는 읽기 전용 명령
.venv/bin/python experiments/run_all.py status --scene office --target cuda-server
```

현재 정상 예상은 `approval_status: awaiting_approval`, office `not_started`다. 이는 접속 성공·미실행 상태이며 CUDA 문제라는 뜻이 아니다. `run`을 실행해 나오는 승인 대기 오류도 설정을 우회할 이유가 아니다.

## 10. 본실험 시작과 장시간 실행

이 절은 **실험 담당자가 승인과 추론 구현 검증을 완료한 뒤에만** 사용한다. 모든 장면은 같은 명령을 사용하고 `--target`의 기본값은 `local`이다.

```bash
# 로컬: 등록된 서버에서 실행. SSH 연결을 유지하는 전경 호출
.venv/bin/python experiments/run_all.py run --scene office --target cuda-server
.venv/bin/python experiments/run_all.py status --scene office --target cuda-server
```

현재 `--target cuda-server`는 SSH 명령 전달 기능이며 독립적인 원격 작업 스케줄러가 아니다. 노트북 종료·네트워크 단절에도 작업이 유지된다고 보장하지 않는다. 장시간 실행은 서버의 `tmux` 세션에서 같은 프로그램을 실행하는 방식이 적합하다.

```bash
# 로컬에서 서버 접속
ssh frameutility-gpu
```

```bash
# 서버
tmux new -s frameutility
cd /home/ubuntu/projects/FrameUtility
.venv/bin/python experiments/run_all.py run --scene office
```

서버 안에서의 기본 `local`은 **지금 명령을 실행하는 서버 자체**를 뜻하므로 서버 GPU를 사용한다. tmux에서 `Ctrl+b`, 다음 `d`로 분리하고 나중에 `tmux attach -t frameutility`로 돌아온다. tmux는 서버 자체가 중지되거나 실습 자원이 회수되는 상황까지 보호하지 않는다.

다른 장면도 `--scene courtyard`처럼 이름만 바꾼다. 결과는 서버 프로젝트의 `results/executions/<scene>/` 아래에 저장되고 장치·호스트·seed는 JSON 기록에 남는다. 같은 장면의 중간 결과를 로컬과 서버에서 섞지 않는다. `run`은 자동 분석하지 않으며 분석 코드도 현재 구현을 요구하지 않는다.

진행 상태는 별도 로컬 터미널에서 `status --target cuda-server`로 확인한다. 필요하면 서버에서 GPU 상태와 장면 로그도 확인한다.

```bash
# 서버
nvidia-smi
tail -n 40 results/executions/office/mast3r_worker_0.log
```

첫 CUDA 본실험에서는 설정 그대로의 VRAM·RAM·시간·cache 크기를 기록한다. OOM이면 float16·worker 수·사진 수·반복 횟수 등을 자동 변경하지 않고 원인과 자원 조건을 실험 담당자에게 전달한다. 실행 중 패키지·모델·코드·GPU 배정을 변경하면 잠금과 충돌할 수 있다.

## 11. 결과 회수와 AWS 중지

`--target cuda-server`는 결과를 자동 다운로드하지 않는다. 회수 시에는 서버의 `results/` 전체를 먼저 별도 수신 위치에 복사한다. 이렇게 하면 로컬의 같은 장면 결과나 `scene_assignments.json`, `runtime_configurations.json`을 덮어쓰지 않는다.

```bash
# 로컬: 수신용 경로이며 run/analyze의 기본 읽기 경로는 아님
mkdir -p results/_incoming/cuda-server
rsync -av --progress --partial \
  frameutility-gpu:/home/ubuntu/projects/FrameUtility/results/ \
  results/_incoming/cuda-server/
```

실험 담당자가 해시·연구 정의·장면 중복을 확인한 뒤 정식 `results/executions/<scene>/`와 연구 메타데이터로 편입한다. 원시 점수, 최종 예측, 입력·환경 기록, 오류·복구 이력을 함께 보존하고 다운로드 검증 전에는 서버 원본을 지우지 않는다.

작업이 끝나면 조교의 운영 정책에 따라 서버를 중지하거나 반납한다. **Stop과 Terminate는 다르다.** EC2는 중지 후에도 EBS 등 남아 있는 자원의 비용이 발생할 수 있고, instance store의 자료는 중지 시 사라질 수 있다. 공인 IP도 중지·재시작 뒤 달라질 수 있으므로 SSH 주소를 다시 확인한다. 실제 예산·보존 정책은 수업 제공 조건을 따른다. [AWS 인스턴스 상태·과금](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html), [AWS 중지·재시작과 저장소](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html).

## 12. 자주 발생하는 문제

| 증상 | 먼저 확인할 것 |
|---|---|
| SSH timeout | 현재 주소·포트, 서버 실행 상태, VPN/bastion, 접속 IP에 대한 허용 경로 |
| `Permission denied (publickey)` | 사용자 이름·키 경로·키 권한·조교의 공개키 등록 여부 |
| SSH는 되지만 `--target cuda-server` 실패 | BatchMode 접속, profile의 절대 프로젝트 경로·Python, 비대화형 환경 |
| `nvidia-smi` 실패 | GPU가 실제 할당됐는지, NVIDIA 드라이버가 제공됐는지 |
| `torch.cuda.is_available()`가 False | 실행 중인 `.venv/bin/python`, CPU wheel 여부, GPU 드라이버·접근 권한 |
| `no kernel image` 또는 GPU 아키텍처 오류 | GPU 세대를 지원하는 공식 Torch CUDA wheel 조합인지 |
| `libGL.so` 등 OpenCV import 오류 | 해당 OS의 시스템 라이브러리. 배포 패키지를 임의로 headless로 바꾸지 않음 |
| RoPE 확장 `undefined symbol` | 설치한 Torch·CUDA·컴파일러와 빌드 산출물의 일치 여부 |
| manifest의 protocol/sets/GT 오류 | 아직 최종 추론 입력으로 완성되지 않은 준비 자료인지 실험 담당자에게 확인 |
| `awaits user approval` / `not yet fully implemented` | 현재 프로젝트 상태에 따른 차단. 서버 설치 오류와 구분 |
| 환경·과학적 절차 변경 오류 | 코드·패키지·장치·입력 변경을 확인. 잠금 JSON을 삭제해 강제로 재개하지 않음 |
| GPU OOM | 실제 VRAM, 다른 사용자의 점유, 계획된 3 worker 사용량. 조건을 바꾸기 전에 담당자와 원인 해결 |

## 13. 팀원에게 전달할 완료 보고

서버 담당자는 아래를 채워 실험 담당자에게 전달한다. 키 파일·비밀번호·AWS 비밀키는 보고에 포함하지 않는다.

```text
제공 서비스 / instance ID 또는 관리 식별자:
서버 OS / CPU 아키텍처:
SSH 별칭 / 접속 방식 / 사용자 / 포트:
서버 프로젝트 절대 경로:
GPU 모델 / 할당 번호 / VRAM / 드라이버:
CPU 수 / RAM / 디스크 종류·여유 공간:
자동 중지·세션 만료·사용 기한:
Python / Torch / TorchVision / Torch CUDA runtime:
실제 CUDA wheel 설치 명령:
pip check 및 작은 CUDA 행렬곱·미분 결과:
RoPE 확장 사용 여부 / 빌드 환경:
프로젝트 커밋·미커밋 변경 / vendor 커밋 대조 결과:
자료 snapshot 식별 / 전송 완료·checksum 확인 결과:
가중치 SHA-256·MASt3R config 확인 결과:
단위 시험 결과 / skip한 시험 이름·이유:
로컬에서 status --target cuda-server 호출 결과:
설치·운영에서 남은 문제:
```

**인프라 준비 완료 기준:** 비대화형 SSH, 실제 CUDA 연산, 필요한 import, 코드·가중치·자료 일치, 원격 status가 확인되고 남은 제약이 기록된 상태다. 최종 본실험 승인·GT 채점 검증 완료 여부는 별도로 실험 담당자가 결정한다.
