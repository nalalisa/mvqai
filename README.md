# MVQ: Mitral & Aortic Valve Quantification

3D TEE(Transesophageal Echocardiography) 기반으로 승모판(Mitral Valve, MV)과 대동맥판(Aortic Valve, AoV)의 구조를 학습하고 정량화하기 위한 연구 프로젝트입니다.  
현재 저장소는 `Hydra + PyTorch Lightning + MONAI` 조합을 중심으로, 반복 실험과 재현성 확보를 최우선으로 두고 설계되어 있습니다.

이 문서는 현재 로컬 폴더에 실제로 존재하는 파일들을 기준으로, 프로젝트의 구조와 설계 의도를 가능한 한 상세하게 설명하는 아키텍처 문서입니다.

---

## 1. 프로젝트 배경과 문제 정의

이 프로젝트가 다루는 핵심 문제는 크게 두 가지입니다.

1. 의료 영상에서 판막 구조를 안정적으로 찾아내고 정량화해야 한다는 점
2. 특히 AoV처럼 얇고 닫힌 링 구조를 학습할 때 단순 voxel overlap 기반 학습만으로는 구조가 쉽게 끊어진다는 점

기존 접근은 대체로 다음 한계를 가집니다.

- segmentation 결과에 후처리 규칙(rule-based geometry)을 적용해 랜드마크나 면적을 계산
- control point를 단순 좌표 회귀로 학습하거나 hard tube mask로 변환해 supervision 제공
- 초음파 노이즈, 해부학적 변이, 비등방성 축 해상도로 인해 구조가 무너질 수 있음

특히 AoV는 다음 조건 때문에 학습이 어렵습니다.

- 전체 3D volume에서 차지하는 비율이 매우 작음
- 얇고 연속적인 폐곡선 형태임
- 예측이 조금만 약해져도 중간이 끊어지거나 ring이 터질 수 있음
- Dice류 지표만으로는 topology 붕괴를 충분히 벌주기 어려움

따라서 본 프로젝트는 단순 segmentation을 넘어서, `soft heatmap regression + topology-aware loss` 방향으로 설계되어 있습니다.

---

## 2. 프로젝트의 핵심 목표

현재 설계에서 지향하는 목표는 아래와 같습니다.

### 2.1 해부학적 구조 보존

모델이 단순히 target 영역에 높은 response를 내는 것을 넘어서, AoV의 링 구조가 끊기지 않도록 학습시키는 것이 중요합니다.  
이를 위해 skeleton 기반의 `Soft-clDice` 계열 loss를 포함합니다.

### 2.2 극단적 클래스 불균형 대응

AoV heatmap은 전체 volume 대비 foreground 비율이 매우 작기 때문에, 기본 MSE나 BCE만 사용하면 배경으로 붕괴하기 쉽습니다.  
이를 완화하기 위해 foreground에 더 큰 가중치를 주는 `Masked/Weighted MSE` 계열 손실을 사용합니다.

### 2.3 실험 재현성

loss 조합, alpha 비율, sigma 값, backbone 아키텍처, trainer 옵션을 코드 수정 없이 Hydra YAML만 바꿔서 실험할 수 있도록 구성합니다.

### 2.4 향후 확장성

현재는 AoV heatmap 중심의 학습 스캐폴드지만, 향후 아래 방향으로 확장이 가능하도록 설계돼 있습니다.

- MV segmentation + AoV heatmap multi-task
- ROI mask 기반 loss 제한
- anisotropy 대응을 위한 DynUNet
- WandB를 통한 시각화 및 실험 추적
- Docker 기반 환경 고정

---

## 3. 기술 스택과 각 도구의 역할

### 3.1 Hydra

Hydra는 실험 파라미터 관리 계층입니다.

역할:

- 모델 선택
- loss 조합 변경
- datamodule 인자 변경
- trainer 옵션 변경
- 실험별 출력 디렉토리 자동 분리
- multirun sweep 수행

이 프로젝트에서 Hydra를 쓰는 이유는, 연구 실험에서 코드보다 설정이 더 자주 바뀌기 때문입니다.  
예를 들어 아래 같은 실험을 코드 수정 없이 수행할 수 있어야 합니다.

```bash
python src/train.py -m loss.alpha=0.3,0.5,0.7
python src/train.py -m model=unet,attention_unet
python src/train.py -m dataset.sigma=1.0,2.0,3.0
```

### 3.2 PyTorch Lightning

Lightning은 학습 루프를 구조화합니다.

역할:

- `training_step`, `validation_step` 관리
- optimizer 연결
- checkpoint 저장
- logger 연동
- seed 설정 및 trainer orchestration

연구 코드가 커질수록 train loop, validation loop, logging, checkpointing이 섞이기 쉬운데, Lightning이 이를 정리해 줍니다.

### 3.3 MONAI

MONAI는 의료영상용 모델과 transform을 제공합니다.

역할:

- 3D UNet, Attention UNet, DynUNet 제공
- 의료영상용 dictionary transform 제공
- 3D volumetric pipeline 구성 지원

이 프로젝트는 일반적인 computer vision보다 3D voxel, anisotropy, small object, soft label이 중요하므로 MONAI의 이점이 큽니다.

### 3.4 Torch

실제 tensor 연산과 모델 파라미터 학습은 PyTorch가 담당합니다.

### 3.5 Docker

PyTorch, CUDA, MONAI, nibabel, scipy 등의 버전 차이를 줄여서 환경 재현성을 확보합니다.

---

## 4. 현재 저장소의 실제 폴더 구조

아래는 현재 로컬 폴더 기준 구조입니다.

```text
MVQ/
├── configs/
│   ├── config.yaml
│   ├── datamodule/
│   │   └── tee_3d.yaml
│   ├── dataset/
│   │   └── tee_3d.yaml
│   ├── loss/
│   │   ├── hybrid_cldice.yaml
│   │   └── masked_mse.yaml
│   ├── model/
│   │   ├── attention_unet.yaml
│   │   ├── dynunet.yaml
│   │   └── unet.yaml
│   └── trainer/
│       └── default.yaml
├── data/
│   ├── processed/
│   ├── raw/
│   └── splits/
│       ├── train.json
│       └── val.json
├── docker/
│   └── Dockerfile
├── notebooks/
├── outputs/
├── scripts/
│   ├── mrk_to_heatmap.py
│   └── preprocess_heatmap.py
├── src/
│   ├── datamodules/
│   │   ├── __init__.py
│   │   ├── tee_3d.py
│   │   └── tee_datamodule.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── combined.py
│   │   ├── combined_loss.py
│   │   ├── masked_mse.py
│   │   └── soft_cldice.py
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── monai_nets.py
│   │   └── monai_wrappers.py
│   ├── __init__.py
│   ├── lightning_module.py
│   └── train.py
├── .gitignore
├── pyproject.toml
└── README.md
```

이 구조에서 중요한 점은, 현재 저장소가 완전히 단일 경로로 정리된 최종본이라기보다, 연구용 스캐폴드와 대안 구현이 함께 존재하는 상태라는 것입니다.  
즉, 일부 영역에는 중복 구현이 있습니다.

대표적으로:

- `src/datamodules/tee_3d.py` 와 `src/datamodules/tee_datamodule.py`
- `src/losses/combined_loss.py` 와 `src/losses/combined.py`
- `src/losses/soft_cldice.py` 와 `src/losses/combined_loss.py` 내부 `SoftClDiceLoss`
- `src/networks/monai_nets.py` 와 `src/networks/monai_wrappers.py`
- `scripts/preprocess_heatmap.py` 와 `scripts/mrk_to_heatmap.py`

README는 이 상태를 숨기지 않고, 실제 현재 구조를 있는 그대로 문서화합니다.

---

## 5. 상위 디렉토리별 역할

### 5.1 `configs/`

실험 설정의 중심입니다.  
학습 로직은 `src/`에 있지만, 어떤 모델을 쓸지, 어떤 loss를 쓸지, 데이터가 어떤 형식인지, trainer가 어떤 옵션으로 돌아갈지는 모두 이 폴더에서 결정됩니다.

하위 그룹은 다음과 같습니다.

- `configs/model/`
  MONAI 네트워크 구조 선택

- `configs/loss/`
  손실 함수와 관련 하이퍼파라미터 정의

- `configs/datamodule/`
  현재 기본 스캐폴드가 사용하는 데이터 로딩 설정

- `configs/dataset/`
  대안 데이터 파이프라인 설정

- `configs/trainer/`
  Lightning trainer 관련 설정

- `configs/config.yaml`
  메인 엔트리 설정

### 5.2 `data/`

실제 학습 데이터가 위치하는 계층입니다.

- `raw/`
  원본 초음파 볼륨, 원본 markup JSON 등

- `processed/`
  heatmap 또는 후처리된 target

- `splits/`
  train/val manifest JSON

### 5.3 `scripts/`

원본 `mrk.json` control points를 soft Gaussian heatmap으로 바꾸는 전처리 스크립트가 위치합니다.

### 5.4 `src/`

학습 파이프라인의 핵심 코드입니다.

- DataModule
- custom loss
- MONAI model wrapper
- LightningModule
- train entrypoint

### 5.5 `docker/`

학습 환경을 고정하기 위한 Dockerfile을 포함합니다.

### 5.6 `outputs/`

Hydra 실행 시 실험별로 생성되는 산출물 위치입니다.  
checkpoint, hydra config snapshot, logger output 등이 들어갈 수 있습니다.

### 5.7 `notebooks/`

현재는 placeholder 성격이지만, 향후 실험 결과 시각화, 분석, metric 정리, ITK-SNAP/Slicer 결과 확인용 notebook이 들어갈 위치입니다.

---

## 6. 핵심 실행 경로 개요

현재 프로젝트를 학습 관점에서 보면, 대표 실행 흐름은 다음과 같습니다.

1. `src/train.py` 실행
2. Hydra가 `configs/config.yaml` 로드
3. 여기서 선택된 `model`, `loss`, `datamodule`, `trainer` config를 읽음
4. `hydra.utils.instantiate()` 로 객체 생성
5. `src/lightning_module.py` 가 model/loss/optimizer를 묶음
6. `src/datamodules/tee_3d.py` 또는 `src/datamodules/tee_datamodule.py` 가 데이터 로드
7. Lightning `Trainer.fit()` 으로 학습 수행

개념 흐름으로 적으면 아래와 같습니다.

```text
configs/*.yaml
    ↓
src/train.py
    ↓
instantiate(model / loss / datamodule / trainer)
    ↓
src/lightning_module.py
    ↓
MONAI network forward
    ↓
Weighted MSE / clDice 계산
    ↓
validation metric logging
```

---

## 7. 메인 설정 파일 설계

### 7.1 `configs/config.yaml`

이 파일은 전체 실험을 조립하는 메인 설정 파일입니다.

일반적으로 다음 내용을 담습니다.

- 기본 defaults
- experiment 이름
- seed
- logger 설정
- optimizer 설정
- trainer 설정
- hydra output 경로

설계 의도는 단순합니다.

- 코드에는 실행 절차만 둔다
- 실험 제어는 config에 둔다

즉, 모델을 바꾸거나 loss weight를 바꾸거나 epoch 수를 바꾸기 위해 `train.py`를 손대지 않는 것이 원칙입니다.

### 7.2 defaults 기반 조립

Hydra의 defaults는 아래 같은 조합 실험을 쉽게 만듭니다.

- `model=unet`
- `model=attention_unet`
- `model=dynunet`
- `loss=masked_mse`
- `loss=hybrid_cldice`

이 구조 덕분에 논문 실험 표를 만들 때 코드 분기 없이 조합형 sweep이 가능합니다.

---

## 8. 모델 설정 계층

모델 설정은 `configs/model/` 아래에 있습니다.

### 8.1 `configs/model/unet.yaml`

기본 3D UNet 설정입니다.

의도:

- 가장 안정적이고 해석 가능한 baseline
- AoV heatmap regression의 기본 기준선
- 이후 AttentionUnet이나 DynUNet과 비교하기 위한 참조 모델

현재 구성은 일반적으로 다음 인자를 포함합니다.

- `spatial_dims`
- `in_channels`
- `out_channels`
- `channels`
- `strides`
- `num_res_units`
- `norm`
- `act`

이 조합은 의료영상 3D segmentation에서 검증된 구조를 그대로 가져온 형태입니다.

### 8.2 `configs/model/attention_unet.yaml`

Attention gate를 사용하는 3D UNet 계열입니다.

의도:

- 작고 중요한 판막 구조에 집중
- skip feature 중 노이즈 성분 억제
- chamber wall, 주변 초음파 artefact, clutter background의 영향 감소

특히 작은 해부학적 구조를 찾는 문제에서는 일반 UNet보다 이점이 있을 수 있습니다.

### 8.3 `configs/model/dynunet.yaml`

DynUNet은 비등방성 의료영상에서 유리한 옵션입니다.

이 프로젝트에서 의미 있는 이유:

- 초음파 volume에서 축별 spacing이 동일하지 않을 수 있음
- Z 방향 정보가 XY보다 거칠 수 있음
- AoV ring이 특정 축에서 찌그러지거나 끊기는 현상을 줄이고 싶음

따라서 `1x2x2` 같은 비대칭 stride를 층별로 줄 수 있는 구조가 중요할 수 있습니다.

---

## 9. 네트워크 래퍼 계층

네트워크 래퍼는 `src/networks/` 아래에 있습니다.

### 9.1 왜 wrapper를 두는가

MONAI 모델을 바로 `_target_`으로 넣을 수도 있지만, wrapper를 두면 아래 장점이 있습니다.

- Hydra instantiate가 더 명확해짐
- 추후 custom head 추가가 쉬움
- multi-head output으로 확장 가능
- 공통 activation/post-processing을 추가할 지점 확보

### 9.2 `src/networks/monai_wrappers.py`

현재 README 요약본 기준 주요 경로로 보이는 파일입니다.

포함 클래스:

- `MonaiUNet`
- `MonaiAttentionUnet`
- `MonaiDynUNet`

이 클래스들은 내부적으로 MONAI의 `UNet`, `AttentionUnet`, `DynUNet` 객체를 멤버로 가지고, `forward`에서 그대로 호출합니다.

즉, 지금 단계에서는 "thin wrapper"입니다.

### 9.3 `src/networks/monai_nets.py`

유사한 역할을 하는 대안 구현입니다.  
실질적으로 wrapper 계층의 또 다른 버전으로 이해하면 됩니다.

현재 상태 해석:

- 기능은 거의 겹침
- 향후 하나로 통합 가능
- 지금은 연구 진행 과정에서 공존하는 구현으로 보는 것이 맞음

---

## 10. 데이터 계층의 두 가지 경로

현재 저장소에는 DataModule 계층이 두 벌 존재합니다.

1. `src/datamodules/tee_3d.py`
2. `src/datamodules/tee_datamodule.py`

둘 다 MONAI dictionary transform 기반이지만, 데이터 스키마와 supervision 철학이 약간 다릅니다.

### 10.1 `src/datamodules/tee_3d.py`

이 파일은 현재 메인 스캐폴드에서 사용하는 단순 경로입니다.

기본 가정:

- 각 샘플은 `image`와 `label`을 가진다
- `label`은 AoV soft heatmap 혹은 dense target 역할을 한다

대표 기능:

- manifest 로드
- train/val transform 구성
- `Dataset` 또는 `CacheDataset` 선택
- `DataLoader` 반환

주요 transform:

- `LoadImaged`
- `EnsureChannelFirstd`
- `ScaleIntensityRanged`
- `RandSpatialCropd`
- `RandFlipd`
- `RandGaussianNoised`
- `EnsureTyped`

설계 의도:

- 빠르게 baseline을 시작할 수 있는 단순한 구조
- image와 label만 있으면 바로 학습 가능
- soft heatmap regression에 적합

장점:

- 구조가 단순함
- 실험 시작이 빠름
- manifest 스키마가 간단함

한계:

- `mask`를 이용한 ROI 제한 supervision에는 바로 연결되지 않음
- `sigma`가 config에 있지만 현재 transform 내부에서 직접 쓰이지 않음
- 더 복잡한 multi-task supervision으로 가기 전 단계의 설계

### 10.2 `src/datamodules/tee_datamodule.py`

이 파일은 보다 명시적인 target/mask 구조를 상정하는 대안 DataModule입니다.

기본 가정:

- 각 샘플은 `image`, `target`, `mask`를 가질 수 있다
- 상대경로는 `data_root` 기준으로 절대경로로 보정된다

주요 transform:

- `LoadImaged`
- `EnsureChannelFirstd`
- `SpatialPadd`
- 선택적 `ScaleIntensityRanged`
- 학습 시 `RandFlipd`
- 학습 시 `RandRotate90d`
- `EnsureTyped`

설계 의도:

- ROI 기반 supervision을 수용
- loss 계산 시 valid region만 강조할 수 있도록 `mask`를 함께 읽음
- 상대경로 기반 manifest를 깔끔하게 다룰 수 있게 함

장점:

- `mask`와 결합한 loss 계산이 쉬움
- 데이터 스키마가 더 풍부함
- 향후 MV/AoV 멀티태스크에도 확장 가능

현재 의미:

- 실험이 진전되면 오히려 이 경로가 더 중심이 될 가능성이 큼
- 현재는 메인 스캐폴드와 공존하는 상태

---

## 11. 데이터 매니페스트 설계

현재 저장소에는 두 가지 manifest 철학이 공존합니다.

### 11.1 단순형: `image + label`

예:

```json
[
  {
    "image": "C:/path/to/patient_01_image.nii.gz",
    "label": "C:/path/to/patient_01_heatmap.nii.gz"
  }
]
```

이 형식은 다음 경우에 적합합니다.

- 빠르게 baseline 학습을 시작할 때
- AoV heatmap regression 단일 task일 때
- loss가 별도 mask 없이도 충분히 동작할 때

### 11.2 확장형: `image + target + mask`

예:

```json
[
  {
    "image": "images/case001.nrrd",
    "target": "targets/case001_heatmap.nrrd",
    "mask": "masks/case001_mask.nrrd"
  }
]
```

이 형식은 다음 경우에 적합합니다.

- 특정 영역에서만 supervision을 주고 싶을 때
- leaflets/annulus/ROI를 다르게 다루고 싶을 때
- loss 계산을 해부학적으로 의미 있는 영역으로 제한하고 싶을 때

현재 저장소의 README는 이 두 형식을 모두 문서화해야 맞습니다.  
왜냐하면 실제 코드가 두 형식을 모두 수용하는 방향으로 존재하기 때문입니다.

---

## 12. Heatmap 생성 전처리 설계

AoV는 point set를 직접 회귀하는 것보다, 닫힌 곡선을 따라 부드러운 3D Gaussian heatmap을 만드는 방식이 학습 안정성 측면에서 더 유리합니다.

전처리 스크립트는 이 목적을 위해 존재합니다.

### 12.1 `scripts/mrk_to_heatmap.py`

이 파일은 현재 요약형 README에서 기준으로 사용되던 전처리 스크립트입니다.

주요 단계:

1. `mrk.json` 로드
2. control point 추출
3. 닫힌 spline 보간
4. 보간 샘플을 따라 3D Gaussian patch 생성
5. voxel-wise max 방식으로 ring heatmap 합성
6. 결과를 `.nrrd`로 저장

핵심 설계 포인트:

- 전체 3D volume에 대해 매 점마다 full Gaussian을 다시 그리는 대신, local patch만 계산하여 효율 확보
- hard label tube가 아니라 soft label ring 생성
- `sigma`를 실험 변수로 둘 수 있음

### 12.2 `scripts/preprocess_heatmap.py`

이 파일은 제가 추가한 대안 전처리 스크립트입니다.

차이점:

- 출력이 `.nii.gz`
- 구현이 더 직관적이고 단순함
- 전체 volume 크기의 Gaussian을 각 점에 대해 생성하는 구조라 이해는 쉽지만, patch 기반보다 계산량은 더 큼

현재 의미:

- 빠르게 prototype을 만들고 검증하기 좋은 버전
- 향후 실제 production-scale preprocessing에서는 `mrk_to_heatmap.py` 방식이 더 효율적일 가능성 있음

### 12.3 두 스크립트가 함께 존재하는 이유

현재 상태는 연구 초기 스캐폴드이기 때문에 다음 두 요구가 동시에 존재합니다.

- 빠르게 이해 가능한 버전 필요
- 실제 데이터셋 전체를 처리할 효율적인 버전 필요

그래서 NIfTI와 NRRD, 직관 구현과 patch 구현이 모두 남아 있는 상태로 볼 수 있습니다.

---

## 13. Loss 설계 철학

이 프로젝트에서 loss 설계는 가장 핵심입니다.

왜냐하면 단순 segmentation보다 아래 두 문제가 더 중요하기 때문입니다.

1. foreground가 극단적으로 작음
2. 링 구조가 끊기지 않아야 함

이 두 문제는 서로 다른 loss가 담당합니다.

### 13.1 Weighted / Masked MSE의 역할

기본 아이디어:

- 배경 voxel은 매우 많다
- foreground는 매우 적다
- 단순 평균 오차를 쓰면 모델이 배경만 잘 맞춰도 loss가 작아진다

따라서 foreground voxel에 큰 가중치를 줘서 모델이 작은 구조를 무시하지 못하게 합니다.

### 13.2 Soft-clDice의 역할

Dice류 loss는 overlap은 볼 수 있어도 topology 자체를 직접 감시하지는 못합니다.  
반면 clDice는 skeleton 또는 중심선 구조를 이용해 연결성을 더 직접적으로 반영합니다.

AoV 같은 얇은 링 구조에서는 다음 상황이 자주 발생합니다.

- 넓게 보면 겹치는 것처럼 보여도 중간이 끊어짐
- 예측이 퍼지거나 터져서 진짜 링이 아님
- 중심선 구조가 맞지 않음

이때 Soft-clDice는 단순 voxel loss보다 더 의미 있는 벌점을 줄 수 있습니다.

---

## 14. 현재 저장소의 Loss 구현 1: `src/losses/combined_loss.py`

이 파일은 통합형 구현입니다.  
하나의 파일 안에 다음 구성요소가 함께 들어 있습니다.

- `SoftClDiceLoss`
- `WeightedMSELoss`
- `CombinedLoss`

### 14.1 `WeightedMSELoss`

역할:

- `y_true`가 foreground threshold보다 큰 위치에 추가 가중치 부여

일반 개념:

```text
weight = foreground_weight if y_true > threshold else 1
loss = mean(weight * (y_pred - y_true)^2)
```

의미:

- heatmap 값이 존재하는 위치를 더 중요하게 본다
- 모델이 배경만 예측하는 해를 선택하지 못하게 한다

### 14.2 `SoftClDiceLoss`

역할:

- differentiable skeletonization 근사를 통해 topology 유사도를 측정

일반 흐름:

1. prediction에서 soft skeleton 근사
2. target에서 soft skeleton 근사
3. skeleton과 원래 신호의 교차 정합도 계산
4. precision / sensitivity 계열 값을 조합해 clDice 계산

의미:

- 구조적 연결성 보존
- 중심선 형상 유사도 반영

### 14.3 `CombinedLoss`

역할:

- 두 loss를 선형 결합

개념식:

```text
L = alpha * WeightedMSE + (1 - alpha) * SoftClDice
```

여기서 `alpha`는 실험의 핵심 하이퍼파라미터입니다.

- `alpha`가 크면 voxel fidelity 중심
- `alpha`가 작으면 topology 중심

즉, 이 프로젝트에서 `alpha sweep`은 매우 중요한 실험입니다.

---

## 15. 현재 저장소의 Loss 구현 2: 분리형 경로

다음 파일들은 보다 모듈화된 분리형 loss 설계입니다.

- `src/losses/masked_mse.py`
- `src/losses/soft_cldice.py`
- `src/losses/combined.py`

### 15.1 `src/losses/masked_mse.py`

제공 클래스:

- `MaskedMSELoss`

특징:

- 내부에서 `torch.sigmoid(preds)` 수행
- foreground weight 반영
- optional `mask` 지원

즉, 단순 weighted MSE를 넘어서, ROI 내부에서만 loss를 제한할 수 있는 구조입니다.

### 15.2 `src/losses/soft_cldice.py`

제공 클래스:

- `SoftCLDiceLoss`

특징:

- 내부 sigmoid 적용
- 축별 max-pooling 기반 erosion/opening 근사
- optional `mask` 지원

이 설계는 topology loss에도 mask-aware supervision을 결합할 여지를 줍니다.

### 15.3 `src/losses/combined.py`

제공 클래스:

- `CombinedLoss`

특징:

- `masked_mse`와 `soft_cldice`를 하위 모듈로 조합
- Hydra instantiate 친화적 구조
- loss별 독립 교체가 쉬움

### 15.4 통합형과 분리형의 차이

통합형 `combined_loss.py`:

- 단순함
- 빠르게 시작하기 좋음
- 파일 수가 적음

분리형 `masked_mse.py + soft_cldice.py + combined.py`:

- mask-aware 실험에 더 유리
- loss를 개별적으로 교체하거나 로깅하기 쉬움
- 연구 확장성이 높음

현재 저장소는 두 버전이 함께 존재하며, 이는 연구 발전 과정에서 자주 나타나는 중간 상태입니다.

---

## 16. 모델과 Loss의 관계

현재 설계는 segmentation logits를 바로 binary mask로 맞추는 것이 아니라, heatmap 또는 soft target에 맞추는 방향에 더 가깝습니다.

즉, 모델의 출력은 대체로 다음 성격을 가집니다.

- 3D dense response map
- AoV ring 위치에서 높은 확률 또는 강도를 가지는 volume
- soft heatmap target과 비교 가능한 tensor

이때 loss는 다음 역할 분담을 합니다.

- `Weighted/Masked MSE`
  개별 voxel의 intensity 수준을 맞춤

- `Soft-clDice`
  중심선/연결성 구조를 맞춤

즉, 이 프로젝트의 핵심 철학은 "값도 맞추고 구조도 맞춘다"입니다.

---

## 17. LightningModule 설계

### 17.1 `src/lightning_module.py`

이 파일은 모델, loss, optimizer를 하나의 학습 단위로 묶습니다.

주요 역할:

- config를 저장
- Hydra로 model instantiate
- Hydra로 loss instantiate
- `forward` 정의
- `training_step`, `validation_step` 정의
- validation metric 집계
- optimizer 반환

### 17.2 설계 포인트

LightningModule이 맡는 주요 책임은 아래와 같습니다.

1. 데이터 배치에서 `image`, `label` 또는 target tensor를 꺼낸다
2. 모델 forward를 수행한다
3. 필요 시 sigmoid를 적용한다
4. loss를 계산한다
5. metric을 로깅한다

현재 설계 철학상 중요한 점은 activation 위치입니다.

- 어떤 loss 구현은 logits를 받아 내부에서 sigmoid를 수행
- 어떤 loss 구현은 probability를 기대

따라서 README 기준으로도, 앞으로 loss 경로를 하나로 통합할 때 이 activation 정책을 명확히 정리하는 것이 중요합니다.

### 17.3 validation metric

현재 또는 계획된 validation metric의 의미는 아래와 같습니다.

- `val/loss`
  전체 목적 함수

- `val/dice`
  overlap 품질

- `val/topology_score`
  ring connectivity 보존 정도를 proxy로 측정

향후 추가 가능 지표:

- connected component 수
- skeleton branch count
- annulus continuity score
- Hausdorff distance
- 중심선 기반 정합 지표

---

## 18. Trainer 설계

### 18.1 `configs/trainer/default.yaml`

Trainer 관련 기본 옵션은 별도 config로 관리됩니다.

대표 인자:

- `accelerator`
- `devices`
- `max_epochs`
- `precision`
- `gradient_clip_val`
- `log_every_n_steps`
- `check_val_every_n_epoch`
- `benchmark`
- `deterministic`

설계 의도:

- trainer 동작을 코드에서 고정하지 않고 실험 가능 항목으로 분리
- mixed precision, gradient clipping, deterministic 여부를 연구 상황에 따라 제어

### 18.2 재현성과 속도 사이의 trade-off

예를 들어:

- `benchmark: true`
  속도는 좋아질 수 있지만 완전 재현성에는 불리할 수 있음

- `deterministic: false`
  일부 연산에서 빠르지만 seed 재현성은 약해질 수 있음

따라서 실제 논문용 실험에서는 이 부분을 더 엄격하게 조정할 필요가 있습니다.

---

## 19. Train 엔트리포인트 설계

### 19.1 `src/train.py`

이 파일은 실제 실행의 시작점입니다.

주요 역할:

1. Hydra 설정 로드
2. 설정 내용을 출력
3. logger 생성
4. LightningModule 생성
5. DataModule 생성
6. Trainer 생성
7. `trainer.fit()` 호출

### 19.2 설계 철학

`train.py`는 최대한 얇아야 합니다.

즉:

- 실험 조합은 config에 있다
- 학습 세부 동작은 LightningModule에 있다
- 데이터 처리 세부는 DataModule에 있다
- 모델 세부는 network wrapper에 있다

이렇게 계층을 분리하면 실험 변경 시 코드 수정 범위가 작아집니다.

---

## 20. 전처리에서 학습까지의 데이터 흐름

이 프로젝트를 end-to-end로 보면 데이터는 다음 흐름을 가집니다.

### 단계 1. 원본 수집

- 3D 초음파 볼륨
- `mrk.json` control points

### 단계 2. Heatmap 생성

`scripts/mrk_to_heatmap.py` 또는 `scripts/preprocess_heatmap.py` 실행

결과:

- AoV ring soft target
- `.nrrd` 또는 `.nii.gz` 형식의 dense heatmap

### 단계 3. Manifest 작성

`data/splits/train.json`, `data/splits/val.json` 에 이미지와 타깃 경로를 기록

### 단계 4. DataModule 로딩

MONAI transform을 거쳐 tensor batch 생성

### 단계 5. 모델 추론

MONAI 3D network가 dense response volume 출력

### 단계 6. Loss 계산

- weighted/ masked MSE
- topology-aware clDice
- 또는 둘의 조합

### 단계 7. 검증 및 기록

- loss
- dice
- topology proxy
- checkpoint
- logger output

---

## 21. Hydra 그룹별 설계 의도

### 21.1 model 그룹

아키텍처 실험을 담당합니다.

예:

- baseline UNet
- Attention U-Net
- DynUNet

### 21.2 loss 그룹

학습 목적 함수 실험을 담당합니다.

예:

- weighted MSE only
- hybrid MSE + clDice

### 21.3 dataset 또는 datamodule 그룹

데이터 스키마 및 transform 파이프라인을 담당합니다.

예:

- 단순 label 기반
- target + mask 기반

### 21.4 trainer 그룹

학습 환경 실험을 담당합니다.

예:

- epoch 수
- precision
- gradient clipping
- deterministic 설정

이 구조 덕분에 실험 표를 구성할 때 축을 명확히 나눌 수 있습니다.

---

## 22. 연구 실험 관점에서 중요한 변수

현재 코드와 README를 종합하면, 이 프로젝트에서 중요한 실험 변수는 다음과 같습니다.

### 22.1 `alpha`

`Weighted/Masked MSE`와 `Soft-clDice`의 혼합 비율입니다.  
이 값은 구조 정확도와 topology 보존 사이의 균형을 결정합니다.

### 22.2 `sigma`

control point를 heatmap으로 바꿀 때의 Gaussian 두께입니다.

작으면:

- sharper supervision
- 작은 localization 오차에 민감

크면:

- smoother supervision
- 더 안정적일 수 있지만 ring이 퍼질 수 있음

### 22.3 모델 구조

- UNet
- AttentionUnet
- DynUNet

이들은 feature aggregation, noise filtering, anisotropy 대응 성격이 다릅니다.

### 22.4 foreground weight

foreground voxel에 주는 가중치입니다.

작으면:

- 작은 구조를 무시할 위험

크면:

- foreground 중심 학습 강화
- 하지만 지나치면 noisy hotspot에 과적합할 위험

### 22.5 ROI mask 사용 여부

mask-aware supervision을 쓰면 background 노이즈를 줄일 수 있지만, 준비해야 할 라벨과 파이프라인이 더 많아집니다.

---

## 23. 현재 설계의 장점

### 23.1 실험 제어가 명확함

Hydra 구조 덕분에 실험 축이 코드가 아니라 설정으로 드러납니다.

### 23.2 topology-aware 학습 방향이 반영돼 있음

의료영상에서 얇은 구조를 다룰 때 중요한 clDice 계열 손실이 이미 설계에 포함돼 있습니다.

### 23.3 MONAI 기반 3D 구조가 준비돼 있음

모델과 transform 모두 의료영상 친화적으로 구성돼 있어 확장성이 좋습니다.

### 23.4 baseline과 고급 경로가 모두 존재함

빠르게 시작할 수 있는 단순 경로와, mask-aware로 확장할 수 있는 대안 경로가 함께 있어 연구 진행이 유연합니다.

---

## 24. 현재 설계의 미완성 또는 중복 지점

현재 저장소는 실행 가능한 스캐폴드이지만, 최종 정리 단계는 아닙니다.

대표적으로 아래가 남아 있습니다.

### 24.1 DataModule 중복

- `tee_3d.py`
- `tee_datamodule.py`

둘 중 무엇을 표준 경로로 삼을지 정리가 필요합니다.

### 24.2 Loss 구현 중복

- `combined_loss.py`
- `combined.py + masked_mse.py + soft_cldice.py`

activation 정책과 mask 지원을 기준으로 하나의 표준으로 정리하는 것이 좋습니다.

### 24.3 Network wrapper 중복

- `monai_nets.py`
- `monai_wrappers.py`

현재는 thin wrapper 수준이므로 하나로 통합하기 쉽습니다.

### 24.4 Heatmap 스크립트 중복

- `.nii.gz` 중심
- `.nrrd` 중심

실제 데이터 포맷 기준으로 하나를 메인 경로로 정리하는 것이 좋습니다.

### 24.5 좌표계 처리

`mrk.json`의 control point가 voxel 좌표인지, world/RAS 좌표인지 명확히 확정해야 합니다.  
실제 3D Slicer 마크업이면 affine 또는 spacing/origin 방향 처리 로직이 필요할 수 있습니다.

---

## 25. 실행 방법

### 25.1 패키지 설치

```bash
pip install -e .
```

### 25.2 기본 학습 실행

```bash
python src/train.py
```

### 25.3 멀티런 실험

loss 비율 sweep:

```bash
python src/train.py -m loss.alpha=0.3,0.5,0.7
```

모델 구조 비교:

```bash
python src/train.py -m model=unet,attention_unet
```

sigma 실험:

```bash
python src/train.py -m dataset.sigma=1.0,2.0,3.0
```

### 25.4 heatmap 생성

NRRD 출력:

```bash
python scripts/mrk_to_heatmap.py ^
  --input data/raw/case001.mrk.json ^
  --output data/processed/case001_heatmap.nrrd ^
  --shape 128 128 128 ^
  --sigma 2.0
```

NIfTI 출력:

```bash
python scripts/preprocess_heatmap.py ^
  --json-path data/raw/patient_01_mrk.json ^
  --output-path data/processed/patient_01_heatmap.nii.gz ^
  --sigma 2.0
```

---

## 26. 데이터 파일 준비 예시

### 26.1 단순형 manifest

```json
[
  {
    "image": "C:/dataset/patient_01_image.nii.gz",
    "label": "C:/dataset/patient_01_heatmap.nii.gz"
  },
  {
    "image": "C:/dataset/patient_02_image.nii.gz",
    "label": "C:/dataset/patient_02_heatmap.nii.gz"
  }
]
```

### 26.2 확장형 manifest

```json
[
  {
    "image": "images/patient_01.nrrd",
    "target": "targets/patient_01_heatmap.nrrd",
    "mask": "masks/patient_01_roi.nrrd"
  },
  {
    "image": "images/patient_02.nrrd",
    "target": "targets/patient_02_heatmap.nrrd",
    "mask": "masks/patient_02_roi.nrrd"
  }
]
```

---

## 27. Docker 설계

### `docker/Dockerfile`

Dockerfile은 CUDA가 포함된 PyTorch 베이스 이미지를 기반으로 프로젝트를 설치합니다.

이 계층이 중요한 이유:

- 병원 서버와 개인 워크스테이션 간 환경 차이 감소
- CUDA/PyTorch/MONAI 버전 고정
- 재학습 재현성 확보
- 실험 환경 공유 용이

향후 개선 가능:

- system package 추가
- non-root user 설정
- volume mount 기준 문서화
- training and inference 분리 이미지 구성

---

## 28. `pyproject.toml` 역할

이 파일은 프로젝트의 Python 패키지 메타데이터와 의존성을 정의합니다.

현재 의미:

- `pip install -e .` 가능
- 필요한 의존성 설치
- `src` 패키지 경로 등록

이 파일 덕분에 로컬 개발 환경과 Docker 환경이 같은 의존성 선언을 공유할 수 있습니다.

---

## 29. 향후 권장 리팩터링 방향

현재 아키텍처는 연구 시작과 실험 확장에는 충분히 좋지만, 장기적으로는 아래 방향으로 정리하는 것이 바람직합니다.

### 29.1 데이터 스키마 통합

`image + label` 또는 `image + target + mask` 중 하나를 표준으로 정합니다.

개인적으로는 향후 확장성을 고려하면 `image + target + mask`가 더 강력합니다.

### 29.2 loss 구현 통합

mask-aware 설계와 activation 정책을 기준으로 하나의 loss 계열로 정리합니다.

### 29.3 network wrapper 통합

`monai_wrappers.py`와 `monai_nets.py`를 하나로 정리합니다.

### 29.4 전처리 포맷 통일

NRRD와 NIfTI 중 프로젝트 기준 포맷을 정합니다.

### 29.5 metric 확장

Dice 외에 topology와 geometry 지표를 더 추가합니다.

예:

- connected components
- loop continuity
- centerline length consistency
- surface distance

---

## 30. 현재 상태 요약

현재 저장소는 다음 상태로 이해하면 가장 정확합니다.

- 단순 문서-only 저장소가 아니라, 실제 학습 스캐폴드가 이미 구축된 상태
- Hydra 기반 실험 관리 구조가 존재
- MONAI 3D 모델 선택지가 준비됨
- AoV ring heatmap용 전처리 코드가 존재
- topology-aware loss 방향이 코드에 반영됨
- 다만 일부 계층은 연구 발전 과정상 중복 구현이 남아 있음

즉, 이 프로젝트는 "완전히 정리된 production 코드"라기보다는,  
"실험을 본격적으로 시작할 수 있을 만큼 충분히 구조화된 연구 아키텍처"입니다.

---

## 31. 핵심 파일 빠른 참조

- 메인 설정: [configs/config.yaml](C:\Users\Andrew\Documents\PROJECT\MVQ\configs\config.yaml)
- 모델 설정: [configs/model/unet.yaml](C:\Users\Andrew\Documents\PROJECT\MVQ\configs\model\unet.yaml)
- 모델 설정: [configs/model/attention_unet.yaml](C:\Users\Andrew\Documents\PROJECT\MVQ\configs\model\attention_unet.yaml)
- 모델 설정: [configs/model/dynunet.yaml](C:\Users\Andrew\Documents\PROJECT\MVQ\configs\model\dynunet.yaml)
- loss 설정: [configs/loss/hybrid_cldice.yaml](C:\Users\Andrew\Documents\PROJECT\MVQ\configs\loss\hybrid_cldice.yaml)
- loss 설정: [configs/loss/masked_mse.yaml](C:\Users\Andrew\Documents\PROJECT\MVQ\configs\loss\masked_mse.yaml)
- dataset 설정: [configs/dataset/tee_3d.yaml](C:\Users\Andrew\Documents\PROJECT\MVQ\configs\dataset\tee_3d.yaml)
- datamodule 설정: [configs/datamodule/tee_3d.yaml](C:\Users\Andrew\Documents\PROJECT\MVQ\configs\datamodule\tee_3d.yaml)
- trainer 설정: [configs/trainer/default.yaml](C:\Users\Andrew\Documents\PROJECT\MVQ\configs\trainer\default.yaml)
- 학습 엔트리포인트: [src/train.py](C:\Users\Andrew\Documents\PROJECT\MVQ\src\train.py)
- Lightning 모듈: [src/lightning_module.py](C:\Users\Andrew\Documents\PROJECT\MVQ\src\lightning_module.py)
- 단순 DataModule: [src/datamodules/tee_3d.py](C:\Users\Andrew\Documents\PROJECT\MVQ\src\datamodules\tee_3d.py)
- 확장 DataModule: [src/datamodules/tee_datamodule.py](C:\Users\Andrew\Documents\PROJECT\MVQ\src\datamodules\tee_datamodule.py)
- 통합형 loss: [src/losses/combined_loss.py](C:\Users\Andrew\Documents\PROJECT\MVQ\src\losses\combined_loss.py)
- 분리형 loss: [src/losses/combined.py](C:\Users\Andrew\Documents\PROJECT\MVQ\src\losses\combined.py)
- 분리형 loss: [src/losses/masked_mse.py](C:\Users\Andrew\Documents\PROJECT\MVQ\src\losses\masked_mse.py)
- 분리형 loss: [src/losses/soft_cldice.py](C:\Users\Andrew\Documents\PROJECT\MVQ\src\losses\soft_cldice.py)
- 네트워크 래퍼: [src/networks/monai_wrappers.py](C:\Users\Andrew\Documents\PROJECT\MVQ\src\networks\monai_wrappers.py)
- 네트워크 래퍼: [src/networks/monai_nets.py](C:\Users\Andrew\Documents\PROJECT\MVQ\src\networks\monai_nets.py)
- heatmap 전처리: [scripts/mrk_to_heatmap.py](C:\Users\Andrew\Documents\PROJECT\MVQ\scripts\mrk_to_heatmap.py)
- heatmap 전처리: [scripts/preprocess_heatmap.py](C:\Users\Andrew\Documents\PROJECT\MVQ\scripts\preprocess_heatmap.py)
- 의존성 정의: [pyproject.toml](C:\Users\Andrew\Documents\PROJECT\MVQ\pyproject.toml)
- Docker 환경: [docker/Dockerfile](C:\Users\Andrew\Documents\PROJECT\MVQ\docker\Dockerfile)

---

## 32. 마지막 메모

현재 README는 단순 사용법 요약이 아니라, 이 저장소의 실제 코드 상태와 연구 아키텍처를 함께 설명하는 문서입니다.  
향후 코드 리팩터링이 진행되면, 특히 중복 구현이 정리되는 시점에 README도 함께 간소화하거나 표준 경로 중심으로 재정렬하는 것이 좋습니다.
