# Lab 3-1: Data Drift Monitoring & Auto-Retraining

## 📋 실습 개요

| 항목 | 내용 |
|------|------|
| **소요시간** | 90분 |
| **난이도** | ⭐⭐⭐ |
| **목표** | 프로덕션 모델의 Data Drift 자동 감지 및 재학습 파이프라인 구축 |

---

## 🎯 학습 목표

이 실습을 통해 다음을 학습합니다:
- **Data Drift 개념** 이해 및 감지 방법
- **Kubeflow Pipeline**을 활용한 Drift 모니터링 자동화
- **MLflow**를 사용한 메트릭 추적
- **조건부 재학습** 파이프라인 구현

---

## 🏗️ 실습 구조

```
Lab 3-1: Drift Monitoring (90분)
├── Part 1: Drift Detection (30분)
│   ├── Drift 개념 이해
│   ├── KS Test (Kolmogorov-Smirnov Test)
│   └── Drift Score 계산
│
├── Part 2: Monitoring Pipeline (30분)
│   ├── Kubeflow Pipeline 자동화
│   ├── MLflow 메트릭 기록
│   └── Alert 시스템 (시뮬레이션)
│
└── Part 3: Auto-Retraining (30분)
    ├── Drift Score 기반 재학습 결정
    ├── 모델 재학습
    └── 배포 시뮬레이션
```

---

## 📁 파일 구조

```
lab3-1_drift-monitoring/
├── README.md                    # ⭐ 이 파일 (실습 가이드)
├── requirements.txt             # Python 패키지
├── scripts/
│   ├── 1_detect_drift.py       # Part 1: Drift 감지 스크립트
│   ├── 2_monitor_pipeline.py   # Part 2: 모니터링 파이프라인
│   └── 3_retrain_pipeline.py   # Part 3: 자동 재학습 파이프라인
└── notebooks/
    ├── 1_drift_detection.ipynb     # Part 1: Drift 감지 노트북
    ├── 2_monitor_pipeline.ipynb    # Part 2: 모니터링 파이프라인 노트북
    └── 3_retrain_pipeline.ipynb    # Part 3: 자동 재학습 노트북
```

---


## 🚀 Part 1: Drift Detection (30분)

### 학습 목표
- Data Drift의 개념 이해
- KS Test를 사용한 Drift 감지
- Drift Score 계산 및 해석

### 실습 방법

**방법 1: Python 스크립트 실행**
```bash
cd lab3-1_drift-monitoring
python scripts/1_detect_drift.py
```

**방법 2: Jupyter Notebook 실행**
1. Kubeflow → Notebooks → 본인 노트북 접속
2. `notebooks/1_drift_detection.ipynb` 실행

### 핵심 개념: KS Test (Kolmogorov-Smirnov Test)

두 데이터 분포가 동일한지 검정하는 통계적 방법입니다.

```python
from scipy.stats import ks_2samp

# KS Test 수행
statistic, p_value = ks_2samp(reference_data, current_data)

# p-value < 0.05이면 분포가 다름 (Drift 감지)
if p_value < 0.05:
    print("Drift detected!")
```

### Drift Score 계산

```python
# Drift가 감지된 Feature 수 / 전체 Feature 수
drift_score = n_drifted_features / total_features

# 예: 1개 feature에서 drift / 9개 전체 feature = 0.11 (11%)
```

### 예상 출력

```
============================================================
  Lab 3-1 Part 1: Data Drift Detection
============================================================

[Step 1] Loading California Housing data...
  Reference data: 5000 samples
  Current data: 3000 samples (with simulated drift)

[Step 2] Performing Drift Detection (KS Test)...
  Feature: MedInc     - Drift: YES (p-value: 0.0000)
  Feature: HouseAge   - Drift: NO  (p-value: 0.4521)
  ...

[Step 3] Drift Summary
  Drifted Features: 1/9
  Drift Score: 0.11 (11%)
  Threshold: 0.30 (30%)
  Status: No significant drift

============================================================
  Part 1 Complete!
============================================================
```

---

## 🔄 Part 2: Monitoring Pipeline (30분)

### 학습 목표
- Kubeflow Pipeline으로 Drift 모니터링 자동화
- MLflow에 메트릭 기록
- Alert 시스템 구축 (시뮬레이션)

### 파이프라인 구조

```
┌─────────────────────┐
│ collect-production- │
│       data          │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    detect-drift     │
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌─────────┐ ┌─────────┐
│  log-   │ │  send-  │
│ metrics │ │  alert  │
└─────────┘ └─────────┘
```

### 실습 방법

**방법 1: Python 스크립트로 YAML 생성**
```bash
python scripts/2_monitor_pipeline.py
# 출력: drift_monitoring_pipeline.yaml
```

**방법 2: Jupyter Notebook 실행 (권장)**
1. `notebooks/2_monitor_pipeline.ipynb` 열기
2. **Step 0** 실행 후 **커널 재시작**
3. **Step 1**부터 순서대로 실행
4. 마지막 셀에서 `drift_monitoring_pipeline.yaml` 생성

### Kubeflow UI에서 실행

1. **Pipelines** → **+ Upload pipeline** 클릭
2. `drift_monitoring_pipeline.yaml` 파일 선택
3. **Create** 클릭
4. 파이프라인 선택 → **+ Create run** 클릭
5. Parameters 설정:
   - `sample_size`: 1000
   - `drift_threshold`: 0.3
   - `mlflow_uri`: (자동 입력)
6. **Start** 클릭

### 예상 결과

```
✅ collect-production-data: Succeeded
✅ detect-drift: Succeeded
✅ log-metrics: Succeeded
✅ send-alert: Succeeded
```

---

## 🔄 Part 3: Auto-Retraining Pipeline (30분)

### 학습 목표
- Drift Score 기반 재학습 결정
- 조건부 파이프라인 실행
- 모델 재학습 및 배포 시뮬레이션

### 파이프라인 구조

```
┌──────────────────────┐
│ check-drift-and-     │
│      decide          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    retrain-model     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    deploy-model      │
└──────────────────────┘
```

### 실습 방법

**방법 1: Python 스크립트로 YAML 생성**
```bash
python scripts/3_retrain_pipeline.py
# 출력: auto_retrain_pipeline.yaml
```

**방법 2: Jupyter Notebook 실행 (권장)**
1. `notebooks/3_retrain_pipeline.ipynb` 열기
2. **Step 0** 실행 후 **커널 재시작**
3. **Step 1**부터 순서대로 실행
4. 마지막 셀에서 `auto_retrain_pipeline.yaml` 생성

### Kubeflow UI에서 실행

1. **Pipelines** → **+ Upload pipeline** 클릭
2. `auto_retrain_pipeline.yaml` 파일 선택
3. **Create** 클릭
4. **+ Create run** 클릭
5. Parameters 설정:
   - `drift_threshold`: 0.3
   - `train_size`: 5000
   - `mlflow_uri`: (자동 입력)
6. **Start** 클릭

### 예상 결과

```
✅ check-drift-and-decide: Succeeded
✅ retrain-model: Succeeded
✅ deploy-model: Succeeded
```

---

## 💡 핵심 개념

### Data Drift란?

프로덕션 데이터의 분포가 학습 데이터와 달라지는 현상입니다.

**원인:**
- 사용자 행동 패턴 변화
- 시장 트렌드 변화
- 계절적 요인
- 데이터 수집 오류

**영향:**
- 모델 성능 저하
- 예측 정확도 감소
- 비즈니스 지표 악화

### KFP Component 정의

```python
@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def my_component(input_value: int) -> str:
    """Component docstring (English only!)"""
    import pandas as pd  # 함수 내부에서 import
    
    result = str(input_value * 2)
    print(f"Result: {result}")  # English only!
    
    return result
```

### Pipeline 정의

```python
@dsl.pipeline(
    name="my-pipeline",           # ASCII only!
    description="my pipeline"     # ASCII only!
)
def my_pipeline(param: int = 10):
    step1 = component1(input=param)
    step2 = component2(input=step1.output)  # .output으로 연결
```

### 실행 순서 제어

```python
# 방법 1: .output 사용 (데이터 전달 + 순서 제어)
step2 = component2(input=step1.output)

# 방법 2: .after() 사용 (순서만 제어)
step2 = component2(input=some_param)
step2.after(step1)
```

---

## ✅ 완료 체크리스트

### Part 1: Drift Detection
- [ ] KS Test 개념 이해
- [ ] `1_detect_drift.py` 또는 노트북 실행 성공
- [ ] Drift Score 계산 결과 확인 (예: 0.11)

### Part 2: Monitoring Pipeline
- [ ] KFP SDK 2.7.0 이상 설치 및 커널 재시작
- [ ] `drift_monitoring_pipeline.yaml` 생성
- [ ] Kubeflow UI에 업로드 성공
- [ ] 파이프라인 실행 성공 (4개 컴포넌트 모두 녹색)

### Part 3: Auto-Retraining Pipeline
- [ ] `auto_retrain_pipeline.yaml` 생성
- [ ] Kubeflow UI에 업로드 성공
- [ ] 파이프라인 실행 성공 (3개 컴포넌트 모두 녹색)

---

## 🔧 트러블슈팅 요약

| 문제 | 증상 | 해결 |
|------|------|------|
| **UTF-8 에러** | `Error 3988 Collation` | Pipeline name/description에 영어만 사용 |
| **KFP 버전 에러** | `unexpected keyword argument 'base_image'` | `pip install kfp==2.7.0` + 커널 재시작 |
| **MLflow 403** | `RBAC: access denied` | 에러 핸들링으로 자동 처리됨 |
| **패키지 미적용** | 이전 버전 로드 | 커널 재시작 |

---

## ⚠️ 중요: 사전 준비사항

### 1. KFP SDK 버전 확인

Kubeflow Pipeline을 사용하려면 **KFP SDK 2.7.0 이상**이 필요합니다.

```bash
# 버전 확인
pip show kfp

# 업그레이드 (필요시)
pip install kfp==2.7.0
```

### 2. 환경 변수 설정

```bash
# 본인의 사용자 번호로 변경
export USER_NUM="01"  # 예: 01, 02, ..., 11, 20

# 네임스페이스 설정
export NAMESPACE="kubeflow-user${USER_NUM}"
```

### 3. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 🚨 알려진 이슈 및 해결방법

### Issue 1: UTF-8 Collation 에러

**증상:**
```
Run creation failed
Error 3988 (HY000): Conversion from collation utf8mb3_general_ci into utf8mb4_0900_ai_ci impossible for parameter
```

**원인:** Kubeflow Pipeline의 MySQL 데이터베이스와 문자셋 충돌

**해결:** Pipeline name, description, docstring, print문에서 **영어(ASCII)만 사용**

```python
# ❌ 잘못된 예
@dsl.pipeline(
    name="드리프트 모니터링",  # 한글 사용 금지!
    description="자동 모니터링 파이프라인"
)

# ✅ 올바른 예
@dsl.pipeline(
    name="drift-monitoring",
    description="automated monitoring pipeline"
)
```

> **참고:** 노트북의 마크다운 셀(설명)은 한글 사용 가능합니다.

---

### Issue 2: KFP SDK 버전 에러

**증상:**
```
TypeError: component() got an unexpected keyword argument 'base_image'
```

**원인:** KFP SDK 버전이 2.7.0 미만

**해결:**
```bash
# KFP 업그레이드
pip install kfp==2.7.0

# Jupyter 노트북에서는 커널 재시작 필수!
# Kernel → Restart Kernel
```

---

### Issue 3: MLflow RBAC 에러

**증상:**
```
mlflow.exceptions.MlflowException: API request to endpoint failed with error code 403
Response body: 'RBAC: access denied'
```

**원인:** MLflow 서버에 인증이 필요하거나 RBAC 설정 문제

**해결:** 
- 본 실습 코드에는 에러 핸들링이 포함되어 있어 MLflow 연결 실패 시에도 파이프라인이 계속 진행됩니다.
- 근본적인 해결이 필요한 경우 강사에게 문의하세요.

---

### Issue 4: 커널 재시작 필요

**증상:** pip install 후에도 이전 버전의 패키지가 로드됨

**해결:**
1. pip install 실행
2. **Kernel → Restart Kernel** 메뉴 클릭
3. 처음부터 셀 다시 실행

---

## 📚 참고 자료

- [Kubeflow Pipelines SDK v2](https://www.kubeflow.org/docs/components/pipelines/v2/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- [Data Drift in ML](https://www.tensorflow.org/tfx/guide/tfdv)

---

## 🎯 다음 단계

- **Lab 3-2**: CI/CD Pipeline - GitHub Actions와 ArgoCD를 활용한 자동화
- **Project**: 팀 프로젝트 - 실전 MLOps 시스템 구축

---

© 2025 현대오토에버 MLOps Training
