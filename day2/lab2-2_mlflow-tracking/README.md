# Lab 2-2: MLflow Tracking & Model Registry

## 📋 실습 개요

| 항목 | 내용 |
|------|------|
| **소요시간** | 80분 |
| **난이도** | ⭐⭐⭐ (중급) |
| **목표** | MLflow로 실험 추적 및 모델 수명주기 관리 |
| **사전 요구사항** | Lab 2-1 완료, Kubeflow Jupyter 접속 가능 |

---

## 🎯 학습 목표

이 실습을 통해 다음을 학습합니다:

- **MLflow Tracking Server** 연결 및 실험 설정
- **실험 추적**: 파라미터, 메트릭, 아티팩트 로깅
- **Autolog 기능**: 한 줄로 자동 로깅 활성화
- **Model Registry**: 모델 버전 관리 및 등록
- **스테이지 전환**: Staging → Production 승격
- **Production 모델 로드**: 스테이지 기반 모델 로딩

---

## 🏗️ 실습 구조

```
Lab 2-2: MLflow Tracking & Model Registry (80분)
│
├── Part 1: 환경 설정 (10분)
│   ├── 라이브러리 import
│   ├── MLflow Tracking Server 연결
│   └── 실험(Experiment) 설정
│
├── Part 2: 데이터 준비 (5분)
│   ├── California Housing 데이터셋 로드
│   └── Train/Test 분할
│
├── Part 3: MLflow Tracking 실습 (30분)
│   ├── 기본 실험 기록 (Linear Regression)
│   ├── 하이퍼파라미터 튜닝 (Random Forest)
│   └── Autolog 자동 로깅 기능
│
├── Part 4: Model Registry 실습 (25분)
│   ├── 모델 학습 및 Registry 등록
│   ├── 스테이지 전환 (Staging → Production)
│   ├── Production 모델 로드 및 추론
│   └── 새 버전 등록 및 조건부 승격
│
└── Part 5: MLflow UI 확인 (10분)
    ├── Experiments 탭 탐색
    ├── Models 탭 탐색
    └── 실험 비교 및 분석
```

---

## 📁 파일 구조

```
lab2-2_mlflow-tracking/
├── README.md                          # ⭐ 이 파일 (실습 가이드)
├── requirements.txt                   # Python 패키지 의존성
│
├── notebooks/
│   └── mlflow_tracking.ipynb          # 📓 Jupyter Notebook 실습 (권장)
│
└── scripts/
    └── mlflow_experiment.py           # 🐍 CLI 스크립트 버전
```

---

## 🔧 사전 요구사항

### 필수 조건

- ✅ Lab 2-1 완료 (FastAPI 모델 서빙)
- ✅ Kubeflow Dashboard 접속 가능
- ✅ Jupyter Notebook 서버 실행 중
- ✅ MLflow Tracking Server 연결 가능

### 필수 패키지

```bash
# Kubeflow Jupyter에서 실행 (필요시)
pip install mlflow==2.9.2 scikit-learn==1.5.2 pandas==2.0.3 boto3==1.34.0 "numpy<2.0.0"
```

**또는 requirements.txt 사용:**

```bash
pip install -r requirements.txt
```

### MLflow 서버 연결 확인

```python
import mlflow

# Kubeflow 내부에서 연결 (클러스터 DNS 사용)
mlflow.set_tracking_uri("http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000")

# 연결 테스트
print(mlflow.get_tracking_uri())
```

---

## 📚 Part 1: 환경 설정 (10분)

### 1.1 라이브러리 Import

```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# MLflow
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

print(f"MLflow Version: {mlflow.__version__}")
```

### 1.2 MLflow Tracking Server 연결

```python
# Tracking URI 설정 (Kubeflow 내부)
MLFLOW_TRACKING_URI = "http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 실험 설정
EXPERIMENT_NAME = "california-housing"
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"✅ Tracking URI: {mlflow.get_tracking_uri()}")
print(f"✅ Experiment: {EXPERIMENT_NAME}")
```

---

## 📚 Part 2: 데이터 준비 (5분)

### California Housing 데이터셋

```python
# 데이터 로드
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"학습 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")
print(f"\n특성(Features): {list(X.columns)}")
```

**데이터셋 정보:**

| 특성 | 설명 |
|------|------|
| MedInc | 블록 그룹의 중앙 소득 |
| HouseAge | 블록 그룹의 평균 주택 연식 |
| AveRooms | 가구당 평균 방 개수 |
| AveBedrms | 가구당 평균 침실 개수 |
| Population | 블록 그룹 인구 |
| AveOccup | 가구당 평균 거주자 수 |
| Latitude | 위도 |
| Longitude | 경도 |

---

## 📚 Part 3: MLflow Tracking 실습 (30분)

### 3.1 MLflow Tracking 핵심 개념

| 개념 | 설명 | 예시 |
|------|------|------|
| **Run** | 하나의 실험 실행 단위 | 모델 1회 학습 |
| **Parameters** | 모델의 하이퍼파라미터 (입력) | n_estimators=100 |
| **Metrics** | 성능 지표 (출력) | r2_score=0.85 |
| **Artifacts** | 모델 파일, 그래프 등 | model.pkl |
| **Tags** | 실행에 대한 메타데이터 | stage=baseline |

### 3.2 실험 1: Linear Regression (Baseline)

```python
with mlflow.start_run(run_name="linear-baseline"):
    # 1. 파라미터 기록
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("fit_intercept", True)
    
    # 2. 모델 학습
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 3. 예측 및 평가
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # 4. 메트릭 기록
    mlflow.log_metrics({
        "rmse": rmse,
        "r2_score": r2,
        "mae": mae
    })
    
    # 5. 모델 저장
    mlflow.sklearn.log_model(model, "model")
    
    # 6. 태그 추가
    mlflow.set_tag("stage", "baseline")
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

### 3.3 실험 2: Random Forest (Hyperparameter Tuning)

```python
with mlflow.start_run(run_name="rf-baseline"):
    # 하이퍼파라미터 설정
    n_estimators = 100
    max_depth = 10
    
    # 파라미터 기록
    mlflow.log_params({
        "model_type": "RandomForest",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": 42
    })
    
    # 모델 학습
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 평가
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    mlflow.log_metrics({"rmse": rmse, "r2_score": r2})
    mlflow.sklearn.log_model(model, "model")
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
```

### 3.4 Autolog 자동 로깅

```python
# Autolog 활성화 - 한 줄로 모든 것을 자동 기록!
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="rf-autolog"):
    model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    # 파라미터, 메트릭, 모델이 자동으로 기록됨!

# Autolog 비활성화
mlflow.sklearn.autolog(disable=True)
```

**Autolog 지원 프레임워크:**

| 프레임워크 | 활성화 코드 |
|------------|-------------|
| Scikit-learn | `mlflow.sklearn.autolog()` |
| TensorFlow | `mlflow.tensorflow.autolog()` |
| PyTorch | `mlflow.pytorch.autolog()` |
| XGBoost | `mlflow.xgboost.autolog()` |

---

## 📚 Part 4: Model Registry 실습 (25분)

### 4.1 Model Registry 개념

**Model Registry**는 모델의 전체 수명주기를 관리하는 중앙 저장소입니다.

**스테이지 흐름:**

```
None → Staging → Production → Archived
 │        │           │           │
 │        │           │           └── 폐기/백업
 │        │           └── 실제 서비스 운영
 │        └── 테스트/검증 환경
 └── 신규 등록
```

### 4.2 모델 학습 및 Registry 등록

```python
MODEL_NAME = "california-housing-rf"

with mlflow.start_run(run_name="rf-registry-v1"):
    n_estimators = 150
    max_depth = 15
    
    mlflow.log_params({
        "model_type": "RandomForest",
        "n_estimators": n_estimators,
        "max_depth": max_depth
    })
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    mlflow.log_metrics({"rmse": rmse, "r2_score": r2})
    
    # ⭐ Registry에 모델 등록 (핵심!)
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name=MODEL_NAME  # 이 파라미터로 Registry에 자동 등록
    )
    
    print(f"✅ 모델이 Registry에 등록되었습니다!")
    print(f"✅ Model Name: {MODEL_NAME}")
```

### 4.3 스테이지 전환

```python
from mlflow import MlflowClient

client = MlflowClient()

# 최신 버전 확인
model_info = client.get_registered_model(MODEL_NAME)
latest_version = model_info.latest_versions[0].version

# Staging으로 승격
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version,
    stage="Staging"
)
print(f"✅ Version {latest_version} → Staging 완료!")

# Production으로 승격
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version,
    stage="Production"
)
print(f"✅ Version {latest_version} → Production 완료!")
```

### 4.4 Production 모델 로드

```python
# ❌ 비추천: 버전 고정
# model = mlflow.pyfunc.load_model("models:/model-name/1")

# ✅ 권장: 스테이지 기반 로드
model_uri = f"models:/{MODEL_NAME}/Production"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# 추론 테스트
sample = X_test.iloc[:5]
predictions = loaded_model.predict(sample)

print("예측 결과:")
for i, (pred, actual) in enumerate(zip(predictions, y_test.iloc[:5])):
    print(f"  Sample {i+1}: 예측={pred:.2f}, 실제={actual:.2f}")
```

### 4.5 조건부 자동 승격

```python
# 성능 기준으로 자동 승격 (R² > 0.8)
R2_THRESHOLD = 0.8

with mlflow.start_run(run_name="rf-registry-v2"):
    model = RandomForestRegressor(n_estimators=200, max_depth=18, random_state=42)
    model.fit(X_train, y_train)
    
    r2 = r2_score(y_test, model.predict(X_test))
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(model, "model", registered_model_name=MODEL_NAME)
    
    if r2 >= R2_THRESHOLD:
        model_info = client.get_registered_model(MODEL_NAME)
        new_version = model_info.latest_versions[0].version
        
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version,
            stage="Production",
            archive_existing_versions=True  # 기존 Production 버전 아카이브
        )
        print(f"🎉 Version {new_version} → Production 자동 승격! (R²={r2:.4f})")
    else:
        print(f"⚠️ 성능 미달로 승격 보류 (R²={r2:.4f} < {R2_THRESHOLD})")
```

---

## 📚 Part 5: MLflow UI 확인 (10분)

### MLflow UI 접속 방법

```bash
# 터미널에서 실행 (클라이언트 PC)
# USER_NUM을 본인 번호로 변경 (예: 01, 02, ..., 20)
kubectl port-forward svc/mlflow-server -n kubeflow-user${USER_NUM} 5000:5000

> ℹ️ **멀티테넌트 환경**
> 
> 각 사용자는 독립된 MLflow 서버를 가지고 있습니다.
> - 자신의 실험만 보입니다 (다른 사용자 실험 격리)
> - 실험 이름에 user prefix를 붙일 필요가 없습니다
> - Artifact는 자신의 S3 버킷에 저장됩니다

# 브라우저에서 접속
http://localhost:5000
```

### UI에서 확인할 내용

**1. Experiments 탭:**
- `california-housing` 실험 클릭
- Run 목록 확인 (linear-baseline, rf-baseline 등)
- 각 Run의 Parameters, Metrics, Artifacts 확인
- Run 비교 기능 (Compare 버튼)

**2. Models 탭:**
- `california-housing-rf` 모델 클릭
- Version 목록 및 Stage 확인
- Source Run 링크로 실험 역추적 (Model Lineage)
- 버전별 메트릭 비교

---

## ✅ 완료 체크리스트

- [ ] MLflow Tracking Server 연결 성공
- [ ] Linear Regression 실험 기록 완료
- [ ] Random Forest 실험 기록 완료
- [ ] Autolog 자동 로깅 테스트 완료
- [ ] Model Registry에 모델 등록 완료
- [ ] 스테이지 전환 (Staging → Production) 완료
- [ ] Production 모델 로드 및 추론 테스트 완료
- [ ] MLflow UI에서 실험 결과 확인 완료

---

## 🛠️ 트러블슈팅

### 문제 1: MLflow 서버 연결 실패

**증상:**
```
ConnectionError: Unable to connect to MLflow server
```

**해결:**
```bash
# MLflow 서버 상태 확인
kubectl get pods -n mlflow-system

# 서비스 확인
kubectl get svc -n mlflow-system

# 로그 확인
kubectl logs -n mlflow-system -l app=mlflow-server
```

### 문제 2: 실험이 보이지 않음

**증상:** MLflow UI에서 실험을 찾을 수 없음

**해결:**
```python
# Tracking URI가 올바르게 설정되었는지 확인
print(mlflow.get_tracking_uri())

# 실험 목록 확인
client = MlflowClient()
experiments = client.search_experiments()
for exp in experiments:
    print(f"{exp.name}: {exp.experiment_id}")
```

### 문제 3: S3 아티팩트 저장 실패

**증상:**
```
NoCredentialsError: Unable to locate credentials
```

**해결:**
- IRSA(IAM Roles for Service Accounts)가 설정되어 있는지 확인
- PodDefault가 올바르게 적용되었는지 확인

```bash
# PodDefault 확인
kubectl get poddefault -n kubeflow-user${USER_NUM}
```

### 문제 4: Model Registry 접근 실패

**증상:**
```
RestException: RESOURCE_DOES_NOT_EXIST
```

**해결:**
```python
# 등록된 모델 목록 확인
client = MlflowClient()
models = client.search_registered_models()
for model in models:
    print(model.name)

# 모델이 없으면 먼저 등록
mlflow.sklearn.log_model(
    model, "model",
    registered_model_name="california-housing-rf"
)
```

### 문제 5: 스테이지 전환 실패

**증상:**
```
MlflowException: Cannot transition model version
```

**해결:**
```python
# 현재 버전 및 스테이지 확인
model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
for mv in model_versions:
    print(f"Version {mv.version}: {mv.current_stage}")
```

---

## 📚 참고 자료

- [MLflow 공식 문서](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking API](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [California Housing 데이터셋](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- [GitHub Repository](https://github.com/fastcampusdevmlops/ha-mlops-pipeline)

---

## 🔗 핵심 코드 요약

```python
# MLflow 연결
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("my-experiment")

# 실험 기록
with mlflow.start_run(run_name="my-run"):
    mlflow.log_params({"param1": value1})
    mlflow.log_metrics({"metric1": value1})
    mlflow.sklearn.log_model(model, "model", 
                             registered_model_name="my-model")

# 스테이지 전환
client = MlflowClient()
client.transition_model_version_stage(
    name="my-model", version=1, stage="Production"
)

# Production 모델 로드
model = mlflow.pyfunc.load_model("models:/my-model/Production")
```

---

## 🚀 다음 실습

**Lab 2-3: KServe 배포**
- KServe 아키텍처 이해
- InferenceService 작성
- MLflow 모델 → KServe 배포
- Canary 배포 전략

---

© 2025 현대오토에버 MLOps Training