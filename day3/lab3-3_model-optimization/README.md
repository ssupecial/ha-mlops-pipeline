# Lab 3-3: Model Optimization

ONNX 변환 & 양자화를 통한 모델 최적화

## 📋 개요

| 항목 | 내용 |
|------|------|
| **실습 시간** | 40분 (Part 1: 15분 / Part 2: 10분 / Part 3: 15분) |
| **난이도** | ⭐⭐⭐ (중급) |
| **학습 목표** | ONNX 변환, 동적 양자화, MLflow 벤치마크 기록 |
| **사전 요구사항** | Lab 3-1, Lab 3-2 완료, Kubeflow Jupyter 접속 가능 |

## 🎯 학습 목표

1. ONNX 포맷으로 모델 변환하여 프레임워크 독립성 확보
2. 동적 양자화를 적용하여 모델 크기 및 추론 속도 최적화
3. 벤치마크 결과를 MLflow에 기록하여 실험 추적

## 📁 디렉토리 구조

```
lab3-3_model-optimization/
├── README.md                     # 이 파일
├── notebook/
│   └── model_optimization.ipynb  # Jupyter Notebook 실습
├── scripts/
│   ├── 1_onnx_conversion.py      # ONNX 변환 스크립트
│   ├── 2_quantization.py         # 양자화 스크립트
│   └── 3_benchmark.py            # 벤치마크 & MLflow 기록
└── outputs/                      # 생성된 모델 저장 (자동 생성)
```

## ⚙️ 사전 준비

### 1. Kubeflow Jupyter Notebook 접속

Kubeflow 대시보드에서 Jupyter Notebook 서버 시작 후 터미널 열기

### 2. 실습 디렉토리 이동

```bash
cd day3/lab3-3_model-optimization
```

### 3. 필요 패키지 설치 (필요 시)

```bash
pip install scikit-learn onnx onnxruntime skl2onnx mlflow boto3
```

### 4. IRSA 설정 확인 (MLflow S3 저장용)

```python
import boto3
sts = boto3.client('sts')
print(sts.get_caller_identity()['Arn'])
# 출력: arn:aws:sts::ACCOUNT_ID:assumed-role/mlflow-s3-access-role/...
```

> ⚠️ **주의**: IRSA가 설정되지 않으면 MLflow 아티팩트 저장 시 `NoCredentialsError` 발생
> 관리자에게 IRSA 설정 요청 또는 `setup-irsa-for-students.sh` 스크립트 실행 필요

## 🚀 실습 진행

### 방법 1: Jupyter Notebook 사용 (권장)

```bash
# Jupyter에서 notebook/lab3-3_model_optimization.ipynb 열기
```

### 방법 2: Python 스크립트 사용

```bash
# Part 1: ONNX 변환
python scripts/1_onnx_conversion.py

# Part 2: 양자화
python scripts/2_quantization.py

# Part 3: 벤치마크 & MLflow 기록
export MLFLOW_TRACKING_URI=http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000
python scripts/3_benchmark.py
```

## 📊 예상 결과

### 모델 크기 비교

| 모델 | 크기 | 변화율 |
|------|------|--------|
| 원본 sklearn | 171.38 KB | - |
| ONNX | 72.17 KB | **-58%** |
| 양자화 | 72.20 KB | -58% |

### 추론 속도 비교 (1000회 평균)

| 모델 | 추론 시간 | 속도 향상 |
|------|-----------|-----------|
| 원본 sklearn | 8.64 ms | 1.0x |
| ONNX | 0.13 ms | **68.4x** |
| 양자화 | 0.13 ms | 68.0x |

## 🔍 MLflow에서 결과 확인

1. MLflow UI 접속: `http://<mlflow-url>:5000`
2. Experiments → `lab3-3-model-optimization` 선택
3. 최신 Run 클릭
4. **Parameters**: n_iterations, quantization_type
5. **Metrics**: original_size_kb, onnx_speedup, quantized_accuracy 등
6. **Artifacts**: model_optimized.onnx, model_quantized.onnx

## ⚠️ 트러블슈팅

### 1. ModuleNotFoundError

```bash
pip install scikit-learn onnx onnxruntime skl2onnx mlflow boto3
```

### 2. NoCredentialsError (S3 접근 불가)

```
NoCredentialsError: Unable to locate credentials
```

**원인**: IRSA(IAM Roles for Service Accounts) 미설정

**해결**:
1. 관리자에게 IRSA 설정 요청
2. Jupyter Pod 재시작 필요

**확인 방법**:
```python
import boto3
sts = boto3.client('sts')
print(sts.get_caller_identity()['Arn'])
# assumed-role/mlflow-s3-access-role 포함되어야 함
```

### 3. MLflow 연결 실패

```bash
# 환경변수 확인
echo $MLFLOW_TRACKING_URI

# 수동 설정
export MLFLOW_TRACKING_URI=http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000
```

### 4. ONNX 변환 시 경고

```
UserWarning: X has feature names, but LogisticRegression was fitted without feature names
```

이 경고는 무시해도 됩니다. 기능에 영향 없음.

## ✅ 실습 완료 체크리스트

- [ ] 1_onnx_conversion.py 실행 완료
- [ ] 2_quantization.py 실행 완료
- [ ] 3_benchmark.py 실행 및 MLflow 기록 완료
- [ ] MLflow UI에서 `lab3-3-model-optimization` 실험 확인
- [ ] Artifacts에서 ONNX 모델 다운로드 가능 확인

## 🚗 자동차 업계 적용 사례

- **운전자 모니터링**: 졸음 감지, 주의력 분산 감지 (30fps 실시간)
- **ADAS**: 차선 이탈 경고, 전방 충돌 경고 (저지연)
- **예측 유지보수**: 엔진 이상 감지, 배터리 수명 예측 (엣지 배포)

## 📚 참고 자료

- [ONNX 공식 문서](https://onnx.ai/)
- [ONNX Runtime 최적화 가이드](https://onnxruntime.ai/docs/performance/)
- [MLflow Tracking 가이드](https://mlflow.org/docs/latest/tracking.html)

---

© 2025 현대오토에버 MLOps Training