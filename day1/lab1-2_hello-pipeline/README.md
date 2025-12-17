# Lab 1-2: Hello World Pipeline

## 📋 실습 개요

| 항목 | 내용 |
|------|------|
| **소요시간** | 40분 |
| **난이도** | ⭐⭐ |
| **목표** | Kubeflow Pipelines로 첫 번째 ML 워크플로우 작성 및 실행 |

## 🎯 학습 목표

이 실습을 통해 다음을 학습합니다:
- **KFP SDK v2** 기본 개념 이해
- **Component** 정의 방법
- **Pipeline** 구성 방법
- **파이프라인 컴파일** (Python → YAML)
- **Kubeflow UI**를 통한 파이프라인 업로드 및 실행
- **실행 결과 및 로그** 확인 방법

---

## 🏗️ 실습 구조

```
Lab 1-2: Hello World Pipeline (40분)
│
├── Part 1: 파이프라인 이해 (10분)
│   ├── Component 개념
│   ├── Pipeline 구조
│   └── DAG (실행 흐름)
│
├── Part 2: 파이프라인 작성 (15분)
│   ├── Component 정의 (add, multiply, print_result)
│   ├── Pipeline 함수 작성
│   └── YAML 컴파일
│
└── Part 3: 파이프라인 실행 (15분)
    ├── Kubeflow UI 업로드
    ├── Run 생성
    ├── 실행 모니터링
    └── 결과 확인
```

---

## 📁 파일 구조

```
lab1-2_hello-pipeline/
├── README.md                    # ⭐ 이 파일 (실습 가이드)
├── hello_pipeline.py            # 파이프라인 Python 스크립트
├── requirements.txt             # Python 패키지 의존성
└── notebooks/
    └── hello_pipeline.ipynb     # Jupyter Notebook 버전
```

---

## 🔧 사전 요구사항

### 필수 조건
- ✅ Lab 1-1 완료 (MLOps 환경 구축)
- ✅ Kubeflow Dashboard 접속 가능
- ✅ Python 3.11+ 환경

### 필수 패키지 설치

```bash
# 패키지 설치
pip install kfp>=2.0.0
```

**또는 requirements.txt 사용:**
```bash
pip install -r requirements.txt
```

---

## 📚 Part 1: 파이프라인 이해 (10분)

### 파이프라인 구조

이 실습에서 만들 파이프라인은 다음과 같은 구조입니다:

```
입력 파라미터
├─ a = 10
├─ b = 20
└─ factor = 3

실행 흐름 (DAG)
┌─────────────┐
│     add     │  ← a, b를 입력받아 합계 계산
│  (a + b)    │
└──────┬──────┘
       │ sum = 30
       ▼
┌─────────────┐
│  multiply   │  ← sum과 factor를 곱함
│ (sum × f)   │
└──────┬──────┘
       │ product = 90
       ▼
┌─────────────┐
│print_result │  ← 최종 결과 출력
│   (출력)     │
└─────────────┘
```

**계산 과정:**
1. `add(10, 20)` → 30
2. `multiply(30, 3)` → 90
3. `print_result(90)` → "Final Result: 90"

### Component란?

**Component**는 파이프라인의 한 단계를 수행하는 독립적인 함수입니다.

**특징:**
- Python 함수로 정의
- `@dsl.component` 데코레이터 사용
- 입력/출력 타입 힌트 필수
- 각 컴포넌트는 별도의 컨테이너에서 실행

**예시:**
```python
@dsl.component(base_image='python:3.11')
def add(a: int, b: int) -> int:
    """두 숫자를 더합니다."""
    result = a + b
    print(f"Add: {a} + {b} = {result}")
    return result
```

### Pipeline이란?

**Pipeline**은 여러 Component를 연결한 워크플로우입니다.

**특징:**
- `@dsl.pipeline` 데코레이터 사용
- Component들의 실행 순서 정의
- Component 간 데이터 전달 (`.output` 사용)
- 파라미터로 실행 시 값 변경 가능

**예시:**
```python
@dsl.pipeline(name='Hello World Pipeline')
def hello_pipeline(a: int = 10, b: int = 20):
    """간단한 파이프라인"""
    add_task = add(a=a, b=b)  # Step 1
    multiply_task = multiply(x=add_task.output)  # Step 2
    print_result(value=multiply_task.output)  # Step 3
```

### DAG (Directed Acyclic Graph)

**DAG**는 파이프라인의 실행 흐름을 나타내는 방향성 비순환 그래프입니다.

**특징:**
- 각 노드는 Component
- 화살표는 데이터 흐름
- 의존성에 따라 자동으로 실행 순서 결정
- Kubeflow UI의 "Graph" 탭에서 시각화

---

## 🚀 Part 2: 파이프라인 작성 (15분)

### Step 2-1: 파이프라인 스크립트 확인

**파일: `hello_pipeline.py`**

```python
"""
Lab 1-2: Hello World Pipeline
간단한 덧셈과 곱셈을 수행하는 Kubeflow Pipeline
"""

from kfp import dsl
from kfp import compiler

# Component 1: 두 숫자 더하기
@dsl.component(base_image='python:3.11')
def add(a: int, b: int) -> int:
    result = a + b
    print(f"Add: {a} + {b} = {result}")
    return result


# Component 2: 숫자에 factor 곱하기
@dsl.component(base_image='python:3.11')
def multiply(x: int, factor: int = 2) -> int:
    result = x * factor
    print(f"Multiply: {x} * {factor} = {result}")
    return result


# Component 3: 최종 결과 출력
@dsl.component(base_image='python:3.11')
def print_result(value: int):
    print("=" * 50)
    print(f"Final Result: {value}")
    print("=" * 50)


# Pipeline 정의
@dsl.pipeline(
    name='Hello World Pipeline',
    description='Simple addition and multiplication pipeline'
)
def hello_pipeline(
    a: int = 3,
    b: int = 5,
    factor: int = 2
):
    # Step 1: a + b 계산
    add_task = add(a=a, b=b)
    
    # Step 2: (a + b) * factor 계산
    multiply_task = multiply(
        x=add_task.output,
        factor=factor
    )
    
    # Step 3: 결과 출력
    print_result(value=multiply_task.output)


# 파이프라인 컴파일
if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=hello_pipeline,
        package_path='hello_pipeline.yaml'
    )
    print("✅ 파이프라인 컴파일 완료: hello_pipeline.yaml")
```

### Step 2-2: 파이프라인 컴파일

```bash
# 파이프라인 컴파일 실행
python pipeline_simple_v2.py
```

**예상 출력:**
```
✅ 파이프라인 컴파일 완료: hello_pipeline.yaml

다음 단계:
  1. Kubeflow Dashboard 접속
  2. Pipelines → Upload pipeline
  3. hello_pipeline.yaml 업로드
```

**생성된 파일 확인:**
```bash
ls -lh hello_pipeline.yaml
```

---

## 🚀 Part 3: 파이프라인 실행 (15분)

### Step 3-1: Kubeflow Dashboard 접속

```bash
# 포트 포워딩 (터미널 1)
export USER_NUM="01"  # 본인 번호
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

**브라우저에서 접속:**
```
http://localhost:8080
```

### Step 3-2: 파이프라인 업로드

**Kubeflow UI에서 진행:**

1. **왼쪽 메뉴에서 "Pipelines" 클릭**

2. **"+ Upload pipeline" 버튼 클릭**

3. **파이프라인 정보 입력:**
   - Pipeline Name: `Hello World Pipeline`
   - Pipeline Description: `Simple addition and multiplication`

4. **"Upload a file" 선택**
   - "Choose file" 클릭
   - `hello_pipeline.yaml` 선택

5. **"Create" 버튼 클릭**

**✅ 성공!** 파이프라인 목록에 "Hello World Pipeline"이 표시됩니다.

### Step 3-3: Run 생성

**파이프라인 상세 페이지에서:**

1. **"Create run" 버튼 클릭**

2. **Run details 입력:**
   - Run name: `hello-run-001` (영어만 사용!)
   - Experiment: "Default" 선택 또는 새로 생성

**⚠️ 중요: 한글 이름 사용 금지!**
- ❌ 잘못된 예: "실험-001", "헬로월드"
- ✅ 올바른 예: "hello-run-001", "test-run-01"

3. **Run parameters 설정:**
   ```
   a: 10
   b: 20
   factor: 3
   ```

4. **"Start" 버튼 클릭**

### Step 3-4: 실행 모니터링

**Run 상세 페이지에서:**

1. **Graph 탭**
   - 파이프라인 DAG 시각화
   - 각 노드의 상태 확인 (Pending → Running → Succeeded)
   - 녹색: 성공, 파란색: 실행 중, 회색: 대기 중

2. **각 노드 클릭하여 상세 정보 확인:**
   - Input Parameters: 입력값
   - Output Parameters: 출력값
   - Logs: 실행 로그

### Step 3-5: 결과 확인

**각 Component의 로그:**

#### add Component 로그
```
Add: 10 + 20 = 30
```

#### multiply Component 로그
```
Multiply: 30 * 3 = 90
```

#### print_result Component 로그
```
==================================================
Final Result: 90
==================================================
```

**✅ 성공!** 모든 Component가 정상적으로 실행되고 예상된 결과가 출력되었습니다.

---

## 🧪 테스트 케이스

**다양한 파라미터로 실습해보세요:**

| a | b | factor | 예상 결과 | 계산 과정 |
|---|---|--------|----------|----------|
| 3 | 5 | 2 | 16 | (3 + 5) × 2 = 16 |
| 10 | 20 | 3 | 90 | (10 + 20) × 3 = 90 |
| 7 | 3 | 5 | 50 | (7 + 3) × 5 = 50 |
| 100 | 200 | 2 | 600 | (100 + 200) × 2 = 600 |

---

## ✅ 완료 체크리스트

### Part 1: 파이프라인 이해 (10분)
- [ ] Component 개념 이해
- [ ] Pipeline 구조 이해
- [ ] DAG (실행 흐름) 이해

### Part 2: 파이프라인 작성 (15분)
- [ ] hello_pipeline.py 코드 이해
- [ ] 파이프라인 컴파일 성공
- [ ] hello_pipeline.yaml 파일 생성 확인

### Part 3: 파이프라인 실행 (15분)
- [ ] Kubeflow Dashboard 접속
- [ ] 파이프라인 업로드 성공
- [ ] Run 생성 및 시작
- [ ] Graph 탭에서 DAG 확인
- [ ] 각 Component 로그 확인
- [ ] 최종 결과 확인 (90)

---

## 🎯 학습 성과

이 실습을 완료하면:

1. ✅ **KFP SDK v2** 기본 사용법
2. ✅ **Component 정의** 방법 (`@dsl.component`)
3. ✅ **Pipeline 구성** 방법 (`@dsl.pipeline`)
4. ✅ **파이프라인 컴파일** (Python → YAML)
5. ✅ **Kubeflow UI 활용** (업로드, 실행, 모니터링)
6. ✅ **Component 간 데이터 전달** (`.output` 사용)

---

## 📖 핵심 개념 정리

### Component
- 파이프라인의 한 단계를 수행하는 독립적인 함수
- `@dsl.component` 데코레이터로 정의
- 각 컴포넌트는 별도의 컨테이너에서 실행

### Pipeline
- 여러 Component를 연결한 워크플로우
- `@dsl.pipeline` 데코레이터로 정의
- Component 간 의존성을 자동으로 관리

### DAG (Directed Acyclic Graph)
- 파이프라인의 실행 흐름을 나타내는 그래프
- 의존성에 따라 실행 순서 자동 결정
- Kubeflow UI에서 시각화

### Experiment
- Run을 논리적으로 그룹화하는 단위
- 여러 Run을 비교하고 분석

### Run
- 특정 파라미터로 파이프라인을 한 번 실행
- 각 Run은 고유한 ID를 가짐

---

## 💡 문제 해결 (Troubleshooting)

### 문제 1: "ModuleNotFoundError: No module named 'kfp'"

**원인:** KFP SDK가 설치되지 않음

**해결 방법:**
```bash
pip install kfp>=2.0.0
# 또는
pip install -r requirements.txt
```

### 문제 2: Pipeline 업로드 후 "Upload failed"

**원인:** YAML 파일이 올바르지 않음

**해결 방법:**
```bash
# 파이프라인 재컴파일
python hello_pipeline.py

# YAML 파일 존재 확인
ls -lh hello_pipeline.yaml

# YAML 문법 검증
python -c "import yaml; yaml.safe_load(open('hello_pipeline.yaml'))"
```

### 문제 3: Run 상태가 "Error"로 표시됨

**원인:** ResourceQuota로 인해 Pod 생성 실패

**증상:**
```
failed quota: kf-resource-quota: must specify cpu for: init,wait; memory for: init,wait
```

**해결 방법 (강사 실행):**
```bash
# LimitRange 설정 확인
kubectl get limitrange -n kubeflow-user${USER_NUM}

# LimitRange가 없으면 생성
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: LimitRange
metadata:
  name: default-limit-range
  namespace: kubeflow-user${USER_NUM}
spec:
  limits:
  - type: Container
    default:
      cpu: "1"
      memory: "1Gi"
    defaultRequest:
      cpu: "200m"
      memory: "256Mi"
    max:
      cpu: "4"
      memory: "8Gi"
    min:
      cpu: "50m"
      memory: "64Mi"
EOF
```

**진단 방법:**
```bash
# Workflow 상태 확인
kubectl get workflows -n kubeflow-user${USER_NUM}

# Workflow 상세 확인
kubectl describe workflow <workflow-name> -n kubeflow-user${USER_NUM}
```

### 문제 4: "Cannot get MLMD objects from Metadata store" 에러

**원인:** 파이프라인 실행 실패로 메타데이터가 생성되지 않음 (UI 표시 문제)

**확인 방법:**
```bash
# 실제 Workflow 상태 확인
kubectl get workflows -n kubeflow-user${USER_NUM}

# Error 상태인 경우 상세 확인
kubectl describe workflow <workflow-name> -n kubeflow-user${USER_NUM}
```

**해결 방법:**
1. Workflow 에러 메시지 확인 (위 명령어 실행)
2. 대부분 ResourceQuota/LimitRange 문제 → 문제 3 해결 방법 적용
3. 해결 후 파이프라인 재실행

**메타데이터 서비스 재시작 (선택사항):**
```bash
kubectl rollout restart deployment metadata-grpc-deployment -n kubeflow
kubectl rollout restart deployment ml-pipeline -n kubeflow
```

### 문제 5: Run 상태가 "Pending"에서 멈춤

**원인:** 리소스 부족 또는 파드 스케줄링 실패

**해결 방법:**
```bash
# 파드 상태 확인
kubectl get pods -n kubeflow-user${USER_NUM}

# 이벤트 확인
kubectl get events -n kubeflow-user${USER_NUM} --sort-by='.lastTimestamp'

# 특정 파드 상세 정보
kubectl describe pod <POD_NAME> -n kubeflow-user${USER_NUM}
```

### 문제 6: "UTF-8 Collation Error"

**원인:** Pipeline/Component 이름에 한글 사용

**해결 방법:**
- ❌ Pipeline name: "헬로 파이프라인"
- ✅ Pipeline name: "Hello Pipeline"
- ❌ Run name: "실험-001"
- ✅ Run name: "experiment-001"

**모든 이름과 description은 영어만 사용하세요!**

### 문제 7: Component 로그에 "Error: ..."

**원인:** Component 코드 오류

**해결 방법:**
```bash
# 로컬에서 Component 함수 테스트
python -c "
from pipeline_simple_v2 import add, multiply, print_result

# Component 함수를 일반 함수처럼 호출
result1 = add.python_func(10, 20)
result2 = multiply.python_func(result1, 3)
print_result.python_func(result2)
"
```

---

## 🔧 유용한 명령어

### 환경 확인

```bash
# 환경 변수 설정
export USER_NUM="01"  # 본인 번호로 변경

# 네임스페이스 Pod 확인
kubectl get pods -n kubeflow-user${USER_NUM}

# Workflow 목록 확인
kubectl get workflows -n kubeflow-user${USER_NUM}

# LimitRange 확인
kubectl get limitrange -n kubeflow-user${USER_NUM}

# ResourceQuota 확인
kubectl describe resourcequota -n kubeflow-user${USER_NUM}
```

### 파이프라인 관련

```bash
# 파이프라인 컴파일
python hello_pipeline.py

# YAML 파일 확인
cat hello_pipeline.yaml

# 포트 포워딩
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

### 문제 진단

```bash
# Workflow 상세 확인
kubectl describe workflow <workflow-name> -n kubeflow-user${USER_NUM}

# 파드 로그 확인
kubectl logs <POD_NAME> -n kubeflow-user${USER_NUM}

# 이벤트 확인
kubectl get events -n kubeflow-user${USER_NUM} --sort-by='.lastTimestamp' | tail -20

# 실패한 Workflow 삭제
kubectl delete workflows --all -n kubeflow-user${USER_NUM}
```

---

## 📚 다음 단계

**Lab 1-3: Batch Data Pipeline**
- AWS S3 Data Lake 구축
- ETL Pipeline 구현
- Pandas로 Batch 데이터 처리
- Bronze → Silver → Gold Layer

---

## 🔗 참고 자료

### 공식 문서
- [Kubeflow Pipelines v2 문서](https://www.kubeflow.org/docs/components/pipelines/v2/)
- [KFP SDK v2 API Reference](https://kubeflow-pipelines.readthedocs.io/en/stable/source/dsl.html)
- [Component 개발 가이드](https://www.kubeflow.org/docs/components/pipelines/v2/components/)

### 트러블슈팅
- [Kubernetes LimitRange 문서](https://kubernetes.io/docs/concepts/policy/limit-range/)
- [Kubernetes ResourceQuota 문서](https://kubernetes.io/docs/concepts/policy/resource-quotas/)
- [Argo Workflows 트러블슈팅](https://argoproj.github.io/argo-workflows/troubleshooting/)

---

© 2025 현대오토에버 MLOps Training