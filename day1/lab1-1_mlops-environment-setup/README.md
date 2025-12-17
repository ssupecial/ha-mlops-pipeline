# Lab 1-1: MLOps 환경 구축

## 📋 실습 개요

| 항목 | 내용 |
|------|------|
| **소요시간** | 65분 |
| **난이도** | ⭐⭐ (초중급) |
| **목표** | AWS EKS 기반 MLOps 플랫폼 환경 확인 및 Tenant 검증 |
| **대상** | 수강생 15명 (user01~user15) + 강사 1명 (user20) |

---

## 🎯 학습 목표

이 실습을 통해 다음을 학습합니다:

- **AWS EKS 클러스터** 연결 및 상태 확인
- **Kubeflow Tenant** 설정 확인 (Profile, Namespace, ServiceAccount)
- **MLflow Tracking Server** 연결 및 PodDefault 확인
- **AWS S3 & ECR** 스토리지 구성 확인
- **MLOps 플랫폼** 전체 아키텍처 이해

---

## 🏗️ 실습 구조

```
Lab 1-1: MLOps 환경 구축 (65분)
│
├── 사전 준비 (10분)
│   ├── 필수 도구 설치 확인
│   ├── 환경 변수 설정
│   ├── AWS 자격 증명 설정
│   └── EKS 클러스터 연결
│
├── Part 1: Kubeflow Tenant 검증 (20분)
│   ├── Namespace 존재 확인
│   ├── Profile 및 Owner Email 확인
│   ├── ServiceAccount 확인
│   ├── ResourceQuota 확인
│   └── 권한 격리 테스트
│
├── Part 2: MLflow 환경 검증 (20분)
│   ├── MLflow Server 상태 확인
│   ├── PostgreSQL 백엔드 확인
│   ├── PodDefault 설정 확인
│   └── MLflow UI 포트 포워딩 테스트
│
└── Part 3: AWS 스토리지 확인 (15분)
    ├── S3 버킷 확인
    ├── ECR 레지스트리 확인
    ├── MLflow Artifacts 폴더 확인
    └── 전체 아키텍처 이해
```

---

## 📁 파일 구조

```
lab1-1_mlops-environment-setup/
├── README.md                          # ⭐ 이 파일 (실습 가이드)
├── verify_all.sh                      # 🔧 통합 검증 스크립트
│
├── 1_kubeflow_setup/
│   ├── verify_kubeflow.sh             # Part 1: Kubeflow 검증 스크립트
│
├── 2_mlflow_setup/
│   ├── verify_mlflow.sh               # Part 2: MLflow 검증 스크립트
│
└── 3_storage_setup/
    ├── verify_storage.sh              # Part 3: Storage 검증 스크립트
```

---

## 🎯 Tenant 구성

본 교육에서는 수강생 15명과 강사 1명에게 각각 독립된 MLOps 환경을 제공합니다.

```
┌────────────────────┬────────────────────────────────┬──────────────────────────────┐
│ Profile            │ Owner Email                    │ 리소스                       │
├────────────────────┼────────────────────────────────┼──────────────────────────────┤
│ profile-user01     │ user01@mlops.local             │ CPU 8, Memory 16Gi           │
│ profile-user02     │ user02@mlops.local             │ CPU 8, Memory 16Gi           │
│ profile-user03     │ user03@mlops.local             │ CPU 8, Memory 16Gi           │
│ profile-user04     │ user04@mlops.local             │ CPU 8, Memory 16Gi           │
│ profile-user05     │ user05@mlops.local             │ CPU 8, Memory 16Gi           │
│ profile-user06     │ user06@mlops.local             │ CPU 8, Memory 16Gi           │
│ profile-user07     │ user07@mlops.local             │ CPU 8, Memory 16Gi           │
│ profile-user08     │ user08@mlops.local             │ CPU 8, Memory 16Gi           │
│ profile-user09     │ user09@mlops.local             │ CPU 8, Memory 16Gi           │
│ profile-user10     │ user10@mlops.local             │ CPU 8, Memory 16Gi           │
│ profile-user11     │ user11@mlops.local             │ CPU 8, Memory 16Gi           │
│ profile-user12     │ user12@mlops.local             │ CPU 8, Memory 16Gi           │
│ profile-user13     │ user13@mlops.local             │ CPU 8, Memory 16Gi           │
│ profile-user14     │ user14@mlops.local             │ CPU 8, Memory 16Gi           │
│ profile-user15     │ user15@mlops.local             │ CPU 8, Memory 16Gi           │
│ profile-user20     │ user20@mlops.local (강사)       │ CPU 16, Memory 32Gi ⭐       │
└────────────────────┴────────────────────────────────┴──────────────────────────────┘
```

---

## 🔧 사전 준비 (10분)

### Step 0-1: 필수 도구 확인

**이 실습을 시작하기 전에 다음 도구가 설치되어 있어야 합니다:**

```bash
# 1. AWS CLI 버전 확인
aws --version
# 예상 출력: aws-cli/2.x.x Python/3.x.x ...

# 2. kubectl 버전 확인
kubectl version --client
# 예상 출력: Client Version: v1.27.x ...

# 3. Git 버전 확인
git --version
# 예상 출력: git version 2.x.x
```

**설치되지 않은 경우:**

| 도구 | macOS | Windows |
|------|-------|---------|
| AWS CLI | `brew install awscli` | [AWS CLI 설치 프로그램](https://aws.amazon.com/cli/) |
| kubectl | `brew install kubectl` | `choco install kubernetes-cli` |
| Git | `brew install git` | [Git for Windows](https://git-scm.com/) |

### Step 0-2: 환경 변수 설정

**⚠️ 매우 중요: 본인의 사용자 번호를 정확히 입력하세요!**

**macOS / Linux:**
```bash
# 사용자 번호 설정 (예: 01, 02, 03... 15, 20)
export USER_NUM="01"  # ⚠️ 반드시 본인 번호로 변경하세요!

# 관련 환경 변수 자동 설정
export NAMESPACE="kubeflow-user${USER_NUM}"
export S3_BUCKET="mlops-training-user${USER_NUM}"
export AWS_REGION="ap-northeast-2"

# 환경 변수 확인
echo "사용자 번호: $USER_NUM"
echo "네임스페이스: $NAMESPACE"
echo "S3 버킷: $S3_BUCKET"
```

**Windows PowerShell:**
```powershell
# 사용자 번호 설정
$env:USER_NUM = "01"  # ⚠️ 반드시 본인 번호로 변경하세요!

# 관련 환경 변수 자동 설정
$env:NAMESPACE = "kubeflow-user$env:USER_NUM"
$env:S3_BUCKET = "mlops-training-user$env:USER_NUM"
$env:AWS_REGION = "ap-northeast-2"

# 환경 변수 확인
echo "사용자 번호: $env:USER_NUM"
echo "네임스페이스: $env:NAMESPACE"
echo "S3 버킷: $env:S3_BUCKET"
```

### Step 0-3: AWS 자격 증명 설정

**강사가 제공한 AWS Access Key와 Secret Key를 준비하세요.**

```bash
# AWS 자격 증명 설정
aws configure

# 입력 항목:
# AWS Access Key ID: (강사 제공)
# AWS Secret Access Key: (강사 제공)
# Default region name: ap-northeast-2
# Default output format: json

# 설정 확인
aws sts get-caller-identity
```

**예상 출력:**
```json
{
    "UserId": "AIDAXXXXXXXXXXXXXXXXX",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/mlops-training"
}
```

### Step 0-4: EKS 클러스터 연결

```bash
# EKS 클러스터 연결
aws eks update-kubeconfig \
    --region ap-northeast-2 \
    --name mlops-training-cluster

# 연결 확인
kubectl cluster-info
```

**예상 출력:**
```
Kubernetes control plane is running at https://XXXXX.ap-northeast-2.eks.amazonaws.com
CoreDNS is running at https://XXXXX.ap-northeast-2.eks.amazonaws.com/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy
```

### Step 0-5: 실습 저장소 클론

```bash
# GitHub 저장소 클론
git clone https://github.com/fastcampusdevmlops/ha-mlops-pipeline.git
cd ha-mlops-pipeline

# Lab 1-1 디렉토리로 이동
cd day1/lab1-1_mlops-environment-setup
```

---

## 🔵 Part 1: Kubeflow Tenant 검증 (20분)

### 개요

Kubeflow Tenant는 각 사용자에게 독립된 MLOps 환경을 제공합니다. 이 섹션에서는 본인의 Tenant가 올바르게 설정되었는지 확인합니다.

### 검증 스크립트 실행

```bash
cd 1_kubeflow_setup

# 실행 권한 부여
chmod +x verify_kubeflow.sh

# 환경 변수 설정 및 실행
export USER_NUM="01"  # 본인 번호로 변경
./verify_kubeflow.sh
```

### 검증 항목

| Step | 검증 항목 | 설명 |
|------|----------|------|
| 1 | Namespace | `kubeflow-user{XX}` 네임스페이스 존재 확인 |
| 2 | Profile | `profile-user{XX}` 프로필 및 Owner email 일치 확인 |
| 3 | ServiceAccount | `default-editor`, `default-viewer` SA 확인 |
| 4 | ResourceQuota | CPU, Memory 할당량 확인 |
| 5 | RoleBinding | 사용자 권한 설정 확인 |
| 6 | PodDefault | MLflow, Pipeline 접근 설정 확인 |
| 7 | 리소스 상태 | Pods, Services, PVC 현황 확인 |
| 8 | 권한 격리 | 다른 사용자 Namespace 접근 차단 확인 |
| 9 | 시스템 상태 | Kubeflow 주요 컴포넌트 상태 확인 |
| 10 | 최종 판단 | 실습 가능 여부 판단 |

### 수동 검증 명령어

스크립트 없이 수동으로 확인하려면:

```bash
# 1. Namespace 확인
kubectl get namespace kubeflow-user${USER_NUM}

# 2. Profile 확인
kubectl get profile profile-user${USER_NUM}
kubectl get profile profile-user${USER_NUM} -o jsonpath='{.spec.owner.name}'

# 3. ServiceAccount 확인
kubectl get serviceaccount -n kubeflow-user${USER_NUM}

# 4. ResourceQuota 확인
kubectl get resourcequota -n kubeflow-user${USER_NUM}

# 5. PodDefault 확인
kubectl get poddefaults -n kubeflow-user${USER_NUM}

# 6. 권한 격리 테스트 (다른 사용자 접근 시도 - 실패해야 정상)
kubectl get pods -n kubeflow-user02  # user01인 경우
```

### 예상 결과

```
============================================================
검증 결과 요약
============================================================

  ✅ 통과: 10
  ❌ 실패: 0
  ⚠️  경고: 0
  📊 총점: 10/10

🎉 모든 검증을 완벽하게 통과했습니다!

다음 단계: Part 2 (MLflow 환경 검증)로 진행하세요.
```

---

## 🟢 Part 2: MLflow 환경 검증 (20분)

### 개요

MLflow는 ML 실험 추적, 모델 관리, 배포를 위한 플랫폼입니다. 이 섹션에서는 MLflow 서버 연결 및 Tenant별 설정을 확인합니다.

### 검증 스크립트 실행

```bash
cd ../2_mlflow_setup

# 실행 권한 부여
chmod +x verify_mlflow.sh

# 환경 변수 설정 및 실행
export USER_NUM="01"  # 본인 번호로 변경
./verify_mlflow.sh
```

### 검증 항목

| Step | 검증 항목 | 설명 |
|------|----------|------|
| 1 | Namespace | Kubeflow Namespace 존재 확인 |
| 2 | Profile | Profile 및 Owner email 확인 |
| 3 | S3 버킷 | `mlops-training-user{XX}` 버킷 확인 |
| 4 | ECR 레지스트리 | `mlops-training/user{XX}*` 확인 |
| 5 | MLflow PodDefault | `access-mlflow` PodDefault 확인 |
| 6 | MLflow Server | MLflow Tracking Server 상태 확인 |
| 7 | 권한 격리 | Namespace 간 접근 차단 확인 |

### MLflow Server 확인

```bash
# MLflow 파드 상태 확인 (자신의 네임스페이스)
kubectl get pods -n kubeflow-user${USER_NUM} -l app=mlflow-server

# MLflow 서비스 확인
kubectl get svc -n kubeflow-user${USER_NUM} | grep mlflow

# PostgreSQL 상태 확인 (MLflow Backend)
kubectl get pods -n mlflow-system -l app=postgres
```

### MLflow UI 접속 테스트

```bash
# MLflow UI 포트 포워딩
kubectl port-forward svc/mlflow-server -n kubeflow-user${USER_NUM} 5000:5000

# 브라우저에서 접속
# http://localhost:5000
```

### PodDefault 확인

PodDefault는 Jupyter Notebook에서 MLflow에 자동으로 연결할 수 있도록 환경 변수를 주입합니다.

```bash
# MLflow PodDefault 확인
kubectl get poddefault access-mlflow -n kubeflow-user${USER_NUM} -o yaml
```

**확인할 환경 변수:**
```yaml
env:
- name: MLFLOW_TRACKING_URI
  value: "http://mlflow-server.kubeflow-user{XX}.svc.cluster.local:5000"
- name: MLFLOW_S3_ENDPOINT_URL
  value: "https://s3.ap-northeast-2.amazonaws.com"
- name: AWS_DEFAULT_REGION
  value: "ap-northeast-2"
```

### 예상 결과

```
============================================================
  검증 결과 요약
============================================================

   ✅ 통과: 7
   ❌ 실패: 0
   ⚠️  경고: 0
   📊 총점: 7/7

🎉 모든 검증을 완벽하게 통과했습니다!

   다음 단계: Part 3 (AWS 스토리지 확인)로 진행하세요.
```

---

## 🟡 Part 3: AWS 스토리지 확인 (15분)

### 개요

MLOps 플랫폼에서 사용하는 AWS 스토리지(S3, ECR)를 확인합니다.

### 검증 스크립트 실행

```bash
cd ../3_storage_setup

# 실행 권한 부여
chmod +x verify_storage.sh

# 환경 변수 설정 및 실행
export USER_NUM="01"  # 본인 번호로 변경
./verify_storage.sh
```

### 검증 항목

| Step | 검증 항목 | 설명 |
|------|----------|------|
| 1 | S3 버킷 | `mlops-training-user{XX}` 버킷 존재 및 접근 확인 |
| 2 | ECR 레지스트리 | `mlops-training/user{XX}` 레지스트리 확인 |
| 3 | MLflow Artifacts | S3 MLflow Artifacts 폴더 확인 |
| 4 | Pipeline Artifacts | Kubeflow Pipeline Artifacts 폴더 확인 |
| 5 | 아키텍처 | 전체 스토리지 아키텍처 요약 |
| 6 | 데이터 흐름 | 학습 → 저장 → 배포 흐름 설명 |

### S3 버킷 확인

```bash
# S3 버킷 존재 확인
aws s3 ls s3://mlops-training-user${USER_NUM} --region ap-northeast-2

# 버킷 내용 확인
aws s3 ls s3://mlops-training-user${USER_NUM}/ --region ap-northeast-2

# MLflow Artifacts 폴더 확인
aws s3 ls s3://mlops-training-user${USER_NUM}/mlflow-artifacts/ --region ap-northeast-2
```

### ECR 레지스트리 확인

```bash
# ECR 레지스트리 확인
aws ecr describe-repositories \
    --repository-names mlops-training/user${USER_NUM} \
    --region ap-northeast-2

# ECR 로그인
aws ecr get-login-password --region ap-northeast-2 | \
    docker login --username AWS --password-stdin \
    $(aws sts get-caller-identity --query Account --output text).dkr.ecr.ap-northeast-2.amazonaws.com
```

### 스토리지 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│              AWS MLOps Storage Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐         ┌──────────────────┐          │
│  │   Kubeflow      │         │     MLflow       │          │
│  │   Pipeline      │────────▶│  Tracking Server │          │
│  │                 │         │    (Port 5000)   │          │
│  └─────────────────┘         └─────────┬────────┘          │
│          │                             │                    │
│          │                   ┌─────────▼────────┐          │
│          │                   │   PostgreSQL     │          │
│          │                   │   (Metadata DB)  │          │
│          │                   └──────────────────┘          │
│          │                                                  │
│  ┌───────▼────────┐                                        │
│  │   AWS S3       │◀────────────────────────────────────── │
│  │  (Artifacts)   │         (Model & Artifact Store)       │
│  │                │                                        │
│  │  📁 mlops-training-user{XX}/                           │
│  │     ├── mlflow-artifacts/                               │
│  │     └── kubeflow-pipeline-artifacts/                    │
│  └────────────────┘                                        │
│                                                             │
│  ┌────────────────┐                                        │
│  │   AWS ECR      │◀────────────────────────────────────── │
│  │  (Container    │         (Container Images)             │
│  │   Registry)    │                                        │
│  │                │                                        │
│  │  📦 mlops-training/user{XX}                             │
│  └────────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 데이터 흐름

```
1. 학습 실행
   └─▶ MLflow Tracking
       ├─▶ S3: Model 파일, Artifacts 저장
       └─▶ PostgreSQL: Metrics, Parameters 기록

2. 모델 배포
   ├─▶ S3: 모델 파일 조회
   ├─▶ ECR: 추론 서버 이미지 저장
   └─▶ KServe: InferenceService 생성

3. 파이프라인 실행
   ├─▶ ECR: 컴포넌트 이미지 사용
   ├─▶ S3: 입력 데이터 로드
   └─▶ S3: 실행 결과 저장
```

---

## 🔧 통합 검증 스크립트

### 전체 환경 검증

세 가지 Part를 한 번에 검증하려면:

```bash
cd ..  # lab1-1_mlops-environment-setup 디렉토리로 이동

# 통합 검증 스크립트 실행
chmod +x verify_all.sh
export USER_NUM="01"  # 본인 번호로 변경
./verify_all.sh
```

### verify_all.sh 내용

```bash
#!/bin/bash
# Lab 1-1 통합 검증 스크립트

echo "============================================================"
echo "  Lab 1-1: MLOps 환경 통합 검증"
echo "============================================================"

# 환경 변수 확인
if [ -z "$USER_NUM" ]; then
    read -p "사용자 번호를 입력하세요 (예: 01): " USER_NUM
    export USER_NUM
fi

echo ""
echo "👤 사용자: user${USER_NUM}"
echo ""

# Part 1: Kubeflow 검증
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Part 1] Kubeflow Tenant 검증"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd 1_kubeflow_setup && ./verify_kubeflow.sh
PART1_RESULT=$?
cd ..

# Part 2: MLflow 검증
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Part 2] MLflow 환경 검증"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd 2_mlflow_setup && ./verify_mlflow.sh
PART2_RESULT=$?
cd ..

# Part 3: Storage 검증
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Part 3] AWS 스토리지 검증"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd 3_storage_setup && ./verify_storage.sh
PART3_RESULT=$?
cd ..

# 최종 결과
echo ""
echo "============================================================"
echo "  최종 검증 결과"
echo "============================================================"
echo ""

TOTAL_FAIL=$((PART1_RESULT + PART2_RESULT + PART3_RESULT))

if [ $TOTAL_FAIL -eq 0 ]; then
    echo "🎉 모든 검증을 통과했습니다!"
    echo ""
    echo "다음 단계: Lab 1-2 (Hello Pipeline)로 진행하세요."
else
    echo "⚠️  일부 검증이 실패했습니다."
    echo ""
    echo "실패한 Part를 확인하고 강사에게 문의하세요."
fi

exit $TOTAL_FAIL
```

---

## 🛠️ 트러블슈팅

### 문제 1: kubectl 명령어 실행 실패

**증상:**
```
error: You must be logged in to the server (Unauthorized)
```

**해결:**
```bash
# kubeconfig 갱신
aws eks update-kubeconfig \
    --region ap-northeast-2 \
    --name mlops-training-cluster

# 연결 확인
kubectl cluster-info
```

### 문제 2: Profile Owner Email 불일치

**증상:**
```
⚠️  Owner email 불일치!
   예상: user07@mlops.local
   실제: user@example.com
```

**해결:**
강사에게 문의하여 Profile Owner 수정을 요청하세요.
```bash
# 강사가 실행할 명령어
kubectl patch profile profile-user07 --type=merge \
    -p '{"spec":{"owner":{"name":"user07@mlops.local"}}}'
```

### 문제 3: S3 버킷 접근 거부

**증상:**
```
An error occurred (AccessDenied) when calling the ListBuckets operation
```

**해결:**
```bash
# AWS 자격증명 재설정
aws configure

# 자격증명 확인
aws sts get-caller-identity
```

### 문제 4: PodDefault 없음

**증상:**
```
❌ MLflow PodDefault 없음: access-mlflow
```

**해결:**
강사에게 문의하여 PodDefault 생성을 요청하세요.

### 문제 5: MLflow Server 연결 실패

**증상:**
```
❌ MLflow Server를 찾을 수 없습니다
```

**해결:**
```bash
# MLflow 파드 상태 확인
kubectl get pods -n mlflow-system

# 파드 로그 확인
kubectl logs -n mlflow-system -l app=mlflow-server
```

### 문제 6: 다른 사용자 Namespace 접근 가능

**증상:**
```
⚠️  다른 네임스페이스 접근 가능 (권한 확인 필요)
```

**해결:**
이 경우 RBAC 설정이 올바르지 않을 수 있습니다. 강사에게 NetworkPolicy 및 RBAC 설정 확인을 요청하세요.

---

## ✅ 완료 체크리스트

### 사전 준비
- [ ] AWS CLI 설치 및 자격 증명 설정
- [ ] kubectl 설치 및 EKS 클러스터 연결
- [ ] 환경 변수 설정 (USER_NUM, NAMESPACE, S3_BUCKET)
- [ ] GitHub 저장소 클론

### Part 1: Kubeflow Tenant
- [ ] Namespace 존재 확인 (`kubeflow-user{XX}`)
- [ ] Profile 및 Owner Email 확인 (`user{XX}@mlops.local`)
- [ ] ServiceAccount 확인 (`default-editor`, `default-viewer`)
- [ ] ResourceQuota 확인
- [ ] 권한 격리 테스트 통과

### Part 2: MLflow 환경
- [ ] MLflow Server 실행 중
- [ ] PostgreSQL 실행 중
- [ ] MLflow PodDefault 존재 (`access-mlflow`)
- [ ] Pipeline PodDefault 존재 (`access-ml-pipeline`)
- [ ] MLflow UI 포트 포워딩 테스트

### Part 3: AWS 스토리지
- [ ] S3 버킷 존재 (`mlops-training-user{XX}`)
- [ ] ECR 레지스트리 확인
- [ ] ECR 로그인 성공

---

## 📚 다음 단계

모든 검증을 통과했다면 다음 실습으로 진행하세요:

**➡️ Lab 1-2: Hello World Pipeline**

```bash
cd ../lab1-2_hello-pipeline
```

Lab 1-2에서는 Kubeflow Pipelines를 사용하여 첫 번째 ML 파이프라인을 작성합니다.

---

## 📞 지원

문제 발생 시 강사에게 다음 정보를 전달하세요:

1. **사용자 번호** (예: 07)
2. **검증 스크립트 실행 결과** 캡처
3. **오류 메시지** 전문
4. **실행한 명령어**

---

## 📖 참고 자료

- [Kubeflow 공식 문서](https://www.kubeflow.org/docs/)
- [MLflow 공식 문서](https://mlflow.org/docs/latest/index.html)
- [AWS EKS 사용자 가이드](https://docs.aws.amazon.com/eks/latest/userguide/)
- [AWS S3 문서](https://docs.aws.amazon.com/s3/)
- [AWS ECR 문서](https://docs.aws.amazon.com/ecr/)

---

© 2025 현대오토에버 MLOps Training
