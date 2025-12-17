# 🔧 트러블슈팅 가이드

## 📋 목차

1. [AWS/EKS 관련](#1-awseks-관련)
2. [Kubeflow 관련](#2-kubeflow-관련)
3. [Pipeline 관련](#3-pipeline-관련)
4. [MLflow 관련](#4-mlflow-관련)
5. [KServe 관련](#5-kserve-관련)
6. [Docker 관련](#6-docker-관련)

---

## 1. AWS/EKS 관련

### ❌ "Unable to locate credentials"

**원인**: AWS CLI 자격 증명이 설정되지 않음

**해결**:
```bash
# 자격 증명 확인
cat ~/.aws/credentials

# 자격 증명 설정
aws configure
```

### ❌ "error: You must be logged in to the server"

**원인**: kubeconfig가 설정되지 않음 또는 만료됨

**해결**:
```bash
# kubeconfig 업데이트
aws eks update-kubeconfig \
    --region ap-northeast-2 \
    --name mlops-training-cluster

# 컨텍스트 확인
kubectl config current-context
```

### ❌ "Error: Kubernetes cluster unreachable"

**원인**: 클러스터 연결 문제

**해결**:
```bash
# 1. 인터넷 연결 확인
ping google.com

# 2. AWS 자격 증명 확인
aws sts get-caller-identity

# 3. kubeconfig 재설정
aws eks update-kubeconfig --name mlops-training-cluster --region ap-northeast-2

# 4. 클러스터 상태 확인
aws eks describe-cluster --name mlops-training-cluster --region ap-northeast-2
```

---

## 2. Kubeflow 관련

### ❌ "Connection refused" (localhost:8080)

**원인**: 포트 포워딩이 실행되지 않음

**해결**:
```bash
# 포트 포워딩 실행
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80

# 백그라운드로 실행
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80 &
```

### ❌ "403 Forbidden"

**원인**: 네임스페이스 접근 권한 없음

**해결**:
```bash
# 권한 확인
kubectl auth can-i get pods -n kubeflow-userXX

# Profile 확인 (Kubeflow)
kubectl get profiles
```

### ❌ Notebook 생성 시 "Pending" 지속

**원인**: 리소스 부족 또는 스토리지 문제

**해결**:
```bash
# Pod 상태 확인
kubectl get pods -n kubeflow-userXX

# 상세 이벤트 확인
kubectl describe pod notebook-userXX-0 -n kubeflow-userXX

# 노드 리소스 확인
kubectl describe nodes | grep -A 5 "Allocated resources"
```

---

## 3. Pipeline 관련

### ❌ "ModuleNotFoundError: No module named 'kfp'"

**원인**: KFP SDK가 설치되지 않음

**해결**:
```bash
pip install kfp==1.8.22
```

### ❌ Pipeline 실행 시 "Pending" 상태 지속

**원인**: Pod 스케줄링 문제

**해결**:
```bash
# Pipeline Pod 확인
kubectl get pods -n kubeflow-userXX | grep pipeline

# 이벤트 확인
kubectl describe pod [pod-name] -n kubeflow-userXX

# 로그 확인
kubectl logs [pod-name] -n kubeflow-userXX
```

### ❌ "ImagePullBackOff"

**원인**: 컨테이너 이미지를 가져올 수 없음

**해결**:
```bash
# 이미지 경로 확인
kubectl describe pod [pod-name] -n kubeflow-userXX | grep Image

# ECR 이미지 존재 확인
aws ecr describe-images --repository-name [repo-name]

# ECR 로그인 (필요한 경우)
aws ecr get-login-password --region ap-northeast-2 | \
    docker login --username AWS --password-stdin [ECR_URI]
```

### ❌ 컴포넌트 간 데이터 전달 실패

**원인**: 출력 참조 방식 오류

**해결**:
```python
# ❌ 잘못된 방법
step2 = component_b(input=step1)

# ✅ 올바른 방법
step2 = component_b(input=step1.output)
```

### ❌ "CrashLoopBackOff"

**원인**: 컨테이너 실행 중 오류 발생

**해결**:
```bash
# 이전 로그 확인
kubectl logs [pod-name] -n kubeflow-userXX --previous

# 컨테이너 상태 확인
kubectl describe pod [pod-name] -n kubeflow-userXX
```

---

## 4. MLflow 관련

### ❌ "MLFLOW_TRACKING_URI not set"

**원인**: MLflow 서버 URI가 설정되지 않음

**해결**:
```python
import os
import mlflow

# 환경 변수 설정
os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000'

# 또는 직접 설정
mlflow.set_tracking_uri('http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000')
```

### ❌ MLflow S3 연결 오류

**원인**: S3/MinIO 자격 증명 문제

**해결**:
```python
import os

os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio-service.kubeflow.svc:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
```

### ❌ "Connection refused" (MLflow UI)

**원인**: 포트 포워딩 필요

**해결**:
```bash
kubectl port-forward svc/mlflow-server-service -n mlflow-system 5000:5000
```

---

## 5. KServe 관련

### ❌ InferenceService "READY=False" 지속

**원인**: 다양한 원인 가능

**해결**:
```bash
# 상태 확인
kubectl get isvc [model-name] -n kubeflow-userXX

# 상세 정보 확인
kubectl describe isvc [model-name] -n kubeflow-userXX

# Predictor Pod 확인
kubectl get pods -l serving.kserve.io/inferenceservice=[model-name] -n kubeflow-userXX

# 로그 확인
kubectl logs -l serving.kserve.io/inferenceservice=[model-name] -n kubeflow-userXX
```

### ❌ "storageUri" 접근 실패

**원인**: S3 버킷 권한 또는 경로 문제

**해결**:
```bash
# S3 경로 확인
aws s3 ls s3://[bucket-name]/[model-path]/

# ServiceAccount IAM 역할 확인
kubectl describe sa default -n kubeflow-userXX
```

### ❌ "502 Bad Gateway"

**원인**: Predictor 컨테이너 문제

**해결**:
```bash
# Predictor Pod 로그 확인
kubectl logs -l serving.kserve.io/inferenceservice=[model-name] \
    -c kserve-container -n kubeflow-userXX
```

---

## 6. Docker 관련

### ❌ "Cannot connect to the Docker daemon"

**원인**: Docker 데몬이 실행되지 않음

**해결**:
```bash
# Linux
sudo systemctl start docker

# macOS/Windows
# Docker Desktop 앱 실행
```

### ❌ "permission denied"

**원인**: Docker 그룹 권한 없음

**해결**:
```bash
# Linux
sudo usermod -aG docker $USER
newgrp docker
```

### ❌ ECR 푸시 실패

**원인**: ECR 로그인 만료 또는 권한 없음

**해결**:
```bash
# ECR 재로그인
aws ecr get-login-password --region ap-northeast-2 | \
    docker login --username AWS --password-stdin [ECR_URI]

# 리포지토리 존재 확인
aws ecr describe-repositories --repository-names [repo-name]

# 리포지토리 생성 (없는 경우)
aws ecr create-repository --repository-name [repo-name]
```

---

## 🔍 일반적인 디버깅 명령어

```bash
# Pod 상태 확인
kubectl get pods -n kubeflow-userXX

# Pod 상세 정보
kubectl describe pod [pod-name] -n kubeflow-userXX

# Pod 로그
kubectl logs [pod-name] -n kubeflow-userXX

# 이전 컨테이너 로그 (CrashLoopBackOff 시)
kubectl logs [pod-name] -n kubeflow-userXX --previous

# 모든 리소스 확인
kubectl get all -n kubeflow-userXX

# 이벤트 확인
kubectl get events -n kubeflow-userXX --sort-by=.lastTimestamp

# Pod 내부 접속
kubectl exec -it [pod-name] -n kubeflow-userXX -- /bin/bash
```

---

