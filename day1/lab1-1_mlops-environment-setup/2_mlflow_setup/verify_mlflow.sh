#!/bin/bash

# ============================================================
# Lab 1-1: MLflow Tenant 검증 스크립트
# ============================================================
# 
# 이 스크립트는 수강생 본인의 MLflow 관련 환경이 올바르게 
# 설정되었는지 확인합니다.
#
# 검증 항목:
#   1. Kubeflow Profile & Namespace
#   2. S3 버킷 (mlops-training-userXX)
#   3. ECR 레지스트리
#   4. MLflow PodDefault
#   5. MLflow Tracking Server 연결
#   6. 권한 격리 테스트
#
# 사용법:
#   export USER_NUM="07"  # 본인의 번호
#   ./verify_mlflow_tenant.sh
#
# ============================================================

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "============================================================"
echo "  Lab 1-1: MLflow Tenant 검증"
echo "============================================================"

# ============================================================
# 환경 변수 확인 및 설정
# ============================================================

if [ -z "$USER_NUM" ]; then
    echo ""
    echo -e "${YELLOW}⚠️  USER_NUM 환경 변수가 설정되지 않았습니다.${NC}"
    echo ""
    read -p "사용자 번호를 입력하세요 (예: 01, 02, 03...): " USER_NUM
    export USER_NUM
fi

# 두 자리 숫자로 변환 (01, 02 형식)
USER_NUM=$(printf "%02d" $USER_NUM 2>/dev/null)

if [ -z "$USER_NUM" ] || [ "$USER_NUM" == "00" ]; then
    echo -e "${RED}❌ 올바른 사용자 번호를 입력하세요.${NC}"
    exit 1
fi

# 환경 변수 설정
NAMESPACE="kubeflow-user${USER_NUM}"
PROFILE_NAME="kubeflow-user${USER_NUM}"
USER_EMAIL="user${USER_NUM}@mlops.local"
S3_BUCKET="mlops-training-user${USER_NUM}"
ECR_REPO_PREFIX="mlops-training/user${USER_NUM}"
AWS_REGION="${AWS_REGION:-ap-northeast-2}"

# 강사/수강생 구분
if [ "$USER_NUM" == "20" ]; then
    USER_ROLE="강사 👨‍🏫"
else
    USER_ROLE="수강생 👨‍🎓"
fi

echo ""
echo -e "${CYAN}📋 검증 정보:${NC}"
echo "   ┌────────────────────────────────────────────────────┐"
echo "   │ 👤 사용자 번호    │ ${USER_NUM} (${USER_ROLE})            "
echo "   │ 📧 이메일         │ ${USER_EMAIL}                      "
echo "   │ 📁 Namespace      │ ${NAMESPACE}                       "
echo "   │ 📋 Profile        │ ${PROFILE_NAME}                    "
echo "   │ 🪣 S3 버킷        │ ${S3_BUCKET}                       "
echo "   │ 🐳 ECR 접두사     │ ${ECR_REPO_PREFIX}                 "
echo "   │ 🌏 AWS 리전       │ ${AWS_REGION}                      "
echo "   └────────────────────────────────────────────────────┘"
echo ""

# 검증 결과 카운터
pass=0
fail=0
warn=0

# ============================================================
# Step 1: Kubeflow Namespace 확인
# ============================================================

echo ""
echo "============================================================"
echo "[1/7] Kubeflow Namespace 확인"
echo "============================================================"

if kubectl get namespace $NAMESPACE > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Namespace 존재: ${NAMESPACE}${NC}"
    
    # Namespace 생성 날짜 확인
    CREATION_TIME=$(kubectl get namespace $NAMESPACE -o jsonpath='{.metadata.creationTimestamp}')
    echo "   생성 날짜: ${CREATION_TIME}"
    
    # istio-injection 레이블 확인
    ISTIO_LABEL=$(kubectl get namespace $NAMESPACE -o jsonpath='{.metadata.labels.istio-injection}' 2>/dev/null)
    if [ "$ISTIO_LABEL" == "enabled" ]; then
        echo -e "   Istio Injection: ${GREEN}enabled${NC}"
    else
        echo -e "   Istio Injection: ${YELLOW}disabled${NC}"
    fi
    
    ((pass++))
else
    echo -e "${RED}❌ Namespace를 찾을 수 없습니다: ${NAMESPACE}${NC}"
    echo ""
    echo -e "${YELLOW}💡 해결 방법:${NC}"
    echo "   강사에게 문의하여 Namespace 생성을 요청하세요."
    ((fail++))
fi

# ============================================================
# Step 2: Kubeflow Profile 확인
# ============================================================

echo ""
echo "============================================================"
echo "[2/7] Kubeflow Profile 확인"
echo "============================================================"

if kubectl get profile $PROFILE_NAME > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Profile 존재: ${PROFILE_NAME}${NC}"
    
    # Profile Owner 확인
    OWNER_EMAIL=$(kubectl get profile $PROFILE_NAME -o jsonpath='{.spec.owner.name}' 2>/dev/null)
    echo "   Owner: ${OWNER_EMAIL}"
    
    # Owner email 일치 확인
    if [ "$OWNER_EMAIL" == "$USER_EMAIL" ]; then
        echo -e "   ${GREEN}✓ Owner email 일치${NC}"
    else
        echo -e "${YELLOW}   ⚠️  Owner email 불일치!${NC}"
        echo "      예상: ${USER_EMAIL}"
        echo "      실제: ${OWNER_EMAIL}"
        echo ""
        echo -e "${YELLOW}💡 해결 방법:${NC}"
        echo "   강사에게 Profile Owner 수정을 요청하세요."
        ((warn++))
    fi
    
    # ResourceQuota 확인
    CPU_QUOTA=$(kubectl get profile $PROFILE_NAME -o jsonpath='{.spec.resourceQuotaSpec.hard.cpu}' 2>/dev/null)
    MEM_QUOTA=$(kubectl get profile $PROFILE_NAME -o jsonpath='{.spec.resourceQuotaSpec.hard.memory}' 2>/dev/null)
    echo "   리소스 할당: CPU ${CPU_QUOTA:-N/A}, Memory ${MEM_QUOTA:-N/A}"
    
    ((pass++))
else
    echo -e "${RED}❌ Profile을 찾을 수 없습니다: ${PROFILE_NAME}${NC}"
    echo ""
    echo -e "${YELLOW}💡 해결 방법:${NC}"
    echo "   강사에게 문의하여 Profile 생성을 요청하세요."
    ((fail++))
fi

# ============================================================
# Step 3: AWS S3 버킷 확인
# ============================================================

echo ""
echo "============================================================"
echo "[3/7] AWS S3 버킷 확인"
echo "============================================================"

# AWS CLI 사용 가능 확인
if ! command -v aws &> /dev/null; then
    echo -e "${YELLOW}⚠️  AWS CLI가 설치되어 있지 않습니다.${NC}"
    echo "   S3 버킷 검증을 건너뜁니다."
    ((warn++))
else
    # AWS 자격증명 확인
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${YELLOW}⚠️  AWS 자격증명이 구성되지 않았습니다.${NC}"
        echo "   aws configure 명령으로 자격증명을 설정하세요."
        ((warn++))
    else
        # S3 버킷 존재 확인
        if aws s3 ls "s3://${S3_BUCKET}" --region ${AWS_REGION} &> /dev/null; then
            echo -e "${GREEN}✅ S3 버킷 존재: ${S3_BUCKET}${NC}"
            
            # 버킷 리전 확인
            BUCKET_REGION=$(aws s3api get-bucket-location --bucket ${S3_BUCKET} --query 'LocationConstraint' --output text 2>/dev/null)
            [ "$BUCKET_REGION" == "None" ] && BUCKET_REGION="us-east-1"
            [ -z "$BUCKET_REGION" ] && BUCKET_REGION="${AWS_REGION}"
            echo "   리전: ${BUCKET_REGION}"
            
            # MLflow Artifacts 폴더 확인
            if aws s3 ls "s3://${S3_BUCKET}/mlflow-artifacts/" --region ${AWS_REGION} &> /dev/null; then
                ARTIFACT_COUNT=$(aws s3 ls "s3://${S3_BUCKET}/mlflow-artifacts/" --region ${AWS_REGION} 2>/dev/null | wc -l)
                echo -e "   ${GREEN}✓ MLflow Artifacts 폴더 존재 (${ARTIFACT_COUNT}개 항목)${NC}"
            else
                echo -e "   ${YELLOW}⚠️  MLflow Artifacts 폴더 없음 (첫 실험 후 자동 생성)${NC}"
            fi
            
            # 버전 관리 상태 확인
            VERSIONING=$(aws s3api get-bucket-versioning --bucket ${S3_BUCKET} --region ${AWS_REGION} --query 'Status' --output text 2>/dev/null)
            echo "   버전 관리: ${VERSIONING:-Disabled}"
            
            ((pass++))
        else
            echo -e "${RED}❌ S3 버킷을 찾을 수 없습니다: ${S3_BUCKET}${NC}"
            echo ""
            echo -e "${YELLOW}💡 해결 방법:${NC}"
            echo "   강사에게 S3 버킷 생성을 요청하거나 직접 생성:"
            echo "   aws s3 mb s3://${S3_BUCKET} --region ${AWS_REGION}"
            ((fail++))
        fi
    fi
fi

# ============================================================
# Step 4: AWS ECR 레지스트리 확인
# ============================================================

echo ""
echo "============================================================"
echo "[4/7] AWS ECR 레지스트리 확인"
echo "============================================================"

if ! command -v aws &> /dev/null; then
    echo -e "${YELLOW}⚠️  AWS CLI가 설치되어 있지 않습니다.${NC}"
    ((warn++))
else
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${YELLOW}⚠️  AWS 자격증명이 구성되지 않았습니다.${NC}"
        ((warn++))
    else
        AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
        ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
        
        echo "   ECR 레지스트리: ${ECR_REGISTRY}"
        echo ""
        
        # 사용자별 ECR 레지스트리 확인
        ECR_IRIS_REPO="${ECR_REPO_PREFIX}/iris-api"
        
        if aws ecr describe-repositories --repository-names ${ECR_IRIS_REPO} --region ${AWS_REGION} &> /dev/null; then
            echo -e "${GREEN}✅ ECR 레지스트리 존재: ${ECR_IRIS_REPO}${NC}"
            
            # 이미지 수 확인
            IMAGE_COUNT=$(aws ecr list-images --repository-name ${ECR_IRIS_REPO} --region ${AWS_REGION} --query 'imageIds | length(@)' --output text 2>/dev/null)
            echo "   이미지 수: ${IMAGE_COUNT:-0}개"
            
            ((pass++))
        else
            echo -e "${YELLOW}⚠️  ECR 레지스트리 없음: ${ECR_IRIS_REPO}${NC}"
            echo "   (실습 진행 시 자동 생성될 수 있습니다)"
            ((warn++))
        fi
        
        # 공용 ECR 레지스트리 확인
        ECR_SHARED_REPO="ml-model-california-housing"
        if aws ecr describe-repositories --repository-names ${ECR_SHARED_REPO} --region ${AWS_REGION} &> /dev/null; then
            echo -e "${GREEN}✅ 공용 ECR 레지스트리 존재: ${ECR_SHARED_REPO}${NC}"
        else
            echo -e "${YELLOW}⚠️  공용 ECR 레지스트리 없음: ${ECR_SHARED_REPO}${NC}"
        fi
    fi
fi

# ============================================================
# Step 5: MLflow PodDefault 확인
# ============================================================

echo ""
echo "============================================================"
echo "[5/7] MLflow PodDefault 확인"
echo "============================================================"

# PodDefault 확인
MLFLOW_PD=$(kubectl get poddefault access-mlflow -n $NAMESPACE -o name 2>/dev/null)
PIPELINE_PD=$(kubectl get poddefault access-ml-pipeline -n $NAMESPACE -o name 2>/dev/null)

if [ -n "$MLFLOW_PD" ]; then
    echo -e "${GREEN}✅ MLflow PodDefault 존재: access-mlflow${NC}"
    
    # MLflow Tracking URI 확인
    MLFLOW_URI=$(kubectl get poddefault access-mlflow -n $NAMESPACE -o jsonpath='{.spec.env[?(@.name=="MLFLOW_TRACKING_URI")].value}' 2>/dev/null)
    echo "   MLFLOW_TRACKING_URI: ${MLFLOW_URI:-N/A}"
    
    ((pass++))
else
    echo -e "${RED}❌ MLflow PodDefault 없음: access-mlflow${NC}"
    echo ""
    echo -e "${YELLOW}💡 해결 방법:${NC}"
    echo "   강사에게 PodDefault 생성을 요청하세요."
    ((fail++))
fi

if [ -n "$PIPELINE_PD" ]; then
    echo -e "${GREEN}✅ Pipeline PodDefault 존재: access-ml-pipeline${NC}"
else
    echo -e "${YELLOW}⚠️  Pipeline PodDefault 없음: access-ml-pipeline${NC}"
    ((warn++))
fi

# ============================================================
# Step 6: MLflow Tracking Server 연결 확인
# ============================================================

echo ""
echo "============================================================"
echo "[6/7] MLflow Tracking Server 연결 확인"
echo "============================================================"

# 사용자 네임스페이스의 MLflow Server Pod 확인
MLFLOW_POD=$(kubectl get pods -n $NAMESPACE -l app=mlflow-server -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -n "$MLFLOW_POD" ]; then
    MLFLOW_STATUS=$(kubectl get pods -n $NAMESPACE -l app=mlflow-server -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
    
    if [ "$MLFLOW_STATUS" == "Running" ]; then
        echo -e "${GREEN}✅ MLflow Server 실행 중 (멀티테넌트)${NC}"
        echo "   Pod: ${MLFLOW_POD}"
        echo "   Namespace: ${NAMESPACE}"
        
        # Service 확인
        MLFLOW_SVC=$(kubectl get svc mlflow-server -n $NAMESPACE -o jsonpath='{.metadata.name}' 2>/dev/null)
        if [ -n "$MLFLOW_SVC" ]; then
            MLFLOW_PORT=$(kubectl get svc mlflow-server -n $NAMESPACE -o jsonpath='{.spec.ports[0].port}' 2>/dev/null)
            echo "   Service: mlflow-server:${MLFLOW_PORT}"
            echo ""
            echo "   내부 접속 URL: http://mlflow-server.${NAMESPACE}.svc.cluster.local:${MLFLOW_PORT}"
        fi
        
        ((pass++))
    else
        echo -e "${YELLOW}⚠️  MLflow Server 상태: ${MLFLOW_STATUS}${NC}"
        ((warn++))
    fi
else
    echo -e "${RED}❌ MLflow Server를 찾을 수 없습니다 (Namespace: ${NAMESPACE})${NC}"
    echo ""
    echo -e "${YELLOW}💡 해결 방법:${NC}"
    echo "   강사에게 MLflow 멀티테넌트 배포를 요청하세요."
    ((fail++))
fi

# PostgreSQL 확인 (MLflow Backend)
POSTGRES_POD=$(kubectl get pods -n mlflow-system -l app=postgres -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -n "$POSTGRES_POD" ]; then
    POSTGRES_STATUS=$(kubectl get pods -n mlflow-system -l app=postgres -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
    if [ "$POSTGRES_STATUS" == "Running" ]; then
        echo -e "${GREEN}✅ PostgreSQL 실행 중 (MLflow Backend)${NC}"
    else
        echo -e "${YELLOW}⚠️  PostgreSQL 상태: ${POSTGRES_STATUS}${NC}"
    fi
fi

# ============================================================
# Step 7: 권한 격리 테스트
# ============================================================

echo ""
echo "============================================================"
echo "[7/7] 권한 격리 테스트"
echo "============================================================"

# 자신의 네임스페이스 접근 테스트
echo "  자신의 네임스페이스 접근 테스트..."
if kubectl get pods -n $NAMESPACE > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓ 자신의 네임스페이스 접근 가능: ${NAMESPACE}${NC}"
else
    echo -e "   ${RED}✗ 자신의 네임스페이스 접근 불가${NC}"
    ((fail++))
fi

# 다른 네임스페이스 접근 테스트
if [ "$USER_NUM" == "01" ]; then
    TEST_NS="kubeflow-user02"
else
    TEST_NS="kubeflow-user01"
fi

echo "  다른 네임스페이스 접근 테스트: ${TEST_NS}..."
if kubectl get pods -n $TEST_NS 2>&1 | grep -q "forbidden\|No resources found\|Error"; then
    echo -e "   ${GREEN}✓ 다른 네임스페이스 접근 차단됨 (정상)${NC}"
    ((pass++))
else
    # 실제로 접근 가능한지 다시 확인
    if kubectl get pods -n $TEST_NS > /dev/null 2>&1; then
        echo -e "   ${YELLOW}⚠️  다른 네임스페이스 접근 가능 (권한 확인 필요)${NC}"
        ((warn++))
    else
        echo -e "   ${GREEN}✓ 다른 네임스페이스 접근 차단됨 (정상)${NC}"
        ((pass++))
    fi
fi

# ============================================================
# Tenant 아키텍처 요약
# ============================================================

echo ""
echo "============================================================"
echo "  MLflow Tenant 아키텍처 요약"
echo "============================================================"
echo ""

# 리소스 카운트 수집
POD_COUNT=$(kubectl get pods -n $NAMESPACE --no-headers 2>/dev/null | wc -l)
SERVICE_COUNT=$(kubectl get svc -n $NAMESPACE --no-headers 2>/dev/null | wc -l)
PVC_COUNT=$(kubectl get pvc -n $NAMESPACE --no-headers 2>/dev/null | wc -l)
SECRET_COUNT=$(kubectl get secrets -n $NAMESPACE --no-headers 2>/dev/null | wc -l)
PD_COUNT=$(kubectl get poddefaults -n $NAMESPACE --no-headers 2>/dev/null | wc -l)

cat <<EOF
  ┌─────────────────────────────────────────────────────────┐
  │            MLflow Multi-Tenant Architecture             │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │  👤 사용자: user${USER_NUM} (${USER_ROLE})                       
  │                                                         │
  │  📦 Kubernetes 리소스:                                  │
  │     ├─ Namespace: ${NAMESPACE}                          
  │     ├─ Profile: ${PROFILE_NAME}                         
  │     ├─ Pods: ${POD_COUNT}개                                      
  │     ├─ Services: ${SERVICE_COUNT}개                              
  │     ├─ PVCs: ${PVC_COUNT}개                                      
  │     ├─ Secrets: ${SECRET_COUNT}개                                
  │     └─ PodDefaults: ${PD_COUNT}개                                
  │                                                         │
  │  ☁️  AWS 리소스:                                        │
  │     ├─ S3 버킷: ${S3_BUCKET}                            
  │     └─ ECR: ${ECR_REPO_PREFIX}                        
  │                                                         │
  │  🔗 MLflow 연결:                                        │
  │     └─ http://mlflow-server.mlflow-system:5000          │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
EOF

# ============================================================
# 검증 결과 요약
# ============================================================

echo ""
echo "============================================================"
echo "  검증 결과 요약"
echo "============================================================"
echo ""

total=$((pass + fail + warn))
echo "   ✅ 통과: ${pass}"
echo "   ❌ 실패: ${fail}"
echo "   ⚠️  경고: ${warn}"
echo "   📊 총점: ${pass}/${total}"
echo ""

# 결과에 따른 메시지
if [ $fail -eq 0 ] && [ $warn -eq 0 ]; then
    echo -e "${GREEN}🎉 모든 검증을 완벽하게 통과했습니다!${NC}"
    echo ""
    echo "   다음 단계: Jupyter Notebook을 생성하여 MLflow 실험을 시작하세요."
    READY_FOR_LAB=true
    
elif [ $fail -eq 0 ]; then
    echo -e "${GREEN}✅ 필수 검증을 통과했습니다!${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  일부 경고 사항이 있지만 실습 진행에는 문제없습니다.${NC}"
    echo "   다음 단계: Jupyter Notebook을 생성하여 MLflow 실험을 시작하세요."
    READY_FOR_LAB=true
    
else
    echo -e "${RED}❌ 일부 필수 항목이 실패했습니다.${NC}"
    echo ""
    echo "   강사에게 문의하여 환경 설정을 완료하세요."
    READY_FOR_LAB=false
fi

# ============================================================
# 실습 가능 여부 최종 판단
# ============================================================

echo ""
echo "============================================================"
echo "  실습 가능 여부"
echo "============================================================"
echo ""

echo "  필수 요구사항:"
echo -n "   ✓ Namespace 존재: "
kubectl get namespace $NAMESPACE > /dev/null 2>&1 && echo -e "${GREEN}Yes${NC}" || echo -e "${RED}No${NC}"

echo -n "   ✓ Profile 존재: "
kubectl get profile $PROFILE_NAME > /dev/null 2>&1 && echo -e "${GREEN}Yes${NC}" || echo -e "${RED}No${NC}"

echo -n "   ✓ MLflow PodDefault: "
kubectl get poddefault access-mlflow -n $NAMESPACE > /dev/null 2>&1 && echo -e "${GREEN}Yes${NC}" || echo -e "${RED}No${NC}"

echo -n "   ✓ MLflow Server 실행: "
kubectl get pods -n mlflow-system -l app=mlflow-server --field-selector=status.phase=Running --no-headers 2>/dev/null | grep -q . && echo -e "${GREEN}Yes${NC}" || echo -e "${RED}No${NC}"

echo ""
if [ "$READY_FOR_LAB" = true ]; then
    echo -e "${GREEN}  ✅ 실습 진행 가능!${NC}"
    echo ""
    echo "  📝 다음 단계:"
    echo "     1. Kubeflow Dashboard에서 Notebook 생성"
    echo "     2. Notebook에서 MLflow 실험 진행"
    echo "     3. S3에 모델 아티팩트 저장 확인"
else
    echo -e "${RED}  ❌ 실습 진행 불가${NC}"
    echo ""
    echo "  강사에게 다음 정보를 전달하세요:"
    echo "     - 사용자 번호: ${USER_NUM}"
    echo "     - Namespace: ${NAMESPACE}"
    echo "     - 검증 결과: 통과 ${pass}, 실패 ${fail}, 경고 ${warn}"
fi

echo ""
echo "============================================================"
echo ""

exit $fail
