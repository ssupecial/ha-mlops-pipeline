#!/usr/bin/env python3
"""
Lab 2-2: MLflow 실험 추적
"""
import numpy as np
import pandas as pd
import os
import boto3

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

# 환경 변수 읽기
USER_NUM = os.getenv('USER_NUM', '01')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-2')
NAMESPACE = f"kubeflow-user{USER_NUM}"

# 환경 변수 설정
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_DEFAULT_REGION"] = AWS_REGION

# MLflow 설정
mlflow.set_tracking_uri("http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000")

try:
    s3_client = boto3.client('s3', region_name=AWS_REGION)

    S3_BUCKET = f"mlops-training-user{USER_NUM}"

    print(f"✅ USER_NUM: {USER_NUM}")
    print(f"✅ NAMESPACE: {NAMESPACE}")
    print(f"✅ AWS Region: {AWS_REGION}")
    print(f"✅ S3 Bucket: {S3_BUCKET}")
    
    # S3 버킷 접근 테스트
    s3_client.head_bucket(Bucket=S3_BUCKET)
    
    print("="*60)
    print("  AWS Credentials 설정 완료!")
    print("="*60)
    print(f"✅ AWS Access Key: {AWS_ACCESS_KEY_ID[:4]}****")
    print(f"✅ S3 접근 테스트: 성공!")
    print("")
    print("이제 3.2 실험을 실행할 수 있습니다!")
    
except Exception as e:
    print(f"❌ 오류: {e}")
    print(f"")
    print(f"해결 방법:")
    print(f"1. AWS_ACCESS_KEY_ID와 AWS_SECRET_ACCESS_KEY를 정확히 입력하세요")
    print(f"2. 강사가 제공한 자격 증명이 맞는지 확인하세요")
    print(f"3. S3 버킷({S3_BUCKET})에 접근 권한이 있는지 확인하세요")

# MLflow 클라이언트 생성
client = MlflowClient()

# ✅ 사용자별 고유한 실험 이름 사용!
EXPERIMENT_NAME = f"california-housing-user{USER_NUM}"

# 사용자별 S3 artifact location 설정
artifact_location = f"s3://{S3_BUCKET}/mlflow-artifacts"

# 실험이 이미 존재하는지 확인
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    # 실험이 없으면 생성
    experiment_id = client.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location=artifact_location
    )
    print(f"\n✅ 새 실험 생성: {EXPERIMENT_NAME}")
    experiment = client.get_experiment(experiment_id)
else:
    print(f"\n✅ 기존 실험 사용: {EXPERIMENT_NAME}")
    experiment_id = experiment.experiment_id

# MLflow에 실험 설정
mlflow.set_experiment(EXPERIMENT_NAME)

# 결과 출력
experiment = client.get_experiment(experiment_id)
print(f"\n📋 실험 정보:")
print(f"   Name: {experiment.name}")
print(f"   ID: {experiment.experiment_id}")
print(f"   Artifact Location: {experiment.artifact_location}")
print(f"   Lifecycle Stage: {experiment.lifecycle_stage}")

# 검증
if S3_BUCKET in experiment.artifact_location and experiment.lifecycle_stage == "active":
    print(f"\n🎉 SUCCESS: 실험이 올바르게 설정되었습니다!")
else:
    print(f"\n❌ ERROR: 문제가 있습니다!")

# 데이터 로드
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

print("="*60)
print("  California Housing 실험")
print("="*60)
print()

# 실험 1: LinearRegression
print("[1/2] LinearRegression 학습 중...")
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
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mae", mae)
    
    # 5. 모델 저장
    mlflow.sklearn.log_model(model, "model")
    
    # 6. 태그 추가
    mlflow.set_tag("stage", "baseline")
    mlflow.set_tag("author", "mlops-training")
    
    # 결과 출력
    print(f"\n📊 Results:")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R2 Score: {r2:.4f}")
    print(f"  - MAE: {mae:.4f}")
    print(f"\n✅ Run ID: {mlflow.active_run().info.run_id}")

# 실험 2: RandomForest
print("[2/2] RandomForest 학습 중...")
with mlflow.start_run(run_name="rf-baseline"):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    
    mlflow.log_params({
        "model_type": "RandomForest",
        "n_estimators": 100,
        "max_depth": 10
    })
    mlflow.log_metrics({"r2_score": r2})
    mlflow.sklearn.log_model(model, "model")
    
    print(f"  R2 Score: {r2:.4f}")
    print()

print("="*60)
print("✅ 실험 완료!")
print()
print("MLflow UI 확인:")
print("kubectl port-forward svc/mlflow-server-service -n mlflow-system 5000:5000")
print("브라우저: http://localhost:5000")
print("="*60)
