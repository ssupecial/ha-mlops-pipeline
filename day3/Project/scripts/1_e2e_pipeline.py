#!/usr/bin/env python3
"""
Day 3 Project: E2E ML Pipeline
================================
California Housing 데이터셋을 사용한 End-to-End MLOps 파이프라인

⚠️ 중요: 실행 전 환경 변수 설정 필요!
   export USER_NUM="01"  # 본인 번호로 변경

파이프라인 단계:
1. Data Load - California Housing 데이터 로드
2. Preprocess - 데이터 전처리
3. Feature Engineering - 피처 엔지니어링
4. Train Model - MLflow로 모델 학습 및 추적
5. Evaluate - 모델 평가
6. Deploy - KServe로 모델 배포 (조건부)
"""

import os
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from kfp import compiler

# ============================================================
# 환경 변수 기반 동적 설정
# ============================================================
USER_NUM = os.getenv('USER_NUM', '01')
DEFAULT_NAMESPACE = f"kubeflow-user{USER_NUM}"
DEFAULT_EXPERIMENT_NAME = f"e2e-pipeline-user{USER_NUM}"
DEFAULT_MODEL_NAME = f"california-model-user{USER_NUM}"

print(f"[설정] USER_NUM: {USER_NUM}")
print(f"[설정] DEFAULT_NAMESPACE: {DEFAULT_NAMESPACE}")

# ============================================================
# Components
# ============================================================

@component(
    base_image="python:3.9-slim",
    packages_to_install=["scikit-learn", "pandas", "numpy"]
)
def load_data(
    data_source: str,
    output_data: Output[Dataset]
):
    """Step 1: 데이터 로드"""
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    import json
    
    print(f"📥 데이터 소스: {data_source}")
    
    if data_source == "sklearn":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    else:
        raise ValueError(f"Unknown data source: {data_source}")
    
    print(f"✅ 데이터 로드 완료: {df.shape}")
    df.to_parquet(output_data.path)


@component(
    base_image="python:3.9-slim",
    packages_to_install=["scikit-learn", "pandas", "numpy"]
)
def preprocess(
    input_data: Input[Dataset],
    X_train_out: Output[Dataset],
    X_test_out: Output[Dataset],
    y_train_out: Output[Dataset],
    y_test_out: Output[Dataset]
):
    """Step 2: 데이터 전처리"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df = pd.read_parquet(input_data.path)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
    
    X_train.to_parquet(X_train_out.path)
    X_test.to_parquet(X_test_out.path)
    pd.DataFrame(y_train).to_parquet(y_train_out.path)
    pd.DataFrame(y_test).to_parquet(y_test_out.path)


@component(
    base_image="python:3.9-slim",
    packages_to_install=["scikit-learn", "pandas", "numpy"]
)
def feature_engineering(
    X_train_in: Input[Dataset],
    X_test_in: Input[Dataset],
    X_train_out: Output[Dataset],
    X_test_out: Output[Dataset]
):
    """Step 3: 피처 엔지니어링"""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import pickle
    
    X_train = pd.read_parquet(X_train_in.path)
    X_test = pd.read_parquet(X_test_in.path)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_parquet(X_train_out.path)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_parquet(X_test_out.path)
    
    print(f"✅ 피처 스케일링 완료")


@component(
    base_image="python:3.9-slim",
    packages_to_install=["scikit-learn", "pandas", "numpy", "mlflow", "boto3"]
)
def train_model(
    X_train: Input[Dataset],
    X_test: Input[Dataset],
    y_train: Input[Dataset],
    y_test: Input[Dataset],
    mlflow_tracking_uri: str,
    experiment_name: str,
    n_estimators: int,
    max_depth: int
) -> str:
    """Step 4: 모델 학습 (MLflow 추적)"""
    import pandas as pd
    import mlflow
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    X_tr = pd.read_parquet(X_train.path)
    X_te = pd.read_parquet(X_test.path)
    y_tr = pd.read_parquet(y_train.path).values.ravel()
    y_te = pd.read_parquet(y_test.path).values.ravel()
    
    with mlflow.start_run() as run:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        
        y_pred = model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2 = r2_score(y_te, y_pred)
        
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth
        })
        mlflow.log_metrics({"rmse": rmse, "r2": r2})
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✅ 모델 학습 완료")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R2: {r2:.4f}")
        print(f"   Run ID: {run.info.run_id}")
        
        return run.info.run_id


@component(
    base_image="python:3.9-slim",
    packages_to_install=["mlflow", "boto3"]
)
def evaluate_model(
    run_id: str,
    mlflow_tracking_uri: str,
    r2_threshold: float
) -> str:
    """Step 5: 모델 평가 및 배포 결정"""
    import mlflow
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    run = mlflow.get_run(run_id)
    r2 = run.data.metrics.get("r2", 0)
    
    print(f"📊 R2 Score: {r2:.4f}")
    print(f"📊 Threshold: {r2_threshold}")
    
    if r2 >= r2_threshold:
        print("✅ 배포 조건 충족!")
        return "deploy"
    else:
        print("⚠️ 배포 조건 미충족")
        return "skip"


@component(
    base_image="python:3.9-slim",
    packages_to_install=["kubernetes", "mlflow", "boto3"]
)
def deploy_model(
    run_id: str,
    model_name: str,
    namespace: str,
    mlflow_tracking_uri: str
):
    """Step 6: KServe로 모델 배포"""
    import mlflow
    from kubernetes import client, config
    
    print(f"🚀 모델 배포 시작")
    print(f"   Model: {model_name}")
    print(f"   Namespace: {namespace}")
    print(f"   Run ID: {run_id}")
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    run = mlflow.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    
    print(f"   Artifact URI: {artifact_uri}")
    print(f"✅ 배포 완료 (시뮬레이션)")


@component(base_image="python:3.9-slim")
def send_alert(run_id: str, message: str):
    """알림 전송"""
    print(f"⚠️ Alert: {message}")
    print(f"   Run ID: {run_id}")


# ============================================================
# Pipeline Definition
# ============================================================

@dsl.pipeline(
    name="E2E ML Pipeline",
    description="End-to-End ML Pipeline with MLflow and KServe"
)
def e2e_pipeline(
    data_source: str = "sklearn",
    mlflow_tracking_uri: str = "http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000",
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    namespace: str = DEFAULT_NAMESPACE,  # ⚠️ 동적 기본값!
    n_estimators: int = 100,
    max_depth: int = 10,
    r2_threshold: float = 0.75
):
    """
    E2E ML Pipeline
    
    Parameters:
    - data_source: 데이터 소스 (sklearn)
    - mlflow_tracking_uri: MLflow 서버 URI
    - experiment_name: MLflow 실험 이름
    - model_name: 배포할 모델 이름
    - namespace: Kubernetes 네임스페이스 (⚠️ 본인 것으로 변경!)
    - n_estimators: RandomForest estimator 수
    - max_depth: 트리 최대 깊이
    - r2_threshold: 배포 결정 임계값
    """
    
    # Step 1: 데이터 로드
    load_task = load_data(data_source=data_source)
    
    # Step 2: 전처리
    preprocess_task = preprocess(
        input_data=load_task.outputs["output_data"]
    )
    
    # Step 3: 피처 엔지니어링
    feature_task = feature_engineering(
        X_train_in=preprocess_task.outputs["X_train_out"],
        X_test_in=preprocess_task.outputs["X_test_out"]
    )
    
    # Step 4: 모델 학습
    train_task = train_model(
        X_train=feature_task.outputs["X_train_out"],
        X_test=feature_task.outputs["X_test_out"],
        y_train=preprocess_task.outputs["y_train_out"],
        y_test=preprocess_task.outputs["y_test_out"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    
    # Step 5: 평가
    evaluate_task = evaluate_model(
        run_id=train_task.output,
        mlflow_tracking_uri=mlflow_tracking_uri,
        r2_threshold=r2_threshold
    )
    
    # Step 6: 조건부 배포
    with dsl.If(evaluate_task.output == "deploy"):
        deploy_model(
            run_id=train_task.output,
            model_name=model_name,
            namespace=namespace,
            mlflow_tracking_uri=mlflow_tracking_uri
        )
    
    with dsl.If(evaluate_task.output == "skip"):
        send_alert(
            run_id=train_task.output,
            message="Model performance below threshold"
        )


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  E2E ML Pipeline Compiler")
    print("=" * 60)
    
    pipeline_file = "e2e_pipeline.yaml"
    
    compiler.Compiler().compile(
        pipeline_func=e2e_pipeline,
        package_path=pipeline_file
    )
    
    print(f"\n✅ Pipeline compiled: {pipeline_file}")
    print(f"\n📋 Default Parameters:")
    print(f"  - data_source: sklearn")
    print(f"  - experiment_name: {DEFAULT_EXPERIMENT_NAME}")
    print(f"  - model_name: {DEFAULT_MODEL_NAME}")
    print(f"  - namespace: {DEFAULT_NAMESPACE}")
    print(f"  - n_estimators: 100")
    print(f"  - max_depth: 10")
    print(f"  - r2_threshold: 0.75")
    print(f"\n⚠️  Important:")
    print(f"  namespace 파라미터가 현재 USER_NUM={USER_NUM} 기준으로 설정됨")
    print(f"  다른 사용자라면 export USER_NUM='XX'로 변경 후 다시 컴파일하세요!")
    print(f"\n🚀 Next: Upload {pipeline_file} to Kubeflow UI")
    print("=" * 60)
