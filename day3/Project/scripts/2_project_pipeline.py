"""
조별 프로젝트 템플릿
====================

이 파일을 복사하여 팀 파이프라인을 구축하세요.

사용법:
    1. 파일 복사
       cp 2_project_pipeline.py my_team_pipeline.py
    
    2. 팀 설정 변경
       # macOS
       sed -i '' 's/team-XX/team-01/g' my_team_pipeline.py
       
       # Linux
       sed -i 's/team-XX/team-01/g' my_team_pipeline.py
    
    3. 피처 엔지니어링 구현 (TODO 부분)
    
    4. 컴파일 및 실행
       python my_team_pipeline.py

현대오토에버 MLOps Training
"""

import os
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset
from kfp import compiler


# ============================================================
# ⚠️ TODO: 팀 설정으로 변경하세요!
# ============================================================
TEAM_NAME = "team-XX"                        # 예: team-01, team-02, ...
USER_NAMESPACE = "kubeflow-user-example-com" # 현재 Kubeflow 프로필 네임스페이스

# MLflow 설정 (변경 불필요)
MLFLOW_TRACKING_URI = "http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000"


# ============================================================
# Component 1: 데이터 로드
# ============================================================
@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.2"]
)
def load_data(
    data_source: str,
    output_data: Output[Dataset]
):
    """California Housing 데이터셋 로드"""
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    
    print("=" * 60)
    print("  Step 1: Load Data")
    print("=" * 60)
    
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    print(f"  Source: {data_source}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    df.to_csv(output_data.path, index=False)
    print(f"  ✅ Data saved")


# ============================================================
# Component 2: 전처리
# ============================================================
@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.2", "numpy==1.24.3"]
)
def preprocess(
    input_data: Input[Dataset],
    X_train_out: Output[Dataset],
    X_test_out: Output[Dataset],
    y_train_out: Output[Dataset],
    y_test_out: Output[Dataset],
    test_size: float = 0.2
):
    """데이터 전처리: Train/Test 분할 및 정규화"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("=" * 60)
    print("  Step 2: Preprocess")
    print("=" * 60)
    
    df = pd.read_csv(input_data.path)
    
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    X_train_scaled.to_csv(X_train_out.path, index=False)
    X_test_scaled.to_csv(X_test_out.path, index=False)
    y_train.to_csv(y_train_out.path, index=False)
    y_test.to_csv(y_test_out.path, index=False)
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  ✅ Preprocessing completed")


# ============================================================
# Component 3: 피처 엔지니어링 ⭐ (평가 항목)
# ============================================================
@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas==2.0.3", "numpy==1.24.3"]
)
def feature_engineering(
    X_train_in: Input[Dataset],
    X_test_in: Input[Dataset],
    X_train_out: Output[Dataset],
    X_test_out: Output[Dataset]
) -> int:
    """
    피처 엔지니어링 - 파생 변수 생성
    
    ⭐ TODO: 팀에서 창의적인 피처를 추가하세요! (평가 항목)
    최소 1개 이상의 파생 피처가 필요합니다.
    
    피처 아이디어:
    - bedroom_ratio: 방 대비 침실 비율
    - people_per_household: 가구당 인구
    - dist_to_bay: Bay Area까지 거리
    - income_category: 소득 구간 (binning)
    - density: 밀집도 지표
    - income_rooms: 소득 × 방 수 상호작용
    
    Returns:
        생성된 새 피처 개수
    """
    import pandas as pd
    import numpy as np
    
    print("=" * 60)
    print("  Step 3: Feature Engineering")
    print("=" * 60)
    
    X_train = pd.read_csv(X_train_in.path)
    X_test = pd.read_csv(X_test_in.path)
    
    original_cols = list(X_train.columns)
    
    def add_features(df):
        """
        ⭐ TODO: 여기에 새로운 피처를 추가하세요!
        """
        df = df.copy()
        
        # 예시 1: 방당 침실 비율
        df['bedroom_ratio'] = df['AveBedrms'] / (df['AveRooms'] + 1e-6)
        
        # 예시 2: 가구당 인구
        df['people_per_household'] = df['Population'] / (df['AveOccup'] + 1e-6)
        
        # TODO: 더 많은 피처 추가!
        # 예시 3: Bay Area까지 거리
        # bay_lat, bay_long = 37.77, -122.42
        # df['dist_to_bay'] = np.sqrt(
        #     (df['Latitude'] - bay_lat)**2 + 
        #     (df['Longitude'] - bay_long)**2
        # )
        
        # 예시 4: 소득과 방 수의 상호작용
        # df['income_rooms'] = df['MedInc'] * df['AveRooms']
        
        return df
    
    X_train_fe = add_features(X_train)
    X_test_fe = add_features(X_test)
    
    new_cols = [c for c in X_train_fe.columns if c not in original_cols]
    
    print(f"  Original features: {len(original_cols)}")
    print(f"  New features: {new_cols}")
    print(f"  Total features: {len(X_train_fe.columns)}")
    
    X_train_fe.to_csv(X_train_out.path, index=False)
    X_test_fe.to_csv(X_test_out.path, index=False)
    
    print(f"  ✅ Feature engineering completed")
    
    return len(new_cols)


# ============================================================
# Component 4: 모델 학습 (MLflow 연동)
# ============================================================
@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "pandas==2.0.3",
        "scikit-learn==1.3.2",
        "mlflow==2.9.2",
        "numpy==1.24.3"
    ]
)
def train_model(
    X_train: Input[Dataset],
    X_test: Input[Dataset],
    y_train: Input[Dataset],
    y_test: Input[Dataset],
    mlflow_tracking_uri: str,
    experiment_name: str,
    team_name: str,
    n_estimators: int = 100,
    max_depth: int = 10
) -> str:
    """
    모델 학습 및 MLflow 메트릭 기록
    
    Note:
        S3 권한 문제를 피하기 위해 artifact 저장은 비활성화되어 있습니다.
        메트릭과 파라미터만 기록합니다.
    
    Returns:
        MLflow Run ID
    """
    import pandas as pd
    import numpy as np
    import mlflow
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import os
    
    print("=" * 60)
    print(f"  Step 4: Train Model - {team_name}")
    print("=" * 60)
    
    # 데이터 로드
    X_train_df = pd.read_csv(X_train.path)
    X_test_df = pd.read_csv(X_test.path)
    y_train_df = pd.read_csv(y_train.path)
    y_test_df = pd.read_csv(y_test.path)
    
    print(f"  Training data: {X_train_df.shape}")
    print(f"  Test data: {X_test_df.shape}")
    
    # MLflow 설정
    os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    print(f"  MLflow URI: {mlflow_tracking_uri}")
    print(f"  Experiment: {experiment_name}")
    
    with mlflow.start_run(run_name=f"{team_name}-run") as run:
        run_id = run.info.run_id
        print(f"  Run ID: {run_id}")
        
        # 파라미터 로깅
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": 42,
            "team": team_name
        })
        mlflow.set_tag("team", team_name)
        
        # 모델 학습
        print(f"  Training RandomForest...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_df, y_train_df.values.ravel())
        
        # 예측 및 평가
        y_pred = model.predict(X_test_df)
        
        r2 = r2_score(y_test_df, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_df, y_pred))
        mae = mean_absolute_error(y_test_df, y_pred)
        
        # 메트릭 로깅
        mlflow.log_metrics({"r2": r2, "rmse": rmse, "mae": mae})
        
        print(f"  Performance:")
        print(f"    - R2 Score: {r2:.4f}")
        print(f"    - RMSE: {rmse:.4f}")
        print(f"    - MAE: {mae:.4f}")
        
        # 피처 중요도 (메트릭으로만 기록)
        feature_importance = dict(zip(
            X_train_df.columns,
            model.feature_importances_
        ))
        
        sorted_importance = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        print(f"  Top 5 Feature Importance:")
        for feat, imp in sorted_importance:
            safe_name = feat.replace(" ", "_")[:15]
            mlflow.log_metric(f"fi_{safe_name}", imp)
            print(f"    - {feat}: {imp:.4f}")
        
        # ⚠️ S3 artifact 저장 비활성화 (권한 문제 방지)
        # mlflow.log_dict(feature_importance, "feature_importance.json")
        # mlflow.sklearn.log_model(model, "model")
        
        print(f"  ✅ Training completed")
    
    return run_id


# ============================================================
# Component 5: 모델 평가
# ============================================================
@component(
    base_image="python:3.9-slim",
    packages_to_install=["mlflow==2.9.2"]
)
def evaluate_model(
    run_id: str,
    mlflow_tracking_uri: str,
    r2_threshold: float = 0.75
) -> str:
    """모델 평가 및 배포 결정"""
    import mlflow
    import os
    
    print("=" * 60)
    print("  Step 5: Evaluate Model")
    print("=" * 60)
    
    os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    
    r2 = float(run.data.metrics.get("r2", 0))
    rmse = float(run.data.metrics.get("rmse", 0))
    mae = float(run.data.metrics.get("mae", 0))
    
    print(f"  Run ID: {run_id}")
    print(f"  Metrics: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    print(f"  Threshold: R2 >= {r2_threshold}")
    
    if r2 >= r2_threshold:
        decision = "deploy"
        print(f"  ✅ Decision: DEPLOY")
    else:
        decision = "skip"
        print(f"  ⚠️ Decision: SKIP")
    
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("deployment_decision", decision)
    
    return decision


# ============================================================
# Component 6: 모델 배포 (KServe)
# ============================================================
@component(
    base_image="python:3.9-slim",
    packages_to_install=["kubernetes==28.1.0"]
)
def deploy_model(
    run_id: str,
    model_name: str,
    namespace: str
):
    """
    KServe InferenceService로 모델 배포
    
    ⚠️ 중요: namespace는 현재 Kubeflow 프로필의 네임스페이스와 
    동일해야 RBAC 권한 오류가 발생하지 않습니다.
    """
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    import time
    
    print("=" * 60)
    print("  Step 6: Deploy Model (KServe)")
    print("=" * 60)
    
    print(f"  Model Name: {model_name}")
    print(f"  Namespace: {namespace}")
    print(f"  Run ID: {run_id}")
    
    try:
        config.load_incluster_config()
        print(f"  Using in-cluster config")
    except:
        config.load_kube_config()
    
    api = client.CustomObjectsApi()
    
    isvc = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": namespace,
            "annotations": {
                "sidecar.istio.io/inject": "false"
            }
        },
        "spec": {
            "predictor": {
                "sklearn": {
                    "storageUri": f"mlflow-artifacts:/{run_id}/model",
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "256Mi"},
                        "limits": {"cpu": "500m", "memory": "512Mi"}
                    }
                }
            }
        }
    }
    
    # 기존 삭제
    print(f"  Creating InferenceService...")
    try:
        api.delete_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=model_name
        )
        time.sleep(5)
    except ApiException as e:
        if e.status != 404:
            raise
    
    # 생성
    api.create_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=namespace,
        plural="inferenceservices",
        body=isvc
    )
    
    print(f"  ✅ InferenceService created: {model_name}")
    
    # 상태 확인
    print(f"  Waiting for deployment (max 60s)...")
    for i in range(6):
        time.sleep(10)
        try:
            status = api.get_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=model_name
            )
            conditions = status.get("status", {}).get("conditions", [])
            ready = next((c for c in conditions if c.get("type") == "Ready"), None)
            if ready and ready.get("status") == "True":
                print(f"  ✅ InferenceService READY!")
                break
            print(f"  ⏳ Status: {ready.get('status') if ready else 'Unknown'} ({(i+1)*10}s)")
        except:
            pass
    
    print(f"  Endpoint:")
    print(f"    http://{model_name}.{namespace}.svc.cluster.local/v1/models/{model_name}:predict")
    print(f"  ✅ Deployment completed!")


# ============================================================
# Component 7: 알림
# ============================================================
@component(base_image="python:3.9-slim")
def send_alert(run_id: str, team_name: str):
    """성능 미달 알림"""
    print("=" * 60)
    print(f"  Alert - {team_name}")
    print("=" * 60)
    print(f"  ⚠️ Model did not meet performance threshold")
    print(f"  Run ID: {run_id}")
    print(f"")
    print(f"  Recommendations:")
    print(f"    1. Add more features")
    print(f"    2. Tune hyperparameters")
    print(f"    3. Try different algorithms")


# ============================================================
# 파이프라인 정의
# ============================================================
@dsl.pipeline(
    name="Team Project Pipeline",
    description="Team Project: E2E ML Pipeline for California Housing"
)
def project_pipeline(
    data_source: str = "sklearn",
    team_name: str = "team-XX",
    experiment_name: str = "team-XX-experiment",
    model_name: str = "team-XX-model",
    namespace: str = "kubeflow-user-example-com",
    mlflow_tracking_uri: str = "http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000",
    n_estimators: int = 100,
    max_depth: int = 10,
    r2_threshold: float = 0.75
):
    """
    조별 프로젝트 파이프라인
    
    ⚠️ namespace 파라미터는 현재 Kubeflow 프로필의 네임스페이스와
    동일해야 합니다. (RBAC 권한 문제 방지)
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
        team_name=team_name,
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
            namespace=namespace
        )
    
    with dsl.If(evaluate_task.output == "skip"):
        send_alert(run_id=train_task.output, team_name=team_name)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"  Team Project Pipeline Compiler")
    print(f"  Team: {TEAM_NAME}")
    print("=" * 60)
    
    pipeline_file = f"{TEAM_NAME}_pipeline.yaml"
    
    compiler.Compiler().compile(
        pipeline_func=project_pipeline,
        package_path=pipeline_file
    )
    
    print(f"\n✅ Pipeline compiled: {pipeline_file}")
    print(f"\n📋 Parameters:")
    print(f"  - team_name: {TEAM_NAME}")
    print(f"  - experiment_name: {TEAM_NAME}-experiment")
    print(f"  - model_name: {TEAM_NAME}-model")
    print(f"  - namespace: {USER_NAMESPACE}")
    print(f"\n⚠️  Important:")
    print(f"  1. namespace는 현재 Kubeflow 프로필과 동일해야 합니다")
    print(f"  2. Feature Engineering에서 최소 1개 이상 피처를 추가하세요")
    print(f"\n🚀 Next: Upload {pipeline_file} to Kubeflow UI")
    print("=" * 60)
