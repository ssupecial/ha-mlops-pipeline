#!/usr/bin/env python3
"""
Lab 3-1 Part 3: Auto-Retraining Pipeline
Kubeflow Pipeline for automatic model retraining when drift is detected

IMPORTANT:
- Use ASCII characters only in pipeline name, description, docstrings, and print statements
- KFP SDK 2.7.0+ required
"""

import os
from kfp import dsl
from kfp import compiler

print("=" * 60)
print("  Lab 3-1 Part 3: Auto-Retraining Pipeline")
print("=" * 60)
print()

# Configuration
USER_NUM = os.getenv("USER_NUM", "01")
NAMESPACE = f"kubeflow-user{USER_NUM}"
MLFLOW_TRACKING_URI = f"http://mlflow-server.kubeflow-user{USER_NUM}.svc.cluster.local:5000"

print(f"  USER_NUM: {USER_NUM}")
print(f"  NAMESPACE: {NAMESPACE}")
print(f"  MLFLOW_URI: {MLFLOW_TRACKING_URI}")
print()

# ============================================================
# Component 1: Check Drift and Decide
# ============================================================
@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=["mlflow==2.9.2", "pandas==2.0.3"]
)
def check_drift_and_decide(drift_threshold: float, mlflow_uri: str) -> str:
    """Check recent drift score and decide whether to retrain"""
    import json
    import os
    
    print(f"Drift Threshold: {drift_threshold}")
    
    drift_score = 0.0
    should_retrain = True  # Default: retrain if cannot check
    
    try:
        import mlflow
        
        print(f"Connecting to MLflow: {mlflow_uri}")
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        
        try:
            mlflow.set_experiment("drift-monitoring")
            
            # Query recent Run
            runs = mlflow.search_runs(
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if len(runs) > 0 and 'metrics.drift_score' in runs.columns:
                drift_score = float(runs.iloc[0]['metrics.drift_score'])
                should_retrain = drift_score > drift_threshold
                print(f"Found drift score from MLflow: {drift_score:.2f}")
            else:
                print("No previous runs found, defaulting to retrain")
                should_retrain = True
                
        except Exception as e:
            print(f"MLflow experiment error: {e}")
            should_retrain = True
            
    except Exception as e:
        print(f"MLflow connection failed: {e}")
        should_retrain = True
    
    result = {
        'should_retrain': bool(should_retrain),
        'drift_score': float(drift_score)
    }
    
    print(f"Decision:")
    print(f"  Drift Score: {drift_score:.2f}")
    print(f"  Should Retrain: {should_retrain}")
    
    return json.dumps(result)


# ============================================================
# Component 2: Retrain Model
# ============================================================
@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas==2.0.3", "numpy", "scikit-learn==1.3.2", "mlflow==2.9.2"]
)
def retrain_model(train_size: int, mlflow_uri: str) -> str:
    """Retrain model with California Housing dataset"""
    import pandas as pd
    import numpy as np
    from sklearn.datasets import fetch_california_housing
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import json
    import os
    import uuid
    
    print(f"Loading training data...")
    
    # Load data
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    # Sample training data
    train_data = df.sample(n=train_size, random_state=42)
    print(f"Training data: {len(train_data)} samples")
    
    X = train_data.drop('MedHouseVal', axis=1)
    y = train_data['MedHouseVal']
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training RandomForest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    # Model version
    model_version = str(uuid.uuid4())[:8]
    
    print(f"Training Results:")
    print(f"  Model version: {model_version}")
    print(f"  MAE: {mae:.4f}")
    
    # Log to MLflow
    try:
        import mlflow
        
        print(f"Connecting to MLflow: {mlflow_uri}")
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        
        try:
            mlflow.set_experiment("auto-retraining")
        except:
            mlflow.create_experiment("auto-retraining")
            mlflow.set_experiment("auto-retraining")
        
        with mlflow.start_run(run_name="retrained-model"):
            mlflow.log_metric("mae", mae)
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("train_size", train_size)
            model_version = mlflow.active_run().info.run_id[:8]
        
        print("Metrics logged to MLflow successfully")
        
    except Exception as e:
        print(f"MLflow logging failed: {e}")
    
    result = {
        'model_version': str(model_version),
        'mae': float(mae),
        'status': 'trained'
    }
    
    print(f"Model trained successfully!")
    
    return json.dumps(result)


# ============================================================
# Component 3: Deploy Model
# ============================================================
@dsl.component(base_image="python:3.9-slim")
def deploy_model(model_result: str) -> str:
    """Deploy model simulation"""
    import json
    
    result = json.loads(model_result)
    model_version = result['model_version']
    mae = result['mae']
    
    print(f"Deploying model version: {model_version}")
    print(f"Model MAE: {mae:.4f}")
    print("Model deployed successfully!")
    
    return "deployed"


# ============================================================
# Pipeline Definition
# ============================================================
@dsl.pipeline(
    name="auto-retrain",
    description="auto retraining pipeline"
)
def auto_retrain_pipeline(
    drift_threshold: float = 0.3,
    train_size: int = 5000,
    mlflow_uri: str = MLFLOW_TRACKING_URI
):
    """Auto retraining pipeline with 3 components"""
    
    # Step 1: Check drift and decide
    check_task = check_drift_and_decide(
        drift_threshold=drift_threshold,
        mlflow_uri=mlflow_uri
    )
    
    # Step 2: Retrain model (after check_task)
    retrain_task = retrain_model(
        train_size=train_size,
        mlflow_uri=mlflow_uri
    )
    retrain_task.after(check_task)  # Sequential execution
    
    # Step 3: Deploy model
    deploy_task = deploy_model(model_result=retrain_task.output)


# ============================================================
# Compile Pipeline
# ============================================================
if __name__ == "__main__":
    pipeline_filename = "auto_retrain_pipeline.yaml"
    
    print("Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=auto_retrain_pipeline,
        package_path=pipeline_filename
    )
    
    print()
    print("=" * 60)
    print("  Pipeline compiled successfully!")
    print("=" * 60)
    print()
    print(f"  Output file: {pipeline_filename}")
    print()
    print("  Next steps:")
    print("    1. Go to Kubeflow UI -> Pipelines")
    print("    2. Click '+ Upload pipeline'")
    print(f"    3. Select '{pipeline_filename}'")
    print("    4. Click 'Create'")
    print("    5. Click '+ Create run'")
    print("    6. Set parameters:")
    print("       - drift_threshold: 0.3")
    print("       - train_size: 5000")
    print("    7. Click 'Start'")
    print("=" * 60)
