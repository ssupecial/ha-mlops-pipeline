#!/usr/bin/env python3
"""
Lab 3-1: Auto-Retraining Pipeline
Auto-retraining pipeline when drift is detected
"""

import os
from kfp import dsl
from kfp import compiler
from kfp.components import create_component_from_func

print("=" * 60)
print("  Lab 3-1: Auto-Retraining Pipeline")
print("=" * 60)

# Component 1: Check Drift and Decide
def check_drift_and_decide(drift_threshold: float = 0.3) -> str:
    """Check recent Drift Score and decide whether to retrain"""
    import mlflow
    import json
    
    mlflow_uri = "http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("drift-monitoring-pipeline")
    
    try:
        # Query recent Run for Drift Score
        runs = mlflow.search_runs(
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if len(runs) > 0:
            drift_score = float(runs.iloc[0]['metrics.drift_score'])
            should_retrain = drift_score > drift_threshold
        else:
            drift_score = 0.0
            should_retrain = False
    except Exception as e:
        print(f"Warning: Could not fetch drift score: {e}")
        drift_score = 0.0
        should_retrain = False
    
    result = {
        'should_retrain': bool(should_retrain),
        'drift_score': float(drift_score)
    }
    
    print(f"Drift Score: {drift_score:.2f}")
    print(f"Should Retrain: {should_retrain}")
    
    return json.dumps(result)

check_drift_op = create_component_from_func(
    func=check_drift_and_decide,
    base_image='python:3.9',
    packages_to_install=['mlflow==2.9.2', 'pandas==2.0.3']
)


# Component 2: Retrain Model
def retrain_model(train_size: int = 5000) -> str:
    """Retrain model - load data internally (metrics only)"""
    import pandas as pd
    import numpy as np
    from sklearn.datasets import fetch_california_housing
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import mlflow
    import json
    
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
    
    # MLflow setup
    mlflow_uri = "http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("auto-retraining")
    
    # Train model
    with mlflow.start_run(run_name="retrained_model"):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        
        # Log metrics only (no model artifact)
        mlflow.log_metric("mae", mae)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("train_size", train_size)
        
        # Model version from run_id
        run_id = mlflow.active_run().info.run_id
        model_version = run_id[:8]
        
        print(f"Model trained successfully")
        print(f"Model version: {model_version}")
        print(f"MAE: {mae:.4f}")
    
    result = {
        'model_version': str(model_version),
        'mae': float(mae),
        'status': 'metrics_logged'
    }
    
    return json.dumps(result)

retrain_model_op = create_component_from_func(
    func=retrain_model,
    base_image='python:3.9',
    packages_to_install=['pandas==2.0.3', 'numpy', 'scikit-learn==1.3.2', 'mlflow==2.9.2']
)


# Component 3: Deploy Model (simulation)
def deploy_model(model_result: str) -> str:
    """Deploy model (simulation)"""
    import json
    
    result = json.loads(model_result)
    model_version = result['model_version']
    mae = result['mae']
    
    print(f"Deploying model version: {model_version}")
    print(f"Model MAE: {mae:.4f}")
    print("Model deployed successfully!")
    
    return "Deployed"

deploy_model_op = create_component_from_func(
    func=deploy_model,
    base_image='python:3.9'
)


# Pipeline Definition
@dsl.pipeline(
    name='Auto-Retraining Pipeline',
    description='Auto-retraining pipeline when drift is detected'
)
def auto_retrain_pipeline(
    drift_threshold: float = 0.3,
    train_size: int = 5000
):
    """Auto-retraining pipeline"""
    
    # Step 1: Check drift and decide to retrain
    check_task = check_drift_op(drift_threshold=drift_threshold)
    
    # Step 2: Retrain model (loads data internally)
    retrain_task = retrain_model_op(train_size=train_size)
    retrain_task.after(check_task)
    
    # Step 3: Deploy model
    deploy_task = deploy_model_op(model_result=retrain_task.output)


# Compile Pipeline
if __name__ == '__main__':
    pipeline_filename = 'auto_retrain_pipeline.yaml'
    
    compiler.Compiler().compile(
        pipeline_func=auto_retrain_pipeline,
        package_path=pipeline_filename
    )
    
    print()
    print("=" * 60)
    print("Pipeline compiled successfully!")
    print("=" * 60)
    print()
    print(f"Output file: {pipeline_filename}")
    print()
    print("Next steps:")
    print("  1. Upload pipeline to Kubeflow UI")
    print("  2. Click Create Run")
    print("  3. Set parameters:")
    print("     - drift_threshold: 0.3")
    print("     - train_size: 5000")
    print("  4. Click Start to execute")
    print()
    print("Note: Each component loads data independently")
    print("      No large data transfer between components")
    print()
