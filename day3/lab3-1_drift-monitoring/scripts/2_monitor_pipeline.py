#!/usr/bin/env python3
"""
Lab 3-1 Part 2: Drift Monitoring Pipeline
Kubeflow Pipeline for automated drift monitoring

IMPORTANT:
- Use ASCII characters only in pipeline name, description, docstrings, and print statements
- KFP SDK 2.7.0+ required
"""

import os
from kfp import dsl
from kfp import compiler

print("=" * 60)
print("  Lab 3-1 Part 2: Monitoring Pipeline")
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
# Component 1: Collect Production Data
# ============================================================
@dsl.component(base_image="python:3.9-slim")
def collect_production_data(sample_size: int = 1000) -> int:
    """Collect production data simulation"""
    print(f"Data collection simulated: {sample_size} samples")
    return sample_size


# ============================================================
# Component 2: Detect Drift
# ============================================================
@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=["scikit-learn==1.3.2", "pandas==2.0.3", "numpy", "scipy"]
)
def detect_drift(sample_size: int, drift_threshold: float = 0.3) -> str:
    """Detect drift using KS test"""
    from sklearn.datasets import fetch_california_housing
    import pandas as pd
    import numpy as np
    from scipy.stats import ks_2samp
    import json
    
    print(f"Loading data for drift detection...")
    
    # Load California Housing dataset
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    # Reference data (baseline)
    reference_data = df.sample(n=2000, random_state=42)
    print(f"Reference data: {len(reference_data)} samples")
    
    # Current data (with simulated drift on MedInc feature)
    current_data = df.sample(n=sample_size, random_state=123)
    current_data = current_data.copy()
    current_data['MedInc'] = current_data['MedInc'] * 1.5 + np.random.normal(0, 0.3, len(current_data))
    print(f"Current data: {len(current_data)} samples")
    
    # Drift detection using KS Test
    n_drifted = 0
    for col in reference_data.columns:
        _, p_value = ks_2samp(reference_data[col], current_data[col])
        if p_value < 0.05:  # Significant difference
            n_drifted += 1
    
    drift_score = n_drifted / len(reference_data.columns)
    drift_detected = drift_score > drift_threshold
    
    result = {
        'drift_detected': bool(drift_detected),
        'drift_score': float(drift_score),
        'n_drifted': int(n_drifted)
    }
    
    print(f"Drift Score: {drift_score:.2f}")
    print(f"Drifted Features: {n_drifted}/{len(reference_data.columns)}")
    print(f"Drift Detected: {drift_detected}")
    
    return json.dumps(result)


# ============================================================
# Component 3: Log Metrics to MLflow
# ============================================================
@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=["mlflow==2.9.2"]
)
def log_metrics(drift_result: str, mlflow_uri: str) -> str:
    """Log metrics to MLflow with error handling"""
    import json
    import os
    
    result = json.loads(drift_result)
    
    print(f"Drift Score: {result['drift_score']:.2f}")
    print(f"Drift Detected: {result['drift_detected']}")
    print(f"Drifted Features: {result['n_drifted']}")
    
    try:
        import mlflow
        
        print(f"Connecting to MLflow: {mlflow_uri}")
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        
        try:
            mlflow.set_experiment("drift-monitoring")
        except Exception:
            print("Creating new experiment...")
            mlflow.create_experiment("drift-monitoring")
            mlflow.set_experiment("drift-monitoring")
        
        with mlflow.start_run(run_name="drift-check"):
            mlflow.log_metric("drift_score", result['drift_score'])
            mlflow.log_metric("drift_detected", 1 if result['drift_detected'] else 0)
            mlflow.log_metric("n_drifted", result['n_drifted'])
            mlflow.set_tag("pipeline", "monitoring")
        
        print("Metrics logged to MLflow successfully")
        return "success"
        
    except Exception as e:
        print(f"MLflow logging failed: {e}")
        print("Continuing without MLflow...")
        return "success-no-mlflow"


# ============================================================
# Component 4: Send Alert
# ============================================================
@dsl.component(base_image="python:3.9-slim")
def send_alert(drift_result: str) -> str:
    """Send alert simulation"""
    import json
    
    result = json.loads(drift_result)
    
    if result['drift_detected']:
        print("ALERT: Data Drift Detected!")
        print(f"Drift Score: {result['drift_score']:.2f}")
        print("Action: Consider model retraining")
        return "alert-sent"
    else:
        print("OK: No significant drift detected")
        return "no-alert"


# ============================================================
# Pipeline Definition
# ============================================================
@dsl.pipeline(
    name="drift-monitoring",
    description="drift monitoring pipeline"
)
def drift_monitoring_pipeline(
    sample_size: int = 1000,
    drift_threshold: float = 0.3,
    mlflow_uri: str = MLFLOW_TRACKING_URI
):
    """Drift monitoring pipeline with 4 components"""
    
    # Step 1: Collect data (simulated)
    collect_task = collect_production_data(sample_size=sample_size)
    
    # Step 2: Detect drift
    detect_task = detect_drift(
        sample_size=collect_task.output,
        drift_threshold=drift_threshold
    )
    
    # Step 3: Log metrics to MLflow
    log_task = log_metrics(
        drift_result=detect_task.output,
        mlflow_uri=mlflow_uri
    )
    
    # Step 4: Send alert
    alert_task = send_alert(drift_result=detect_task.output)


# ============================================================
# Compile Pipeline
# ============================================================
if __name__ == "__main__":
    pipeline_filename = "drift_monitoring_pipeline.yaml"
    
    print("Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=drift_monitoring_pipeline,
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
    print("       - sample_size: 1000")
    print("       - drift_threshold: 0.3")
    print("    7. Click 'Start'")
    print("=" * 60)
