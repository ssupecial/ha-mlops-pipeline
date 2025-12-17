#!/usr/bin/env python3
"""
Lab 3-1: Drift Monitoring Pipeline
Kubeflow Pipeline for automated Drift monitoring
"""

import os
from kfp import dsl
from kfp import compiler
from kfp.components import create_component_from_func

print("=" * 60)
print("  Lab 3-1: Monitoring Pipeline")
print("=" * 60)

# Component 1: Collect Production Data
def collect_production_data(sample_size: int = 1000) -> int:
    """Collect production data (simulation) - return sample size only"""
    print(f"Data collection simulated: {sample_size} samples")
    return sample_size

collect_production_data_op = create_component_from_func(
    func=collect_production_data,
    base_image='python:3.9'
)


# Component 2: Detect Drift
def detect_drift(sample_size: int, drift_threshold: float = 0.3) -> str:
    """Detect drift - load data internally"""
    from sklearn.datasets import fetch_california_housing
    import pandas as pd
    import numpy as np
    from scipy.stats import ks_2samp
    import json
    
    print(f"Loading data for drift detection...")
    
    # Load data
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    # Reference data
    reference_data = df.sample(n=2000, random_state=42)
    print(f"Reference data: {len(reference_data)} samples")
    
    # Current data (with simulated drift)
    current_data = df.sample(n=sample_size, random_state=123)
    current_data['MedInc'] = current_data['MedInc'] * 1.5 + np.random.normal(0, 0.3, len(current_data))
    print(f"Current data: {len(current_data)} samples")
    
    # Drift detection (KS Test)
    n_drifted = 0
    for col in reference_data.columns:
        _, p_value = ks_2samp(reference_data[col], current_data[col])
        if p_value < 0.05:
            n_drifted += 1
    
    drift_score = n_drifted / len(reference_data.columns)
    drift_detected = drift_score > drift_threshold
    
    # Return result as JSON string
    result = {
        'drift_detected': bool(drift_detected),
        'drift_score': float(drift_score),
        'n_drifted': int(n_drifted)
    }
    
    print(f"Drift Score: {drift_score:.2f}")
    print(f"Drifted Features: {n_drifted}/{len(reference_data.columns)}")
    print(f"Drift Detected: {drift_detected}")
    
    return json.dumps(result)

detect_drift_op = create_component_from_func(
    func=detect_drift,
    base_image='python:3.9',
    packages_to_install=['scikit-learn==1.3.2', 'pandas==2.0.3', 'numpy', 'scipy']
)


# Component 3: Log Metrics
def log_metrics(drift_result: str) -> str:
    """Log metrics to MLflow"""
    import mlflow
    import json
    
    # Parse JSON
    result = json.loads(drift_result)
    
    mlflow_uri = "http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("drift-monitoring-pipeline")
    
    with mlflow.start_run(run_name="pipeline_drift_check"):
        mlflow.log_metric("drift_score", result['drift_score'])
        mlflow.log_metric("drift_detected", 1 if result['drift_detected'] else 0)
        mlflow.log_metric("n_drifted", result['n_drifted'])
        mlflow.set_tag("pipeline", "monitoring")
    
    print("Metrics logged to MLflow")
    return "Success"

log_metrics_op = create_component_from_func(
    func=log_metrics,
    base_image='python:3.9',
    packages_to_install=['mlflow==2.9.2']
)


# Component 4: Send Alert
def send_alert(drift_result: str) -> str:
    """Send alert (simulation)"""
    import json
    
    result = json.loads(drift_result)
    
    if result['drift_detected']:
        print("ALERT: Data Drift Detected!")
        print(f"Drift Score: {result['drift_score']:.2f}")
        print("Action: Consider model retraining")
        return "Alert sent"
    else:
        print("OK: No significant drift detected")
        return "No alert needed"

send_alert_op = create_component_from_func(
    func=send_alert,
    base_image='python:3.9'
)


# Pipeline Definition
@dsl.pipeline(
    name='Drift Monitoring Pipeline',
    description='Automated Data Drift monitoring pipeline'
)
def drift_monitoring_pipeline(
    sample_size: int = 1000,
    drift_threshold: float = 0.3
):
    """Drift monitoring pipeline"""
    
    # Step 1: Collect data (simulated)
    collect_task = collect_production_data_op(sample_size=sample_size)
    
    # Step 2: Detect drift (loads data internally)
    detect_task = detect_drift_op(
        sample_size=collect_task.output,
        drift_threshold=drift_threshold
    )
    
    # Step 3: Log metrics
    log_task = log_metrics_op(drift_result=detect_task.output)
    
    # Step 4: Send alert
    alert_task = send_alert_op(drift_result=detect_task.output)


# Compile Pipeline
if __name__ == '__main__':
    pipeline_filename = 'drift_monitoring_pipeline.yaml'
    
    compiler.Compiler().compile(
        pipeline_func=drift_monitoring_pipeline,
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
    print("     - sample_size: 1000")
    print("     - drift_threshold: 0.3")
    print("  4. Click Start to execute")
    print()
    print("Note: Each component loads data independently")
    print("      No large data transfer between components")
    print()
