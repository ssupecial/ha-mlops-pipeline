#!/usr/bin/env python3
"""
Lab 3-1 Part 1: Data Drift Detection
Detect data drift using KS Test (Kolmogorov-Smirnov Test)
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from scipy.stats import ks_2samp

print("=" * 60)
print("  Lab 3-1 Part 1: Data Drift Detection")
print("=" * 60)
print()

# Configuration
DRIFT_THRESHOLD = 0.3  # 30% of features drifted
SIGNIFICANCE_LEVEL = 0.05  # p-value threshold

# ============================================================
# Step 1: Load Data
# ============================================================
print("[Step 1] Loading California Housing data...")

data = fetch_california_housing(as_frame=True)
df = data.frame

# Reference data (baseline - historical data)
reference_data = df.sample(n=5000, random_state=42)
print(f"  Reference data: {len(reference_data)} samples")

# Current data (production data with simulated drift)
current_data = df.sample(n=3000, random_state=123)
current_data = current_data.copy()

# Simulate drift on MedInc feature (increase by 50% + noise)
current_data['MedInc'] = current_data['MedInc'] * 1.5 + np.random.normal(0, 0.3, len(current_data))
print(f"  Current data: {len(current_data)} samples (with simulated drift)")
print()

# ============================================================
# Step 2: Drift Detection using KS Test
# ============================================================
print("[Step 2] Performing Drift Detection (KS Test)...")
print()

drift_results = []

for col in reference_data.columns:
    # KS Test: Tests if two samples come from the same distribution
    statistic, p_value = ks_2samp(reference_data[col], current_data[col])
    
    # p < 0.05 means statistically significant difference (drift detected)
    drift_detected = p_value < SIGNIFICANCE_LEVEL
    
    drift_results.append({
        'feature': col,
        'ks_statistic': statistic,
        'p_value': p_value,
        'drift': drift_detected
    })
    
    status = "YES" if drift_detected else "NO "
    print(f"  Feature: {col:12s} - Drift: {status} (p-value: {p_value:.4f})")

print()

# ============================================================
# Step 3: Calculate Drift Score
# ============================================================
print("[Step 3] Drift Summary")
print()

drift_df = pd.DataFrame(drift_results)
n_drifted = drift_df['drift'].sum()
n_total = len(drift_df)
drift_score = n_drifted / n_total

print(f"  Drifted Features: {n_drifted}/{n_total}")
print(f"  Drift Score: {drift_score:.2f} ({drift_score*100:.0f}%)")
print(f"  Threshold: {DRIFT_THRESHOLD:.2f} ({DRIFT_THRESHOLD*100:.0f}%)")

if drift_score > DRIFT_THRESHOLD:
    print(f"  Status: DRIFT DETECTED! Consider retraining.")
else:
    print(f"  Status: No significant drift")

print()

# ============================================================
# Step 4: MLflow Logging (Optional)
# ============================================================
print("[Step 4] Logging to MLflow...")

try:
    import mlflow
    
    # Get MLflow URI from environment or use default
    MLFLOW_TRACKING_URI = os.getenv(
        'MLFLOW_TRACKING_URI',
        'http://mlflow-server-service.mlflow-system.svc.cluster.local:5000'
    )
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        mlflow.set_experiment("drift-monitoring")
    except:
        mlflow.create_experiment("drift-monitoring")
        mlflow.set_experiment("drift-monitoring")
    
    with mlflow.start_run(run_name="drift-detection"):
        mlflow.log_metric("drift_score", drift_score)
        mlflow.log_metric("n_drifted", n_drifted)
        mlflow.log_metric("n_total_features", n_total)
        mlflow.set_tag("drift_detected", str(drift_score > DRIFT_THRESHOLD))
    
    print(f"  MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"  Metrics logged successfully!")
    
except Exception as e:
    print(f"  MLflow logging skipped: {e}")
    print(f"  (This is OK for local testing)")

print()

# ============================================================
# Summary
# ============================================================
print("=" * 60)
print("  Part 1 Complete!")
print("=" * 60)
print()
print("  Key Learnings:")
print("  1. KS Test compares two distributions")
print("  2. p-value < 0.05 indicates drift")
print("  3. Drift Score = drifted features / total features")
print("  4. Threshold determines when to retrain")
print()
print("  Next: Part 2 - Monitoring Pipeline")
print("=" * 60)
