#!/usr/bin/env python3
"""
Lab 3-1: Data Drift Detection
scipy + MLflow를 이용한 Drift 감지 (evidently 버전 독립적)
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from scipy.stats import ks_2samp
import mlflow

print("=" * 60)
print("  Lab 3-1: Data Drift Detection")
print("=" * 60)
print()

# 환경 변수
MLFLOW_TRACKING_URI = os.getenv(
    'MLFLOW_TRACKING_URI',
    'http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000'
)
EXPERIMENT_NAME = "drift-monitoring"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Step 1: 데이터 로드
print("[Step 1] Load Data")
data = fetch_california_housing(as_frame=True)
df = data.frame

# Reference data (과거 데이터)
reference_data = df.sample(n=5000, random_state=42)
print(f"  ✅ Reference data: {len(reference_data)} samples")

# Current data (현재 데이터 - Drift 시뮬레이션)
current_data = df.sample(n=3000, random_state=123)

# Drift 시뮬레이션: MedInc feature에 변화 주기
current_data = current_data.copy()
current_data['MedInc'] = current_data['MedInc'] * 1.5 + np.random.normal(0, 0.3, len(current_data))
print(f"  ✅ Current data: {len(current_data)} samples (with simulated drift)")
print()

# Step 2: Drift Detection (KS Test)
print("[Step 2] Detect Data Drift (Kolmogorov-Smirnov Test)")
drift_results = []

for col in reference_data.columns:
    # KS Test: 두 분포가 같은지 검정
    statistic, p_value = ks_2samp(reference_data[col], current_data[col])
    
    # p < 0.05이면 통계적으로 유의미한 차이 (Drift)
    drift_detected = p_value < 0.05
    
    drift_results.append({
        'feature': col,
        'ks_statistic': statistic,
        'p_value': p_value,
        'drift': drift_detected
    })
    
    if drift_detected:
        print(f"  🔴 {col:15s} - Drift Detected (KS={statistic:.3f}, p={p_value:.4f})")
    else:
        print(f"  🟢 {col:15s} - No Drift       (KS={statistic:.3f}, p={p_value:.4f})")

# 전체 Drift 요약
n_drifted = sum([r['drift'] for r in drift_results])
drift_score = n_drifted / len(drift_results)
dataset_drift = drift_score > 0.3

print()
print(f"  📊 Dataset Drift: {dataset_drift}")
print(f"  📊 Drifted Features: {n_drifted} / {len(drift_results)}")
print(f"  📊 Drift Score: {drift_score:.2f}")
print()

# Step 3: HTML 리포트 생성 (간단한 버전)
print("[Step 3] Generate HTML Report")

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Drift Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ background-color: #e8f5e9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .drift {{ color: #d32f2f; font-weight: bold; }}
        .no-drift {{ color: #388e3c; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Data Drift Detection Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p class="metric">Dataset Drift: <span class="{'drift' if dataset_drift else 'no-drift'}">{dataset_drift}</span></p>
            <p class="metric">Drift Score: <span class="{'drift' if drift_score > 0.3 else 'no-drift'}">{drift_score:.2%}</span></p>
            <p class="metric">Drifted Features: {n_drifted} / {len(drift_results)}</p>
        </div>
        
        <h2>Feature-level Analysis</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>KS Statistic</th>
                <th>P-value</th>
                <th>Drift Detected</th>
            </tr>
"""

for result in sorted(drift_results, key=lambda x: x['ks_statistic'], reverse=True):
    drift_class = 'drift' if result['drift'] else 'no-drift'
    drift_text = '🔴 Yes' if result['drift'] else '🟢 No'
    html_content += f"""
            <tr>
                <td><strong>{result['feature']}</strong></td>
                <td>{result['ks_statistic']:.4f}</td>
                <td>{result['p_value']:.4f}</td>
                <td class="{drift_class}">{drift_text}</td>
            </tr>
"""

html_content += """
        </table>
        
        <h2>Interpretation</h2>
        <ul>
            <li><strong>KS Statistic</strong>: 두 분포의 차이 정도 (0~1, 클수록 차이 큼)</li>
            <li><strong>P-value</strong>: 통계적 유의성 (< 0.05이면 Drift)</li>
            <li><strong>Drift Score</strong>: Drift가 감지된 Feature 비율</li>
        </ul>
        
        <h2>Recommendation</h2>
"""

if drift_score > 0.3:
    html_content += """
        <div style="background-color: #ffebee; padding: 20px; border-radius: 5px; border-left: 4px solid #d32f2f;">
            <p><strong>⚠️ Action Required:</strong> Drift Score가 높습니다 (> 0.3)</p>
            <ul>
                <li>모델 재학습을 고려하세요</li>
                <li>Feature Engineering 재검토</li>
                <li>데이터 수집 프로세스 확인</li>
            </ul>
        </div>
"""
else:
    html_content += """
        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 5px; border-left: 4px solid #388e3c;">
            <p><strong>✅ OK:</strong> Drift가 감지되지 않았거나 경미합니다 (< 0.3)</p>
            <ul>
                <li>현재 모델을 계속 사용 가능</li>
                <li>정기적인 모니터링 계속</li>
            </ul>
        </div>
"""

html_content += """
    </div>
</body>
</html>
"""

report_path = "drift_report.html"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"  ✅ drift_report.html 생성 완료")
print()

# Step 4: MLflow에 기록
print("[Step 4] Log to MLflow")
with mlflow.start_run(run_name="drift_detection"):
    # 메트릭 기록
    mlflow.log_metric("drift_score", drift_score)
    mlflow.log_metric("n_drifted_features", n_drifted)
    mlflow.log_metric("dataset_drift", 1 if dataset_drift else 0)
    
    # Feature별 메트릭
    for result in drift_results:
        mlflow.log_metric(f"ks_{result['feature']}", result['ks_statistic'])
        mlflow.log_metric(f"pval_{result['feature']}", result['p_value'])
    
    # 아티팩트 기록
    mlflow.log_artifact(report_path)
    
    # 태그
    mlflow.set_tag("drift_detected", str(dataset_drift))
    mlflow.set_tag("method", "ks_test")
    
    print("  ✅ Metrics logged to MLflow")
    print(f"     - drift_score: {drift_score:.2f}")
    print(f"     - n_drifted_features: {n_drifted}")

print()
print("=" * 60)
print("✅ Drift Detection 완료!")
print("=" * 60)
print()
print(f"📊 결과:")
print(f"  - drift_report.html (다운로드하여 브라우저에서 확인)")
print()
print(f"📋 다음 단계:")
print(f"  1. drift_report.html을 브라우저에서 열어보세요")
print(f"  2. MLflow UI에서 메트릭을 확인하세요:")
print(f"     kubectl port-forward svc/mlflow-server-service -n mlflow-system 5000:5000")
print(f"     http://localhost:5000")
print(f"  3. Drift Score > 0.3이면 재학습을 고려하세요")
print()
