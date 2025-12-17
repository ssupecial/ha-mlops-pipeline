#!/usr/bin/env python3
"""
Lab 3-3: Benchmark & MLflow
원본, ONNX, 양자화 모델의 성능 벤치마크 및 MLflow 기록

실행: 
  export MLFLOW_TRACKING_URI=http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000
  python scripts/3_benchmark.py

사전 요구: 
  - python scripts/1_onnx_conversion.py 실행 완료
  - python scripts/2_quantization.py 실행 완료
  - IRSA 설정 완료 (S3 접근용)
"""

import os
import time
import boto3
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import onnxruntime as ort
import mlflow
from mlflow import MlflowClient

def measure_inference_time(predict_fn, X, n_iterations=1000):
    """추론 시간 측정 (밀리초 단위)"""
    # 워밍업
    for _ in range(10):
        predict_fn(X)
    
    # 실제 측정
    start = time.perf_counter()
    for _ in range(n_iterations):
        predict_fn(X)
    end = time.perf_counter()
    
    return ((end - start) / n_iterations) * 1000  # ms로 변환

def main():

    # ⚠️ 강사가 제공한 본인의 AWS 자격 증명으로 변경!
    AWS_ACCESS_KEY_ID = "YOUR_AWS_ACCESS_KEY_ID"          # 수정 필요!
    AWS_SECRET_ACCESS_KEY = "YOUR_AWS_SECRET_ACCESS_KEY"  # 수정 필요!
    AWS_REGION = "ap-northeast-2"

    # ⚠️ 본인의 사용자 번호로 변경!
    USER_NUM = "YOUR_USER_NUM"
    NAMESPACE = f"kubeflow-user{USER_NUM}"

    # 환경 변수 설정
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_DEFAULT_REGION"] = AWS_REGION

    print(f"✅ 사용자 설정 완료:")

    print("=" * 60)
    print("Lab 3-3: Benchmark & MLflow 기록")
    print("=" * 60)
    
    # =========================================================================
    # Step 0: 사전 확인
    # =========================================================================
    print("\n🔍 Step 0: 사전 확인")
    print("-" * 40)
    
    # 파일 존재 확인
    files = {
        'original': 'outputs/model_original.pkl',
        'onnx': 'outputs/model_optimized.onnx',
        'quantized': 'outputs/model_quantized.onnx'
    }
    
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"   ❌ 오류: {path} 파일이 없습니다.")
            print("   먼저 1_onnx_conversion.py와 2_quantization.py를 실행하세요.")
            return
        print(f"   ✅ {name}: {path}")
    
    # AWS 자격 증명 확인 (IRSA)
    print("\n   AWS 자격 증명 확인...")
    try:
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"   ✅ AWS Identity: {identity['Arn']}")
    except Exception as e:
        print(f"   ⚠️ AWS 자격 증명 경고: {e}")
        print("   MLflow 아티팩트 저장이 실패할 수 있습니다.")
        print("   IRSA 설정을 확인하세요.")
    
    # =========================================================================
    # Step 1: 모델 및 데이터 로드
    # =========================================================================
    print("\n📂 Step 1: 모델 및 데이터 로드")
    print("-" * 40)
    
    # 테스트 데이터 준비
    iris = load_iris()
    X, y = iris.data, iris.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_float32 = X_test.astype(np.float32)
    
    print(f"   테스트 샘플 수: {len(X_test)}")
    
    # 원본 sklearn 모델 로드
    with open(files['original'], 'rb') as f:
        sklearn_model = pickle.load(f)
    
    # ONNX 모델 로드
    onnx_session = ort.InferenceSession(
        files['onnx'], 
        providers=['CPUExecutionProvider']
    )
    input_name = onnx_session.get_inputs()[0].name
    
    # 양자화 모델 로드
    quant_session = ort.InferenceSession(
        files['quantized'], 
        providers=['CPUExecutionProvider']
    )
    
    print("   ✅ 모든 모델 로드 완료")
    
    # =========================================================================
    # Step 2: 모델 크기 비교
    # =========================================================================
    print("\n📏 Step 2: 모델 크기 비교")
    print("-" * 40)
    
    original_size = os.path.getsize(files['original']) / 1024
    onnx_size = os.path.getsize(files['onnx']) / 1024
    quant_size = os.path.getsize(files['quantized']) / 1024
    
    print(f"   원본 sklearn: {original_size:.2f} KB")
    print(f"   ONNX: {onnx_size:.2f} KB ({((onnx_size/original_size)-1)*100:+.1f}%)")
    print(f"   양자화: {quant_size:.2f} KB ({((quant_size/original_size)-1)*100:+.1f}%)")
    
    # =========================================================================
    # Step 3: 추론 속도 벤치마크
    # =========================================================================
    print("\n⚡ Step 3: 추론 속도 벤치마크")
    print("-" * 40)
    
    n_iterations = 1000
    print(f"   반복 횟수: {n_iterations}회")
    
    # 원본 sklearn 추론
    print("\n   원본 sklearn 측정 중...")
    original_time = measure_inference_time(
        lambda x: sklearn_model.predict(x), 
        X_test, 
        n_iterations
    )
    
    # ONNX 추론
    print("   ONNX Runtime 측정 중...")
    onnx_time = measure_inference_time(
        lambda x: onnx_session.run(None, {input_name: x.astype(np.float32)}),
        X_test,
        n_iterations
    )
    
    # 양자화 모델 추론
    print("   양자화 모델 측정 중...")
    quant_time = measure_inference_time(
        lambda x: quant_session.run(None, {input_name: x.astype(np.float32)}),
        X_test,
        n_iterations
    )
    
    # 속도 향상 계산
    onnx_speedup = original_time / onnx_time
    quant_speedup = original_time / quant_time
    
    print(f"\n   📊 벤치마크 결과:")
    print(f"   {'모델':<15} {'추론 시간':<15} {'속도 향상':<10}")
    print(f"   {'-' * 40}")
    print(f"   {'원본 sklearn':<15} {original_time:.4f} ms{'':<6} 1.0x")
    print(f"   {'ONNX':<15} {onnx_time:.4f} ms{'':<6} {onnx_speedup:.1f}x")
    print(f"   {'양자화':<15} {quant_time:.4f} ms{'':<6} {quant_speedup:.1f}x")
    
    # =========================================================================
    # Step 4: 정확도 검증
    # =========================================================================
    print("\n🎯 Step 4: 정확도 검증")
    print("-" * 40)
    
    # 예측
    original_pred = sklearn_model.predict(X_test)
    onnx_pred = onnx_session.run(None, {input_name: X_test_float32})[0]
    quant_pred = quant_session.run(None, {input_name: X_test_float32})[0]
    
    # 정확도 계산
    test_accuracy = accuracy_score(y_test, original_pred)
    onnx_accuracy = accuracy_score(y_test, onnx_pred)
    quant_accuracy = accuracy_score(y_test, quant_pred)
    
    print(f"   원본 sklearn 정확도: {test_accuracy:.4f}")
    print(f"   ONNX 정확도: {onnx_accuracy:.4f}")
    print(f"   양자화 정확도: {quant_accuracy:.4f}")
    
    # =========================================================================
    # Step 5: MLflow 기록
    # =========================================================================
    print("\n📝 Step 5: MLflow 기록")
    print("-" * 40)
    
    # MLflow 서버 URI 설정
    mlflow_uri = os.environ.get(
        'MLFLOW_TRACKING_URI',
        'http://mlflow-server.kubeflow-user${USER_NUM}.svc.cluster.local:5000'
    )
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"   MLflow URI: {mlflow_uri}")


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
    EXPERIMENT_NAME = f"lab3-3-model-optimization-user{USER_NUM}"

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
    
    # MLflow Run 시작
    with mlflow.start_run(run_name="benchmark-results") as run:
        # 크기 메트릭
        mlflow.log_metric("original_size_kb", original_size)
        mlflow.log_metric("onnx_size_kb", onnx_size)
        mlflow.log_metric("quantized_size_kb", quant_size)
        
        # 속도 메트릭
        mlflow.log_metric("original_inference_ms", original_time)
        mlflow.log_metric("onnx_inference_ms", onnx_time)
        mlflow.log_metric("quantized_inference_ms", quant_time)
        mlflow.log_metric("onnx_speedup", onnx_speedup)
        mlflow.log_metric("quantized_speedup", quant_speedup)
        
        # 정확도 메트릭
        mlflow.log_metric("original_accuracy", test_accuracy)
        mlflow.log_metric("onnx_accuracy", onnx_accuracy)
        mlflow.log_metric("quantized_accuracy", quant_accuracy)
        
        # 파라미터
        mlflow.log_param("n_iterations", n_iterations)
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("quantization_type", "dynamic_uint8")
        
        # 아티팩트 (모델 파일)
        print("\n   아티팩트 업로드 중...")
        try:
            mlflow.log_artifact(files['onnx'])
            mlflow.log_artifact(files['quantized'])
            print("   ✅ 아티팩트 업로드 완료")
        except Exception as e:
            print(f"   ⚠️ 아티팩트 업로드 실패: {e}")
            print("   IRSA 설정을 확인하세요.")
        
        print(f"\n   ✅ MLflow 기록 완료!")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   실험: {EXPERIMENT_NAME}")
    
    # =========================================================================
    # 최종 결과 요약
    # =========================================================================
    print("\n" + "=" * 60)
    print("📋 최종 결과 요약")
    print("=" * 60)
    print(f"   {'모델':<15} {'크기':<12} {'추론 시간':<15} {'속도 향상':<10} {'정확도':<10}")
    print(f"   {'-' * 62}")
    print(f"   {'원본 sklearn':<15} {original_size:.2f} KB{'':<4} {original_time:.4f} ms{'':<6} {'1.0x':<10} {test_accuracy:.4f}")
    print(f"   {'ONNX':<15} {onnx_size:.2f} KB{'':<4} {onnx_time:.4f} ms{'':<6} {onnx_speedup:.1f}x{'':<6} {onnx_accuracy:.4f}")
    print(f"   {'양자화':<15} {quant_size:.2f} KB{'':<4} {quant_time:.4f} ms{'':<6} {quant_speedup:.1f}x{'':<6} {quant_accuracy:.4f}")
    print("=" * 60)
    
    print("\n✅ Lab 3-3 실습 완료!")
    print("   MLflow UI에서 실험 결과를 확인하세요.")

if __name__ == "__main__":
    main()
