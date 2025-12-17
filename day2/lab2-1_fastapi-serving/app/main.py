"""
Lab 2-1: FastAPI 모델 서빙
==========================

Iris 분류 모델을 FastAPI로 서빙하는 REST API

Endpoints:
    - GET  /           : API 정보
    - GET  /health     : Health check
    - POST /predict    : 단일 예측
    - POST /predict/batch : 배치 예측
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import joblib
import numpy as np
from pathlib import Path
import logging
from contextlib import asynccontextmanager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# 앱 시작/종료 이벤트 (Lifespan Context Manager)
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 시작/종료 관리

    FastAPI 0.93.0+ 권장 방식
    """
    # 시작 (startup)
    logger.info("=" * 60)
    logger.info("  Iris Classification API 시작")
    logger.info("=" * 60)
    yield
    # 종료 (shutdown)
    logger.info("Iris Classification API 종료")


# FastAPI 앱 생성 (lifespan 이벤트 핸들러 포함)
app = FastAPI(
    title="Iris Classification API",
    description="Iris 꽃 분류를 위한 ML 모델 서빙 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# 모델 로드
MODEL_PATH = Path(__file__).parent.parent / "model.joblib"
IRIS_SPECIES = {0: "setosa", 1: "versicolor", 2: "virginica"}

try:
    logger.info(f"모델 로드 시도: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
    logger.info("✅ 모델 로드 성공")
except Exception as e:
    model = None
    MODEL_LOADED = False
    logger.error(f"❌ 모델 로드 실패: {e}")


# ============================================================
# Pydantic 모델 정의
# ============================================================


class IrisFeatures(BaseModel):
    """Iris 피처 입력 모델"""

    sepal_length: float = Field(
        ..., ge=0, le=10, description="꽃받침 길이 (cm)", example=5.1
    )
    sepal_width: float = Field(
        ..., ge=0, le=10, description="꽃받침 너비 (cm)", example=3.5
    )
    petal_length: float = Field(
        ..., ge=0, le=10, description="꽃잎 길이 (cm)", example=1.4
    )
    petal_width: float = Field(
        ..., ge=0, le=10, description="꽃잎 너비 (cm)", example=0.2
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }
    )


class PredictionResponse(BaseModel):
    """예측 결과 모델"""

    prediction: str = Field(..., description="예측된 Iris 품종")
    confidence: float = Field(..., description="예측 신뢰도 (0-1)")

    model_config = ConfigDict(
        json_schema_extra={"example": {"prediction": "setosa", "confidence": 0.98}}
    )


class BatchPredictionResponse(BaseModel):
    """배치 예측 결과 모델"""

    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check 응답 모델"""

    status: str
    is_model_loaded: bool = Field(alias="model_loaded")

    model_config = ConfigDict(populate_by_name=True)


class InfoResponse(BaseModel):
    """API 정보 응답 모델"""

    message: str
    version: str
    is_model_loaded: bool = Field(alias="model_loaded")

    model_config = ConfigDict(populate_by_name=True)


# ============================================================
# API 엔드포인트
# ============================================================


@app.get("/", response_model=InfoResponse)
async def root():
    """
    API 기본 정보 반환

    Returns:
        API 메시지, 버전, 모델 로드 상태
    """
    logger.info("API 정보 요청")
    return InfoResponse(
        message="Iris Classification API", version="1.0.0", is_model_loaded=MODEL_LOADED
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check 엔드포인트

    Kubernetes liveness/readiness probe에서 사용

    Returns:
        서비스 상태 및 모델 로드 여부
    """
    status = "healthy" if MODEL_LOADED else "unhealthy"
    logger.info(f"Health check: {status}")

    return HealthResponse(status=status, is_model_loaded=MODEL_LOADED)


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """
    단일 샘플 예측

    Args:
        features: Iris 꽃의 4가지 측정값

    Returns:
        예측된 품종과 신뢰도

    Raises:
        HTTPException: 모델이 로드되지 않은 경우 (503)
    """
    if not MODEL_LOADED:
        logger.error("예측 요청 실패: 모델이 로드되지 않음")
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please check server logs."
        )

    try:
        # 입력 데이터 변환
        input_data = np.array(
            [
                [
                    features.sepal_length,
                    features.sepal_width,
                    features.petal_length,
                    features.petal_width,
                ]
            ]
        )

        # 예측
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        result = PredictionResponse(
            prediction=IRIS_SPECIES[prediction],
            confidence=float(probabilities[prediction]),
        )

        logger.info(
            f"예측 성공: {result.prediction} " f"(신뢰도: {result.confidence:.3f})"
        )

        return result

    except Exception as e:
        logger.error(f"예측 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(features_list: List[IrisFeatures]):
    """
    배치 예측

    여러 샘플을 한 번에 예측

    Args:
        features_list: Iris 피처 리스트

    Returns:
        각 샘플에 대한 예측 결과 리스트

    Raises:
        HTTPException: 모델이 로드되지 않은 경우 (503)
    """
    if not MODEL_LOADED:
        logger.error("배치 예측 요청 실패: 모델이 로드되지 않음")
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please check server logs."
        )

    try:
        # 입력 데이터 변환
        input_data = np.array(
            [
                [f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]
                for f in features_list
            ]
        )

        # 배치 예측
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)

        # 결과 생성
        results = []
        for pred, probs in zip(predictions, probabilities):
            results.append(
                PredictionResponse(
                    prediction=IRIS_SPECIES[pred], confidence=float(probs[pred])
                )
            )

        logger.info(f"배치 예측 성공: {len(results)}개 샘플")

        return BatchPredictionResponse(predictions=results)

    except Exception as e:
        logger.error(f"배치 예측 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
