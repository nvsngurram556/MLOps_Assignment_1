"""
FastAPI application for California Housing price prediction.
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    HousingFeatures,
    ModelMetrics,
    PredictionRequest,
    PredictionResponse,
)
from utils.data_loader import DataLoader

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# Prometheus metrics
PREDICTION_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["model_version", "endpoint"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds", "Prediction latency", ["model_version", "endpoint"]
)
ERROR_COUNT = Counter(
    "prediction_errors_total", "Total prediction errors", ["error_type"]
)

# Global variables
model = None
scaler = None
model_metadata = {}
app_start_time = time.time()


class ModelService:
    """Service for handling model operations."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = {}
        self.feature_names = DataLoader().get_feature_names()

    def load_model(self, model_path: str = "artifacts/best_model.joblib") -> None:
        """Load the trained model and scaler."""
        try:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("Model loaded successfully", model_path=model_path)
            else:
                logger.warning(
                    "Model file not found, using fallback", model_path=model_path
                )
                # Fallback to any available model
                models_dir = "models"
                if os.path.exists(models_dir):
                    model_files = [
                        f for f in os.listdir(models_dir) if f.endswith(".joblib")
                    ]
                    if model_files:
                        fallback_path = os.path.join(models_dir, model_files[0])
                        self.model = joblib.load(fallback_path)
                        logger.info("Fallback model loaded", model_path=fallback_path)

            # Load scaler
            scaler_path = "data/scaler.joblib"
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded successfully")

            # Load metadata
            metadata_path = "artifacts/model_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
                logger.info("Model metadata loaded")
            else:
                self.metadata = {
                    "model_name": "unknown",
                    "val_rmse": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "feature_names": self.feature_names,
                }

        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise

    def preprocess_features(self, features: HousingFeatures) -> np.ndarray:
        """Preprocess input features for prediction."""
        # Convert to list
        feature_values = [
            features.MedInc,
            features.HouseAge,
            features.AveRooms,
            features.AveBedrms,
            features.Population,
            features.AveOccup,
            features.Latitude,
            features.Longitude,
        ]

        # Add engineered features
        rooms_per_household = features.AveRooms / features.AveOccup
        bedrooms_per_room = features.AveBedrms / features.AveRooms
        population_per_household = features.Population / features.AveOccup

        feature_values.extend(
            [rooms_per_household, bedrooms_per_room, population_per_household]
        )

        # Convert to numpy array and reshape
        feature_array = np.array(feature_values).reshape(1, -1)

        # Scale features if scaler is available
        if self.scaler is not None:
            feature_array = self.scaler.transform(feature_array)

        return feature_array

    def predict(self, features: HousingFeatures) -> float:
        """Make a single prediction."""
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        feature_array = self.preprocess_features(features)
        prediction = self.model.predict(feature_array)[0]

        return float(prediction)

    def predict_batch(self, features_list: List[HousingFeatures]) -> List[float]:
        """Make batch predictions."""
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        feature_arrays = []
        for features in features_list:
            feature_array = self.preprocess_features(features)
            feature_arrays.append(feature_array[0])  # Remove batch dimension

        batch_array = np.array(feature_arrays)
        predictions = self.model.predict(batch_array)

        return [float(pred) for pred in predictions]


# Initialize model service
model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting California Housing Prediction API")
    try:
        model_service.load_model()
        logger.info("Model service initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize model service", error=str(e))

    yield

    # Shutdown
    logger.info("Shutting down California Housing Prediction API")


# Initialize FastAPI app
app = FastAPI(
    title="California Housing Price Prediction API",
    description="Machine learning API for predicting California housing prices with MLOps best practices",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def log_request(
    request_data: Dict[str, Any], response_data: Dict[str, Any], endpoint: str
):
    """Log prediction request and response."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "request": request_data,
        "response": response_data,
    }

    # Log to structured logger
    logger.info("Prediction made", **log_entry)

    # TODO: Store in database for analysis
    # This could be implemented with SQLAlchemy and a database


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "California Housing Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    current_time = datetime.now().isoformat()
    uptime = time.time() - app_start_time

    return HealthResponse(
        status="healthy",
        timestamp=current_time,
        model_loaded=model_service.model is not None,
        model_version=model_service.metadata.get("model_name", "unknown"),
        uptime_seconds=uptime,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make a single prediction."""
    start_time = time.time()

    try:
        # Make prediction
        prediction = model_service.predict(request.features)

        # Create response
        response = PredictionResponse(
            prediction=prediction,
            model_version=model_service.metadata.get("model_name", "unknown"),
            timestamp=datetime.now().isoformat(),
        )

        # Log request and response
        background_tasks.add_task(
            log_request, request.dict(), response.dict(), "predict_single"
        )

        # Update metrics
        PREDICTION_COUNT.labels(
            model_version=response.model_version, endpoint="predict"
        ).inc()

        PREDICTION_LATENCY.labels(
            model_version=response.model_version, endpoint="predict"
        ).observe(time.time() - start_time)

        return response

    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest, background_tasks: BackgroundTasks
):
    """Make batch predictions."""
    start_time = time.time()

    try:
        # Make predictions
        predictions = model_service.predict_batch(request.features_list)

        # Create response
        response = BatchPredictionResponse(
            predictions=predictions,
            model_version=model_service.metadata.get("model_name", "unknown"),
            timestamp=datetime.now().isoformat(),
            count=len(predictions),
        )

        # Log request and response
        background_tasks.add_task(
            log_request,
            {"count": len(request.features_list)},
            {"count": len(predictions), "model_version": response.model_version},
            "predict_batch",
        )

        # Update metrics
        PREDICTION_COUNT.labels(
            model_version=response.model_version, endpoint="predict_batch"
        ).inc(len(predictions))

        PREDICTION_LATENCY.labels(
            model_version=response.model_version, endpoint="predict_batch"
        ).observe(time.time() - start_time)

        return response

    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        logger.error("Batch prediction failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", response_model=ModelMetrics)
async def get_model_info():
    """Get information about the current model."""
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metadata = model_service.metadata

    return ModelMetrics(
        model_name=metadata.get("model_name", "unknown"),
        model_version="1.0.0",
        rmse=metadata.get("val_rmse", 0.0),
        mae=0.0,  # TODO: Add to metadata
        r2=0.0,  # TODO: Add to metadata
        training_date=metadata.get("timestamp", datetime.now().isoformat()),
    )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return JSONResponse(
        content=generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST
    )


@app.post("/model/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining (placeholder)."""
    # This would trigger a retraining pipeline
    background_tasks.add_task(log_retrain_request)

    return {
        "message": "Retraining triggered",
        "timestamp": datetime.now().isoformat(),
        "status": "queued",
    }


async def log_retrain_request():
    """Log retraining request."""
    logger.info("Model retraining triggered")
    # TODO: Implement actual retraining logic


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail, timestamp=datetime.now().isoformat()
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    ERROR_COUNT.labels(error_type=type(exc).__name__).inc()
    logger.error("Unexpected error", error=str(exc))

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc),
            timestamp=datetime.now().isoformat(),
        ).dict(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
