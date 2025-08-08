"""
Pydantic schemas for API input validation and responses.
"""

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, validator


class HousingFeatures(BaseModel):
    """Schema for housing features input."""

    MedInc: float = Field(
        ...,
        description="Median income in block group (in tens of thousands of dollars)",
        ge=0.5,
        le=15.0,
    )
    HouseAge: float = Field(
        ..., description="Median house age in block group (in years)", ge=1.0, le=52.0
    )
    AveRooms: float = Field(
        ..., description="Average number of rooms per household", ge=1.0, le=50.0
    )
    AveBedrms: float = Field(
        ..., description="Average number of bedrooms per household", ge=0.1, le=10.0
    )
    Population: float = Field(
        ..., description="Block group population", ge=3.0, le=35000.0
    )
    AveOccup: float = Field(
        ..., description="Average number of household members", ge=0.5, le=50.0
    )
    Latitude: float = Field(..., description="Block group latitude", ge=32.0, le=42.0)
    Longitude: float = Field(
        ..., description="Block group longitude", ge=-125.0, le=-114.0
    )

    @validator("AveBedrms")
    def bedrooms_reasonable(cls, v, values):
        """Validate that bedrooms are reasonable compared to total rooms."""
        if "AveRooms" in values and v > values["AveRooms"]:
            raise ValueError("Average bedrooms cannot exceed average rooms")
        return v

    class Config:
        schema_extra = {
            "example": {
                "MedInc": 8.3014,
                "HouseAge": 21.0,
                "AveRooms": 6.238137,
                "AveBedrms": 0.971880,
                "Population": 2401.0,
                "AveOccup": 2.109842,
                "Latitude": 37.86,
                "Longitude": -122.22,
            }
        }


class PredictionRequest(BaseModel):
    """Schema for prediction request."""

    features: HousingFeatures = Field(
        ..., description="Housing features for prediction"
    )

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "MedInc": 8.3014,
                    "HouseAge": 21.0,
                    "AveRooms": 6.238137,
                    "AveBedrms": 0.971880,
                    "Population": 2401.0,
                    "AveOccup": 2.109842,
                    "Latitude": 37.86,
                    "Longitude": -122.22,
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""

    features_list: List[HousingFeatures] = Field(
        ...,
        description="List of housing features for batch prediction",
        min_items=1,
        max_items=100,
    )

    class Config:
        schema_extra = {
            "example": {
                "features_list": [
                    {
                        "MedInc": 8.3014,
                        "HouseAge": 21.0,
                        "AveRooms": 6.238137,
                        "AveBedrms": 0.971880,
                        "Population": 2401.0,
                        "AveOccup": 2.109842,
                        "Latitude": 37.86,
                        "Longitude": -122.22,
                    },
                    {
                        "MedInc": 5.6431,
                        "HouseAge": 18.0,
                        "AveRooms": 5.817352,
                        "AveBedrms": 1.073446,
                        "Population": 2775.0,
                        "AveOccup": 2.611321,
                        "Latitude": 39.78,
                        "Longitude": -121.97,
                    },
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""

    prediction: float = Field(
        ..., description="Predicted house price (in hundreds of thousands of dollars)"
    )
    model_version: str = Field(
        ..., description="Version of the model used for prediction"
    )
    timestamp: str = Field(..., description="Timestamp of prediction")
    confidence_interval: Optional[List[float]] = Field(
        None, description="95% confidence interval [lower, upper]"
    )

    class Config:
        schema_extra = {
            "example": {
                "prediction": 4.526,
                "model_version": "1.0.0",
                "timestamp": "2024-01-01T12:00:00Z",
                "confidence_interval": [3.8, 5.2],
            }
        }


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""

    predictions: List[float] = Field(..., description="List of predicted house prices")
    model_version: str = Field(
        ..., description="Version of the model used for prediction"
    )
    timestamp: str = Field(..., description="Timestamp of prediction")
    count: int = Field(..., description="Number of predictions made")

    class Config:
        schema_extra = {
            "example": {
                "predictions": [4.526, 2.345],
                "model_version": "1.0.0",
                "timestamp": "2024-01-01T12:00:00Z",
                "count": 2,
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "model_loaded": True,
                "model_version": "1.0.0",
                "uptime_seconds": 3600.0,
            }
        }


class ModelMetrics(BaseModel):
    """Schema for model metrics response."""

    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Version of the model")
    rmse: float = Field(..., description="Root Mean Square Error")
    mae: float = Field(..., description="Mean Absolute Error")
    r2: float = Field(..., description="R-squared score")
    training_date: str = Field(..., description="Date when model was trained")

    class Config:
        schema_extra = {
            "example": {
                "model_name": "random_forest",
                "model_version": "1.0.0",
                "rmse": 0.65,
                "mae": 0.52,
                "r2": 0.60,
                "training_date": "2024-01-01T12:00:00Z",
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")

    class Config:
        schema_extra = {
            "example": {
                "error": "Validation error",
                "details": "MedInc must be between 0.5 and 15.0",
                "timestamp": "2024-01-01T12:00:00Z",
            }
        }
