"""Navi Mumbai House Price Predictor - FastAPI Backend.

Production-ready REST API for serving house price predictions
using a trained Gradient Boosting model.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

# Configure structured logging (Google Cloud Logging compatible).
logging.basicConfig(
    level=logging.INFO,
    format='{"severity":"%(levelname)s","message":"%(message)s","timestamp":"%(asctime)s"}',
)
logger = logging.getLogger(__name__)

# Global model artifacts.
model_artifacts: dict[str, Any] = {}


def load_model() -> dict[str, Any]:
    """Load the serialized model pipeline from disk.

    Returns:
        Dictionary containing model, scaler, encoder, and metadata.

    Raises:
        FileNotFoundError: If model.pkl is not found.
    """
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run train_model.py first."
        )
    artifacts = joblib.load(model_path)
    logger.info("Model loaded successfully from %s", model_path)
    return artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for model loading."""
    global model_artifacts
    try:
        model_artifacts = load_model()
        logger.info("Model ready. Locations: %s", model_artifacts["location_classes"])
    except FileNotFoundError as e:
        logger.error("Failed to load model: %s", e)
    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title="Navi Mumbai House Price Predictor API",
    description="ML-powered REST API for predicting house prices in Navi Mumbai.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for frontend.
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ──────────────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    """Request schema for house price prediction."""

    location: str = Field(
        ..., description="Location in Navi Mumbai", examples=["kharghar"]
    )
    area_sqft: float = Field(
        ..., gt=0, le=5000, description="Area in square feet", examples=[1000.0]
    )
    bhk: int = Field(
        ..., ge=1, le=6, description="Number of bedrooms", examples=[2]
    )
    bathrooms: int = Field(
        ..., ge=1, le=6, description="Number of bathrooms", examples=[2]
    )
    floor: int = Field(
        ..., ge=0, le=100, description="Floor number", examples=[10]
    )
    total_floors: int = Field(
        ..., ge=1, le=100, description="Total floors in building", examples=[20]
    )
    age_of_property: float = Field(
        ..., ge=0, le=50, description="Age of property in years", examples=[5.0]
    )
    parking: int = Field(
        ..., ge=0, le=1, description="Parking available (0 or 1)", examples=[1]
    )
    lift: int = Field(
        ..., ge=0, le=1, description="Lift available (0 or 1)", examples=[1]
    )


class PredictionResponse(BaseModel):
    """Response schema for house price prediction."""

    predicted_price: float
    predicted_price_formatted: str
    price_per_sqft: float
    price_per_sqft_formatted: str
    confidence_range: dict[str, float]
    confidence_range_formatted: dict[str, str]
    location_avg_price: float
    location_avg_price_formatted: str
    input_summary: dict[str, Any]


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Response schema for model information endpoint."""

    model_name: str
    metrics: dict[str, float]
    feature_importance: dict[str, float]
    locations: list[str]
    location_stats: dict[str, Any]
    total_training_samples: int


# ── Helper Functions ─────────────────────────────────────────────────────────


def format_inr(amount: float) -> str:
    """Format a number into Indian Rupee format with lakhs/crores.

    Args:
        amount: The amount in INR.

    Returns:
        Formatted string like '₹45.5 Lakh' or '₹1.2 Cr'.
    """
    if amount >= 1_00_00_000:
        return f"₹{amount / 1_00_00_000:.2f} Cr"
    elif amount >= 1_00_000:
        return f"₹{amount / 1_00_000:.2f} Lakh"
    else:
        return f"₹{amount:,.0f}"


# ── API Endpoints ────────────────────────────────────────────────────────────


@app.get("/")
async def root():
    """Root endpoint returning API status and links to documentation."""
    return {
        "message": "Welcome to Navi Mumbai House Price Predictor API",
        "docs_url": "/docs",
        "health_check": "/api/health"
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for monitoring and readiness probes."""
    return HealthResponse(
        status="healthy" if model_artifacts else "degraded",
        model_loaded=bool(model_artifacts),
        version="1.0.0",
    )


@app.get("/api/locations", response_model=list[str])
async def get_locations() -> list[str]:
    """Return list of supported Navi Mumbai locations."""
    if not model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return sorted(model_artifacts["location_classes"])


@app.get("/api/model-info", response_model=ModelInfoResponse)
async def get_model_info() -> ModelInfoResponse:
    """Return model metadata, metrics, and feature importance."""
    if not model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return ModelInfoResponse(
        model_name="Gradient Boosting Regressor",
        metrics=model_artifacts["metrics"],
        feature_importance=model_artifacts["feature_importance"],
        locations=sorted(model_artifacts["location_classes"]),
        location_stats=model_artifacts["location_stats"],
        total_training_samples=sum(
            s["count"] for s in model_artifacts["location_stats"].values()
        ),
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest) -> PredictionResponse:
    """Predict house price based on property features.

    Args:
        request: Property features for prediction.

    Returns:
        Predicted price with confidence interval and insights.

    Raises:
        HTTPException: If model not loaded or invalid location.
    """
    if not model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    start_time = time.time()

    # Validate location.
    location_lower = request.location.strip().lower()
    if location_lower not in model_artifacts["location_classes"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown location: '{request.location}'. "
            f"Supported: {sorted(model_artifacts['location_classes'])}",
        )

    # Prepare input features.
    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    label_encoder = model_artifacts["label_encoder"]
    features = model_artifacts["features"]

    input_data = pd.DataFrame([{
        "location": location_lower,
        "area_sqft": request.area_sqft,
        "bhk": request.bhk,
        "bathrooms": request.bathrooms,
        "floor": request.floor,
        "total_floors": request.total_floors,
        "age_of_property": request.age_of_property,
        "parking": request.parking,
        "lift": request.lift,
    }])

    # Encode and scale.
    input_data["location"] = label_encoder.transform(input_data["location"])
    input_scaled = pd.DataFrame(
        scaler.transform(input_data[features]),
        columns=features,
    )

    # Predict.
    predicted_price = float(model.predict(input_scaled)[0])
    predicted_price = max(predicted_price, 0)  # Floor at zero.

    # Compute confidence interval (±15% based on model RMSE).
    rmse = model_artifacts["metrics"]["rmse"]
    confidence_lower = max(predicted_price - rmse, 0)
    confidence_upper = predicted_price + rmse

    # Price per sqft.
    price_per_sqft = predicted_price / request.area_sqft if request.area_sqft > 0 else 0

    # Location average price.
    loc_stats = model_artifacts["location_stats"].get(location_lower, {})
    location_avg = loc_stats.get("mean_price", 0)

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        "Prediction: %s, %.0f sqft, %d BHK → ₹%.0f (%.1fms)",
        location_lower, request.area_sqft, request.bhk, predicted_price, elapsed_ms,
    )

    return PredictionResponse(
        predicted_price=round(predicted_price, 2),
        predicted_price_formatted=format_inr(predicted_price),
        price_per_sqft=round(price_per_sqft, 2),
        price_per_sqft_formatted=format_inr(price_per_sqft),
        confidence_range={
            "lower": round(confidence_lower, 2),
            "upper": round(confidence_upper, 2),
        },
        confidence_range_formatted={
            "lower": format_inr(confidence_lower),
            "upper": format_inr(confidence_upper),
        },
        location_avg_price=round(location_avg, 2),
        location_avg_price_formatted=format_inr(location_avg),
        input_summary={
            "location": location_lower,
            "area_sqft": request.area_sqft,
            "bhk": request.bhk,
            "bathrooms": request.bathrooms,
            "floor": request.floor,
            "total_floors": request.total_floors,
            "age_of_property": request.age_of_property,
            "parking": bool(request.parking),
            "lift": bool(request.lift),
        },
    )
