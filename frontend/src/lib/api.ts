/**
 * @fileoverview API client for Navi Mumbai House Price Predictor backend.
 * Handles all HTTP communication with the FastAPI prediction service.
 */

/** Base URL for the API, configured via environment variable. */
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";

/** Input fields for a price prediction request. */
export interface PredictionInput {
  location: string;
  area_sqft: number;
  bhk: number;
  bathrooms: number;
  floor: number;
  total_floors: number;
  age_of_property: number;
  parking: number;
  lift: number;
}

/** Response from the prediction endpoint. */
export interface PredictionResult {
  predicted_price: number;
  predicted_price_formatted: string;
  price_per_sqft: number;
  price_per_sqft_formatted: string;
  confidence_range: { lower: number; upper: number };
  confidence_range_formatted: { lower: string; upper: string };
  location_avg_price: number;
  location_avg_price_formatted: string;
  input_summary: {
    location: string;
    area_sqft: number;
    bhk: number;
    bathrooms: number;
    floor: number;
    total_floors: number;
    age_of_property: number;
    parking: boolean;
    lift: boolean;
  };
}

/** Location-level price statistics. */
export interface LocationStats {
  mean_price: number;
  median_price: number;
  min_price: number;
  max_price: number;
  count: number;
  avg_price_per_sqft: number;
}

/** Full model metadata response. */
export interface ModelInfo {
  model_name: string;
  metrics: {
    r2_score: number;
    rmse: number;
    mae: number;
    train_score: number;
    test_score: number;
  };
  feature_importance: Record<string, number>;
  locations: string[];
  location_stats: Record<string, LocationStats>;
  total_training_samples: number;
}

/**
 * Fetch predictions from the API.
 *
 * @param input - Property features for prediction.
 * @returns The prediction result with price and insights.
 * @throws Error if the request fails.
 */
export async function predictPrice(
  input: PredictionInput
): Promise<PredictionResult> {
  const response = await fetch(`${API_BASE_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(
      error.detail || `Prediction failed with status ${response.status}`
    );
  }

  return response.json();
}

/**
 * Fetch list of supported locations.
 *
 * @returns Array of location names.
 */
export async function getLocations(): Promise<string[]> {
  const response = await fetch(`${API_BASE_URL}/api/locations`);
  if (!response.ok) {
    throw new Error("Failed to fetch locations");
  }
  return response.json();
}

/**
 * Fetch model information and metrics.
 *
 * @returns Model metadata including metrics, locations, and feature importance.
 */
export async function getModelInfo(): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE_URL}/api/model-info`);
  if (!response.ok) {
    throw new Error("Failed to fetch model info");
  }
  return response.json();
}

/**
 * Check API health status.
 *
 * @returns Health status object.
 */
export async function checkHealth(): Promise<{
  status: string;
  model_loaded: boolean;
  version: string;
}> {
  const response = await fetch(`${API_BASE_URL}/api/health`);
  if (!response.ok) {
    throw new Error("Health check failed");
  }
  return response.json();
}
