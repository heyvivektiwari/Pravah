"""Navi Mumbai House Price Predictor - Model Training Script.

Trains a Gradient Boosting Regressor on Navi Mumbai real estate data
and exports the model pipeline (model + scaler + encoder) for serving.

Usage:
    python train_model.py

Outputs:
    model.pkl - Serialized model pipeline via joblib
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load CSV and apply data cleaning rules.

    Args:
        filepath: Path to the cleaned CSV dataset.

    Returns:
        Cleaned pandas DataFrame.
    """
    df = pd.read_csv(filepath)

    # Remove rows with negative or zero area/price (data artifacts).
    df = df[df["area_sqft"] > 0]
    df = df[df["actual_price"] > 0]

    # Cap outliers: remove rows where area > 3000 sqft (likely data errors).
    df = df[df["area_sqft"] <= 3000]

    # Round non-integer BHK/bathrooms to nearest value.
    df["bhk"] = df["bhk"].round().astype(int)
    df["bathrooms"] = df["bathrooms"].round().astype(int)

    # Normalize location names.
    df["location"] = df["location"].str.strip().str.lower()

    # Ensure parking and lift are binary.
    df["parking"] = df["parking"].astype(int)
    df["lift"] = df["lift"].astype(int)

    print(f"Dataset loaded: {len(df)} rows after cleaning.")
    return df


def train_model(df: pd.DataFrame) -> dict:
    """Train Gradient Boosting Regressor and return model artifacts.

    Args:
        df: Cleaned DataFrame with all features and target.

    Returns:
        Dictionary containing trained model, scaler, encoder, and metadata.
    """
    features = [
        "location",
        "area_sqft",
        "bhk",
        "bathrooms",
        "floor",
        "total_floors",
        "age_of_property",
        "parking",
        "lift",
    ]
    target = "actual_price"

    x_data = df[features].copy()
    y_data = df[target].copy()

    # Encode categorical 'location' column.
    label_encoder = LabelEncoder()
    x_data["location"] = label_encoder.fit_transform(x_data["location"])

    # Scale numerical features.
    scaler = StandardScaler()
    x_scaled = pd.DataFrame(
        scaler.fit_transform(x_data),
        columns=x_data.columns,
        index=x_data.index,
    )

    # Split data: 80% train, 20% test.
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y_data, test_size=0.2, random_state=42
    )

    # Train Gradient Boosting Regressor with tuned hyperparameters.
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.9,
        random_state=42,
    )
    model.fit(x_train, y_train)

    # Evaluate on test set.
    y_pred = model.predict(x_test)
    metrics = {
        "r2_score": round(r2_score(y_test, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
        "train_score": round(model.score(x_train, y_train), 4),
        "test_score": round(model.score(x_test, y_test), 4),
    }

    # Feature importance.
    importance = dict(zip(features, model.feature_importances_.tolist()))

    # Location statistics for insights.
    location_stats = {}
    for loc in label_encoder.classes_:
        loc_data = df[df["location"] == loc]["actual_price"]
        location_stats[loc] = {
            "mean_price": round(loc_data.mean(), 2),
            "median_price": round(loc_data.median(), 2),
            "min_price": round(loc_data.min(), 2),
            "max_price": round(loc_data.max(), 2),
            "count": int(len(loc_data)),
            "avg_price_per_sqft": round(
                (df[df["location"] == loc]["actual_price"]
                 / df[df["location"] == loc]["area_sqft"]).mean(), 2
            ),
        }

    print("\n=== Model Training Results ===")
    print(f"R² Score:     {metrics['r2_score']}")
    print(f"RMSE:         ₹{metrics['rmse']:,.0f}")
    print(f"MAE:          ₹{metrics['mae']:,.0f}")
    print(f"Train Score:  {metrics['train_score']}")
    print(f"Test Score:   {metrics['test_score']}")
    print("\n=== Feature Importance ===")
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:20s}: {imp:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "features": features,
        "target": target,
        "metrics": metrics,
        "feature_importance": importance,
        "location_classes": label_encoder.classes_.tolist(),
        "location_stats": location_stats,
    }


def save_model(model_artifacts: dict, output_path: str) -> None:
    """Save model artifacts to disk using joblib.

    Args:
        model_artifacts: Dictionary with model, scaler, encoder, metadata.
        output_path: File path for the saved model.
    """
    joblib.dump(model_artifacts, output_path)
    file_size = os.path.getsize(output_path)
    print(f"\nModel saved to: {output_path} ({file_size / 1024:.1f} KB)")


def main() -> None:
    """Main entry point for model training."""
    # Locate the CSV dataset.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try looking in the current directory (Docker environment)
    docker_csv_path = os.path.join(script_dir, "navi_mumbai_real_estate_uncleaned_2500_cleaned.csv")
    # Try looking in the parent directory (local development)
    local_csv_path = os.path.join(script_dir, "..", "navi_mumbai_real_estate_uncleaned_2500_cleaned.csv")
    
    if os.path.exists(docker_csv_path):
        csv_path = docker_csv_path
    elif os.path.exists(local_csv_path):
        csv_path = local_csv_path
    else:
        print(f"Error: Dataset not found. Checked {docker_csv_path} and {local_csv_path}")
        sys.exit(1)

    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        sys.exit(1)

    df = load_and_clean_data(csv_path)
    model_artifacts = train_model(df)
    save_model(model_artifacts, os.path.join(script_dir, "model.pkl"))

    print("\nModel training complete! Ready for deployment.")


if __name__ == "__main__":
    main()
