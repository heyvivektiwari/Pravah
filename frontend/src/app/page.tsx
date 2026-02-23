"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  type PredictionInput,
  type PredictionResult,
  predictPrice,
  getLocations,
} from "@/lib/api";

/** Default form values for initial state. */
const DEFAULT_FORM: PredictionInput = {
  location: "kharghar",
  area_sqft: 1000,
  bhk: 2,
  bathrooms: 2,
  floor: 10,
  total_floors: 20,
  age_of_property: 5,
  parking: 1,
  lift: 1,
};

/**
 * Home page component with the prediction form and results display.
 */
export default function HomePage() {
  const [form, setForm] = useState<PredictionInput>(DEFAULT_FORM);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [locations, setLocations] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch locations on mount.
  useEffect(() => {
    getLocations()
      .then(setLocations)
      .catch(() => {
        // Fallback locations if API is down.
        setLocations([
          "airoli",
          "belapur",
          "cbd belapur",
          "ghansoli",
          "kharghar",
          "nerul",
          "panvel",
          "ulwe",
          "vashi",
        ]);
      });
  }, []);

  /** Update a form field value. */
  const updateField = useCallback(
    (field: keyof PredictionInput, value: string | number) => {
      setForm((prev) => ({ ...prev, [field]: value }));
    },
    []
  );

  /** Submit prediction request. */
  const handlePredict = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const prediction = await predictPrice(form);
      setResult(prediction);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Prediction failed. Check API."
      );
    } finally {
      setLoading(false);
    }
  }, [form]);

  return (
    <>
      {/* Navigation */}
      <nav className="navbar">
        <Link href="/" className="logo">
          <span className="logo-icon">🏠</span>
          <span>PricePredict</span>
        </Link>
        <ul className="nav-links">
          <li>
            <Link href="/" className="active">
              Predict
            </Link>
          </li>
          <li>
            <Link href="/analytics">Analytics</Link>
          </li>
        </ul>
      </nav>

      {/* Hero */}
      <section className="hero animate-fade-in">
        <div className="hero-badge">
          <span className="dot" />
          ML-Powered Price Estimation
        </div>
        <h1>
          Navi Mumbai
          <br />
          <span className="gradient-text">House Price Predictor</span>
        </h1>
        <p>
          Get instant, AI-driven property valuations across 9 prime Navi Mumbai
          locations. Powered by Gradient Boosting with 86% accuracy.
        </p>
      </section>

      {/* Main Content */}
      <main className="main-content">
        {/* Prediction Form */}
        <div className="glass-card animate-fade-in delay-1">
          <div className="card-header">
            <div className="icon teal">📊</div>
            <div>
              <h2>Property Details</h2>
              <p>Enter property specifications for valuation</p>
            </div>
          </div>

          <div className="form-grid">
            {/* Location */}
            <div className="form-group full-width">
              <label>Location</label>
              <select
                value={form.location}
                onChange={(e) => updateField("location", e.target.value)}
              >
                {locations.map((loc) => (
                  <option key={loc} value={loc}>
                    {loc.charAt(0).toUpperCase() + loc.slice(1)}
                  </option>
                ))}
              </select>
            </div>

            {/* Area */}
            <div className="form-group">
              <label>Area (sq ft)</label>
              <input
                type="number"
                value={form.area_sqft}
                onChange={(e) =>
                  updateField("area_sqft", parseFloat(e.target.value) || 0)
                }
                placeholder="e.g. 1000"
                min={100}
                max={5000}
              />
            </div>

            {/* BHK */}
            <div className="form-group">
              <label>BHK</label>
              <select
                value={form.bhk}
                onChange={(e) => updateField("bhk", parseInt(e.target.value))}
              >
                {[1, 2, 3, 4, 5, 6].map((n) => (
                  <option key={n} value={n}>
                    {n} BHK
                  </option>
                ))}
              </select>
            </div>

            {/* Bathrooms */}
            <div className="form-group">
              <label>Bathrooms</label>
              <select
                value={form.bathrooms}
                onChange={(e) =>
                  updateField("bathrooms", parseInt(e.target.value))
                }
              >
                {[1, 2, 3, 4, 5, 6].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </div>

            {/* Floor */}
            <div className="form-group">
              <label>Floor</label>
              <input
                type="number"
                value={form.floor}
                onChange={(e) =>
                  updateField("floor", parseInt(e.target.value) || 0)
                }
                placeholder="e.g. 10"
                min={0}
                max={100}
              />
            </div>

            {/* Total Floors */}
            <div className="form-group">
              <label>Total Floors</label>
              <input
                type="number"
                value={form.total_floors}
                onChange={(e) =>
                  updateField("total_floors", parseInt(e.target.value) || 1)
                }
                placeholder="e.g. 20"
                min={1}
                max={100}
              />
            </div>

            {/* Age */}
            <div className="form-group">
              <label>Property Age (yrs)</label>
              <input
                type="number"
                value={form.age_of_property}
                onChange={(e) =>
                  updateField(
                    "age_of_property",
                    parseFloat(e.target.value) || 0
                  )
                }
                placeholder="e.g. 5"
                min={0}
                max={50}
                step={0.5}
              />
            </div>

            {/* Parking */}
            <div className="form-group">
              <label>Parking</label>
              <div className="toggle-group">
                <button
                  type="button"
                  className={`toggle-btn ${form.parking === 1 ? "active" : ""}`}
                  onClick={() => updateField("parking", 1)}
                >
                  ✓ Yes
                </button>
                <button
                  type="button"
                  className={`toggle-btn ${form.parking === 0 ? "active" : ""}`}
                  onClick={() => updateField("parking", 0)}
                >
                  ✗ No
                </button>
              </div>
            </div>

            {/* Lift */}
            <div className="form-group">
              <label>Lift</label>
              <div className="toggle-group">
                <button
                  type="button"
                  className={`toggle-btn ${form.lift === 1 ? "active" : ""}`}
                  onClick={() => updateField("lift", 1)}
                >
                  ✓ Yes
                </button>
                <button
                  type="button"
                  className={`toggle-btn ${form.lift === 0 ? "active" : ""}`}
                  onClick={() => updateField("lift", 0)}
                >
                  ✗ No
                </button>
              </div>
            </div>
          </div>

          {/* Submit */}
          <button
            className="predict-btn"
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="loading-spinner" />
                Predicting...
              </>
            ) : (
              "🔮 Predict Price"
            )}
          </button>

          {error && (
            <div className="error-message">
              <span>⚠</span> {error}
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="results-panel animate-slide-in delay-2">
          {result ? (
            <div className="glass-card">
              <div className="card-header">
                <div className="icon blue">💰</div>
                <div>
                  <h2>Predicted Valuation</h2>
                  <p>
                    {result.input_summary.area_sqft} sq ft • {result.input_summary.bhk} BHK •{" "}
                    {result.input_summary.location.charAt(0).toUpperCase() +
                      result.input_summary.location.slice(1)}
                  </p>
                </div>
              </div>

              <div className="price-display">
                <div className="price-label">Estimated Market Value</div>
                <div className="price-amount">
                  {result.predicted_price_formatted}
                </div>
                <div className="price-confidence">
                  Range: {result.confidence_range_formatted.lower} —{" "}
                  {result.confidence_range_formatted.upper}
                </div>
              </div>

              <div className="insights-grid">
                <div className="insight-card">
                  <div className="insight-label">Price / sq ft</div>
                  <div className="insight-value teal">
                    {result.price_per_sqft_formatted}
                  </div>
                </div>
                <div className="insight-card">
                  <div className="insight-label">Location Avg</div>
                  <div className="insight-value blue">
                    {result.location_avg_price_formatted}
                  </div>
                </div>
                <div className="insight-card">
                  <div className="insight-label">Lower Bound</div>
                  <div className="insight-value purple">
                    {result.confidence_range_formatted.lower}
                  </div>
                </div>
                <div className="insight-card">
                  <div className="insight-label">Upper Bound</div>
                  <div className="insight-value gold">
                    {result.confidence_range_formatted.upper}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="glass-card">
              <div className="empty-state">
                <div className="empty-state-icon">🏙️</div>
                <h3>Ready to Predict</h3>
                <p>
                  Fill in the property details on the left and click
                  &quot;Predict Price&quot; to get an instant AI-powered
                  valuation for any Navi Mumbai property.
                </p>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        Built with FastAPI + Next.js • Gradient Boosting ML Model •
        Pravah Project © {new Date().getFullYear()}
      </footer>
    </>
  );
}
