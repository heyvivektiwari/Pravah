"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { type ModelInfo, getModelInfo } from "@/lib/api";

/**
 * Format a number in Indian Rupee format (Lakhs/Crores).
 */
function formatINR(amount: number): string {
    if (amount >= 1_00_00_000) return `₹${(amount / 1_00_00_000).toFixed(2)} Cr`;
    if (amount >= 1_00_000) return `₹${(amount / 1_00_000).toFixed(2)} Lakh`;
    return `₹${amount.toLocaleString("en-IN")}`;
}

/**
 * Analytics dashboard showing model metrics, feature importance,
 * and location-wise price statistics.
 */
export default function AnalyticsPage() {
    const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        getModelInfo()
            .then(setModelInfo)
            .catch(console.error)
            .finally(() => setLoading(false));
    }, []);

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
                        <Link href="/">Predict</Link>
                    </li>
                    <li>
                        <Link href="/analytics" className="active">
                            Analytics
                        </Link>
                    </li>
                </ul>
            </nav>

            {/* Header */}
            <section className="analytics-header animate-fade-in">
                <h1>
                    Model <span className="gradient-text">Analytics</span>
                </h1>
                <p>
                    Performance metrics, feature analysis, and location insights for the
                    Gradient Boosting prediction model.
                </p>
            </section>

            <div className="analytics-content">
                {loading ? (
                    <div className="glass-card" style={{ textAlign: "center", padding: "64px" }}>
                        <div className="loading-spinner" style={{ width: 32, height: 32, margin: "0 auto 16px", border: "3px solid rgba(255,255,255,0.1)", borderTopColor: "var(--accent-teal)", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
                        <p style={{ color: "var(--text-secondary)" }}>Loading model data...</p>
                    </div>
                ) : modelInfo ? (
                    <>
                        {/* Metrics Cards */}
                        <div className="metrics-grid animate-fade-in delay-1">
                            <div className="metric-card">
                                <div className="metric-icon">🎯</div>
                                <div className="metric-value teal">
                                    {(modelInfo.metrics.r2_score * 100).toFixed(1)}%
                                </div>
                                <div className="metric-label">R² Accuracy</div>
                            </div>
                            <div className="metric-card">
                                <div className="metric-icon">📉</div>
                                <div className="metric-value blue">
                                    {formatINR(modelInfo.metrics.rmse)}
                                </div>
                                <div className="metric-label">RMSE</div>
                            </div>
                            <div className="metric-card">
                                <div className="metric-icon">📊</div>
                                <div className="metric-value purple">
                                    {formatINR(modelInfo.metrics.mae)}
                                </div>
                                <div className="metric-label">MAE</div>
                            </div>
                            <div className="metric-card">
                                <div className="metric-icon">🗂️</div>
                                <div className="metric-value gold">
                                    {modelInfo.total_training_samples.toLocaleString()}
                                </div>
                                <div className="metric-label">Training Samples</div>
                            </div>
                            <div className="metric-card">
                                <div className="metric-icon">📍</div>
                                <div className="metric-value pink">
                                    {modelInfo.locations.length}
                                </div>
                                <div className="metric-label">Locations</div>
                            </div>
                        </div>

                        {/* Feature Importance */}
                        <div className="glass-card animate-fade-in delay-2" style={{ marginBottom: 32 }}>
                            <div className="card-header">
                                <div className="icon teal">⚡</div>
                                <div>
                                    <h2>Feature Importance</h2>
                                    <p>
                                        How much each property attribute influences the predicted
                                        price
                                    </p>
                                </div>
                            </div>
                            <div className="feature-list">
                                {Object.entries(modelInfo.feature_importance)
                                    .sort(([, a], [, b]) => b - a)
                                    .map(([feature, importance]) => (
                                        <div key={feature} className="feature-item">
                                            <span className="feature-name">
                                                {feature.replace(/_/g, " ")}
                                            </span>
                                            <div className="feature-bar-container">
                                                <div
                                                    className="feature-bar"
                                                    style={{
                                                        width: `${(importance * 100).toFixed(1)}%`,
                                                    }}
                                                />
                                            </div>
                                            <span className="feature-value">
                                                {(importance * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    ))}
                            </div>
                        </div>

                        {/* Location Stats */}
                        <div className="glass-card animate-fade-in delay-3">
                            <div className="card-header">
                                <div className="icon blue">📍</div>
                                <div>
                                    <h2>Location-wise Price Analysis</h2>
                                    <p>Average pricing and data coverage by area</p>
                                </div>
                            </div>
                            <div style={{ overflowX: "auto" }}>
                                <table className="location-table">
                                    <thead>
                                        <tr>
                                            <th>Location</th>
                                            <th>Avg Price</th>
                                            <th>Median Price</th>
                                            <th>₹/sq ft</th>
                                            <th>Samples</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {Object.entries(modelInfo.location_stats)
                                            .sort(([, a], [, b]) => b.mean_price - a.mean_price)
                                            .map(([location, stats]) => (
                                                <tr key={location}>
                                                    <td className="location-name">{location}</td>
                                                    <td className="location-price">
                                                        {formatINR(stats.mean_price)}
                                                    </td>
                                                    <td className="location-price">
                                                        {formatINR(stats.median_price)}
                                                    </td>
                                                    <td style={{ color: "var(--text-secondary)" }}>
                                                        {formatINR(stats.avg_price_per_sqft)}
                                                    </td>
                                                    <td className="location-count">{stats.count}</td>
                                                </tr>
                                            ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </>
                ) : (
                    <div className="glass-card" style={{ textAlign: "center", padding: "48px" }}>
                        <div className="empty-state-icon">⚠️</div>
                        <h3 style={{ color: "var(--text-secondary)", marginBottom: 8 }}>
                            Could not load model data
                        </h3>
                        <p style={{ color: "var(--text-muted)" }}>
                            Make sure the backend API is running at the configured URL.
                        </p>
                    </div>
                )}
            </div>

            {/* Footer */}
            <footer className="footer">
                Built with FastAPI + Next.js • Gradient Boosting ML Model •
                Pravah Project © {new Date().getFullYear()}
            </footer>
        </>
    );
}
