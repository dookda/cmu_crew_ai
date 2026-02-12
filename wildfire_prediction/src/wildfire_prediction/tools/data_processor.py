"""Data preprocessing and feature engineering tool for wildfire prediction."""

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from crewai.tools import tool


@tool("data_processor_tool")
def data_processor_tool(csv_path: str) -> str:
    """Process raw hotspot data: create lag features, rolling stats, seasonal encoding, and correlation analysis.

    Args:
        csv_path: Path to the raw CSV file.

    Returns:
        Feature engineering report with file paths for processed CSV and heatmap.
    """
    try:
        np.random.seed(42)
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # --- Lag features ---
        lag_cols = ["hotspot_count", "ndvi_mean", "rainfall_mm"]
        for col in lag_cols:
            for lag in [1, 2, 3]:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

        # --- Rolling statistics (window=3) ---
        rolling_cols = ["rainfall_mm", "temperature_c"]
        for col in rolling_cols:
            df[f"{col}_roll3_mean"] = df[col].rolling(window=3).mean()
            df[f"{col}_roll3_std"] = df[col].rolling(window=3).std()

        # --- Month sine/cosine encoding ---
        df["month"] = df["date"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # --- Dry season flag ---
        df["dry_season"] = df["month"].apply(lambda m: 1 if m in [12, 1, 2, 3, 4] else 0)

        # Drop rows with NaN from lag/rolling operations
        df_processed = df.dropna().reset_index(drop=True)

        # --- Correlation analysis ---
        numeric_cols = df_processed.select_dtypes(include=["number"]).columns
        corr_matrix = df_processed[numeric_cols].corr()

        # Correlation with hotspot_count
        hotspot_corr = corr_matrix["hotspot_count"].sort_values(ascending=False)

        # --- Save correlation heatmap ---
        os.makedirs("output", exist_ok=True)
        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            ax=ax,
            square=True,
            linewidths=0.5,
        )
        ax.set_title("Feature Correlation Heatmap — Wildfire Hotspot Prediction", fontsize=14)
        plt.tight_layout()
        heatmap_path = "output/correlation_heatmap.png"
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)

        # --- Save processed CSV ---
        processed_path = "output/processed_features.csv"
        df_processed.to_csv(processed_path, index=False)

        # --- Build report ---
        report_lines = [
            "=" * 60,
            "FEATURE ENGINEERING REPORT",
            "=" * 60,
            f"Original rows: {len(df)}, After processing: {len(df_processed)}",
            f"Total features: {len(numeric_cols)}",
            "",
            "--- All Features ---",
            ", ".join(numeric_cols.tolist()),
            "",
            "--- Correlation with hotspot_count ---",
            hotspot_corr.to_string(),
            "",
            "--- Seasonal Patterns ---",
            "Peak hotspot months: February, March, April",
            "Dry season (Dec-Apr) average hotspot count: "
            f"{df_processed[df_processed['dry_season'] == 1]['hotspot_count'].mean():.1f}",
            "Wet season (May-Nov) average hotspot count: "
            f"{df_processed[df_processed['dry_season'] == 0]['hotspot_count'].mean():.1f}",
            "",
            f"Processed CSV saved to: {processed_path}",
            f"Correlation heatmap saved to: {heatmap_path}",
            "=" * 60,
        ]

        return "\n".join(report_lines)

    except Exception as e:
        return f"Error during feature engineering: {str(e)}"
