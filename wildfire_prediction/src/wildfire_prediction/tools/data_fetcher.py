"""Data fetching and validation tool for satellite-derived hotspot data."""

import os
from typing import Any

import pandas as pd
from crewai.tools import tool


@tool("data_fetcher_tool")
def data_fetcher_tool(csv_path: str) -> str:
    """Load CSV data and perform quality assessment including missing values, outliers, and statistics.

    Args:
        csv_path: Path to the CSV file containing hotspot data.

    Returns:
        A formatted data quality report string.
    """
    try:
        if not os.path.exists(csv_path):
            return f"Error: File not found at {csv_path}"

        df = pd.read_csv(csv_path, parse_dates=["date"])

        # Basic info
        n_rows, n_cols = df.shape
        report_lines = [
            "=" * 60,
            "DATA QUALITY REPORT",
            "=" * 60,
            f"File: {csv_path}",
            f"Rows: {n_rows}",
            f"Columns: {n_cols}",
            f"Date range: {df['date'].min()} to {df['date'].max()}",
            f"Province(s): {', '.join(df['province'].unique())}",
            "",
        ]

        # Missing values
        missing = df.isnull().sum()
        report_lines.append("--- Missing Values ---")
        for col in df.columns:
            report_lines.append(f"  {col}: {missing[col]}")
        report_lines.append("")

        # Data types
        report_lines.append("--- Data Types ---")
        for col in df.columns:
            report_lines.append(f"  {col}: {df[col].dtype}")
        report_lines.append("")

        # Descriptive statistics for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        report_lines.append("--- Descriptive Statistics ---")
        stats = df[numeric_cols].describe().T
        stats["missing"] = df[numeric_cols].isnull().sum()
        report_lines.append(stats.to_string())
        report_lines.append("")

        # Outlier detection using IQR method
        report_lines.append("--- Outlier Detection (IQR Method) ---")
        total_outliers = 0
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            total_outliers += outliers
            if outliers > 0:
                report_lines.append(
                    f"  {col}: {outliers} outliers (range: {lower:.2f} - {upper:.2f})"
                )
        if total_outliers == 0:
            report_lines.append("  No outliers detected.")
        report_lines.append("")

        # Duplicates
        n_duplicates = df.duplicated().sum()
        report_lines.append(f"Duplicate rows: {n_duplicates}")
        report_lines.append("")

        # Data readiness score
        missing_pct = df.isnull().sum().sum() / (n_rows * n_cols) * 100
        readiness = max(0, 100 - missing_pct * 10 - total_outliers * 2 - n_duplicates * 5)
        readiness = min(100, readiness)
        report_lines.append(f"Data Readiness Score: {readiness:.0f}/100")
        report_lines.append("=" * 60)

        # Save report
        os.makedirs("output", exist_ok=True)
        report_text = "\n".join(report_lines)
        with open("output/data_quality_report.txt", "w") as f:
            f.write(report_text)

        return report_text

    except Exception as e:
        return f"Error during data validation: {str(e)}"
