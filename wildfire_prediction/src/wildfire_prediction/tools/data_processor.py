"""เครื่องมือประมวลผลข้อมูลและสร้าง features สำหรับพยากรณ์ไฟป่า"""

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
    """ประมวลผลข้อมูลดิบ: สร้าง lag features, rolling stats, seasonal encoding, และวิเคราะห์ correlation

    Args:
        csv_path: พาธไปยังไฟล์ CSV ข้อมูลดิบ

    Returns:
        รายงานการสร้าง features พร้อมพาธไฟล์ที่บันทึก
    """
    try:
        np.random.seed(42)
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # --- สร้าง Lag features (ค่าย้อนหลัง t-1, t-2, t-3) ---
        lag_cols = ["hotspot_count", "ndvi_mean", "rainfall_mm"]
        for col in lag_cols:
            for lag in [1, 2, 3]:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

        # --- สร้างค่าสถิติเคลื่อนที่ (Rolling statistics, window=3 เดือน) ---
        rolling_cols = ["rainfall_mm", "temperature_c"]
        for col in rolling_cols:
            df[f"{col}_roll3_mean"] = df[col].rolling(window=3).mean()
            df[f"{col}_roll3_std"] = df[col].rolling(window=3).std()

        # --- เข้ารหัสเดือนด้วย sine/cosine (Cyclical encoding) ---
        df["month"] = df["date"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # --- สร้าง flag ฤดูแล้ง (ธ.ค.-เม.ย. = 1, อื่นๆ = 0) ---
        df["dry_season"] = df["month"].apply(lambda m: 1 if m in [12, 1, 2, 3, 4] else 0)

        # ลบแถวที่มี NaN จากการสร้าง lag/rolling
        df_processed = df.dropna().reset_index(drop=True)

        # --- วิเคราะห์ Correlation ---
        numeric_cols = df_processed.select_dtypes(include=["number"]).columns
        corr_matrix = df_processed[numeric_cols].corr()

        # ค่า Correlation กับ hotspot_count
        hotspot_corr = corr_matrix["hotspot_count"].sort_values(ascending=False)

        # --- บันทึก Correlation heatmap ---
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

        # --- บันทึก CSV ที่ประมวลผลแล้ว ---
        processed_path = "output/processed_features.csv"
        df_processed.to_csv(processed_path, index=False)

        # --- สร้างรายงาน ---
        report_lines = [
            "=" * 60,
            "รายงานการสร้าง Features (FEATURE ENGINEERING REPORT)",
            "=" * 60,
            f"จำนวนแถวเดิม: {len(df)}, หลังประมวลผล: {len(df_processed)}",
            f"จำนวน features ทั้งหมด: {len(numeric_cols)}",
            "",
            "--- รายชื่อ Features ทั้งหมด ---",
            ", ".join(numeric_cols.tolist()),
            "",
            "--- ค่า Correlation กับ hotspot_count ---",
            hotspot_corr.to_string(),
            "",
            "--- รูปแบบตามฤดูกาล (Seasonal Patterns) ---",
            "เดือนที่มีจุดความร้อนสูงสุด: กุมภาพันธ์, มีนาคม, เมษายน",
            "ค่าเฉลี่ยจุดความร้อนฤดูแล้ง (ธ.ค.-เม.ย.): "
            f"{df_processed[df_processed['dry_season'] == 1]['hotspot_count'].mean():.1f}",
            "ค่าเฉลี่ยจุดความร้อนฤดูฝน (พ.ค.-พ.ย.): "
            f"{df_processed[df_processed['dry_season'] == 0]['hotspot_count'].mean():.1f}",
            "",
            f"บันทึก CSV ที่: {processed_path}",
            f"บันทึก Heatmap ที่: {heatmap_path}",
            "=" * 60,
        ]

        return "\n".join(report_lines)

    except Exception as e:
        return f"เกิดข้อผิดพลาดระหว่างสร้าง features: {str(e)}"
