"""เครื่องมือดึงข้อมูลและตรวจสอบคุณภาพข้อมูลดาวเทียมจุดความร้อน"""

import os
from typing import Any

import pandas as pd
from crewai.tools import tool


@tool("data_fetcher_tool")
def data_fetcher_tool(csv_path: str) -> str:
    """โหลดไฟล์ CSV และตรวจสอบคุณภาพข้อมูล รวมถึง missing values, outliers, และสถิติพื้นฐาน

    Args:
        csv_path: พาธไปยังไฟล์ CSV ที่มีข้อมูลจุดความร้อน

    Returns:
        รายงานคุณภาพข้อมูลในรูปแบบข้อความ
    """
    try:
        if not os.path.exists(csv_path):
            return f"Error: ไม่พบไฟล์ที่ {csv_path}"

        df = pd.read_csv(csv_path, parse_dates=["date"])

        # ข้อมูลพื้นฐาน
        n_rows, n_cols = df.shape
        report_lines = [
            "=" * 60,
            "รายงานคุณภาพข้อมูล (DATA QUALITY REPORT)",
            "=" * 60,
            f"ไฟล์: {csv_path}",
            f"จำนวนแถว: {n_rows}",
            f"จำนวนคอลัมน์: {n_cols}",
            f"ช่วงวันที่: {df['date'].min()} ถึง {df['date'].max()}",
            f"จังหวัด: {', '.join(df['province'].unique())}",
            "",
        ]

        # ค่าที่หายไป (Missing Values)
        missing = df.isnull().sum()
        report_lines.append("--- ค่าที่หายไป (Missing Values) ---")
        for col in df.columns:
            report_lines.append(f"  {col}: {missing[col]}")
        report_lines.append("")

        # ชนิดข้อมูล
        report_lines.append("--- ชนิดข้อมูล (Data Types) ---")
        for col in df.columns:
            report_lines.append(f"  {col}: {df[col].dtype}")
        report_lines.append("")

        # สถิติเชิงพรรณนาสำหรับคอลัมน์ตัวเลข
        numeric_cols = df.select_dtypes(include=["number"]).columns
        report_lines.append("--- สถิติเชิงพรรณนา (Descriptive Statistics) ---")
        stats = df[numeric_cols].describe().T
        stats["missing"] = df[numeric_cols].isnull().sum()
        report_lines.append(stats.to_string())
        report_lines.append("")

        # ตรวจหาค่าผิดปกติด้วยวิธี IQR
        report_lines.append("--- ตรวจหาค่าผิดปกติ (Outlier Detection - IQR Method) ---")
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
                    f"  {col}: {outliers} ค่าผิดปกติ (ช่วง: {lower:.2f} - {upper:.2f})"
                )
        if total_outliers == 0:
            report_lines.append("  ไม่พบค่าผิดปกติ")
        report_lines.append("")

        # แถวซ้ำ
        n_duplicates = df.duplicated().sum()
        report_lines.append(f"แถวซ้ำ: {n_duplicates}")
        report_lines.append("")

        # คะแนนความพร้อมของข้อมูล
        missing_pct = df.isnull().sum().sum() / (n_rows * n_cols) * 100
        readiness = max(0, 100 - missing_pct * 10 - total_outliers * 2 - n_duplicates * 5)
        readiness = min(100, readiness)
        report_lines.append(f"คะแนนความพร้อมของข้อมูล: {readiness:.0f}/100")
        report_lines.append("=" * 60)

        # บันทึกรายงาน
        os.makedirs("output", exist_ok=True)
        report_text = "\n".join(report_lines)
        with open("output/data_quality_report.txt", "w") as f:
            f.write(report_text)

        return report_text

    except Exception as e:
        return f"เกิดข้อผิดพลาดระหว่างตรวจสอบข้อมูล: {str(e)}"
