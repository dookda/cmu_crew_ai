"""เครื่องมือสร้างแผนที่และ visualization สำหรับประเมินความเสี่ยงไฟป่า"""

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from crewai.tools import tool


# ข้อมูลจังหวัดภาคเหนือ พร้อมพิกัดและระดับความเสี่ยงจากข้อมูลย้อนหลัง
NORTHERN_PROVINCES = {
    "Chiang Mai": {"lat": 18.7883, "lon": 98.9853, "risk": "Critical", "avg_hotspots": 450},
    "Chiang Rai": {"lat": 19.9105, "lon": 99.8406, "risk": "High", "avg_hotspots": 320},
    "Lampang": {"lat": 18.2888, "lon": 99.4909, "risk": "High", "avg_hotspots": 280},
    "Lamphun": {"lat": 18.5744, "lon": 99.0087, "risk": "Medium", "avg_hotspots": 150},
    "Mae Hong Son": {"lat": 19.3020, "lon": 97.9654, "risk": "Critical", "avg_hotspots": 500},
    "Nan": {"lat": 18.7756, "lon": 100.7730, "risk": "High", "avg_hotspots": 350},
    "Phayao": {"lat": 19.1664, "lon": 99.9019, "risk": "Medium", "avg_hotspots": 180},
    "Phrae": {"lat": 18.1445, "lon": 100.1403, "risk": "Medium", "avg_hotspots": 200},
    "Uttaradit": {"lat": 17.6200, "lon": 100.0993, "risk": "Low", "avg_hotspots": 100},
}

# สีของ marker ตามระดับความเสี่ยง
RISK_ICON_COLORS = {
    "Critical": "red",      # วิกฤต
    "High": "orange",       # สูง
    "Medium": "beige",      # ปานกลาง
    "Low": "green",         # ต่ำ
}


@tool("map_generator_tool")
def map_generator_tool(processed_csv_path: str) -> str:
    """สร้างแผนที่ความเสี่ยงไฟป่าและ visualizations สำหรับจังหวัดภาคเหนือ

    Args:
        processed_csv_path: พาธไปยังไฟล์ CSV ที่ประมวลผลแล้ว

    Returns:
        สรุป visualizations ที่สร้างทั้งหมดพร้อมพาธไฟล์
    """
    try:
        os.makedirs("output", exist_ok=True)
        df = pd.read_csv(processed_csv_path, parse_dates=["date"])

        generated_files = []

        # --- 1. กราฟอนุกรมเวลา (Time series plot) ---
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df["date"], df["hotspot_count"], "b-", linewidth=1.2, label="Hotspot Count")
        ax.fill_between(df["date"], df["hotspot_count"], alpha=0.2, color="blue")
        ax.set_title("Wildfire Hotspot Time Series — Chiang Mai (2015–2024)", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Monthly Hotspot Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        ts_path = "output/hotspot_timeseries.png"
        fig.savefig(ts_path, dpi=150)
        plt.close(fig)
        generated_files.append(ts_path)

        # --- 2. Heatmap รายเดือน (เดือน x ปี) ---
        df["year"] = df["date"].dt.year
        df["month_num"] = df["date"].dt.month
        pivot = df.pivot_table(
            values="hotspot_count", index="month_num", columns="year", aggfunc="mean"
        )
        month_labels = [
            "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
            "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.",
        ]

        fig, ax = plt.subplots(figsize=(12, 6))
        import seaborn as sns
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            ax=ax,
            linewidths=0.5,
            xticklabels=True,
            yticklabels=month_labels,
        )
        ax.set_title("Monthly Hotspot Intensity Heatmap (Month × Year)", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Month")
        plt.tight_layout()
        heatmap_path = "output/monthly_heatmap.png"
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        generated_files.append(heatmap_path)

        # --- 3. แผนที่ Interactive ด้วย Folium ---
        import folium

        # สร้างแผนที่ศูนย์กลางภาคเหนือ
        m = folium.Map(location=[18.8, 99.0], zoom_start=7, tiles="OpenStreetMap")

        # วาง marker สำหรับแต่ละจังหวัด
        for province, info in NORTHERN_PROVINCES.items():
            risk = info["risk"]
            color = RISK_ICON_COLORS[risk]

            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4 style="margin: 0;">{province}</h4>
                <hr style="margin: 4px 0;">
                <b>ระดับความเสี่ยง:</b> {risk}<br>
                <b>จุดความร้อนเฉลี่ย/เดือน:</b> {info['avg_hotspots']}<br>
                <b>ละติจูด:</b> {info['lat']:.4f}<br>
                <b>ลองจิจูด:</b> {info['lon']:.4f}
            </div>
            """

            folium.Marker(
                location=[info["lat"], info["lon"]],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{province} — {risk}",
                icon=folium.Icon(color=color, icon="fire", prefix="fa"),
            ).add_to(m)

        # เพิ่มคำอธิบายสัญลักษณ์ (Legend)
        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                    background: white; padding: 12px; border-radius: 5px;
                    border: 2px solid grey; font-family: Arial;">
            <h4 style="margin: 0 0 8px 0;">ระดับความเสี่ยงไฟป่า</h4>
            <i style="background:red; width:12px; height:12px; display:inline-block;
               border-radius:50%;"></i> วิกฤต (Critical)<br>
            <i style="background:orange; width:12px; height:12px; display:inline-block;
               border-radius:50%;"></i> สูง (High)<br>
            <i style="background:#F0E68C; width:12px; height:12px; display:inline-block;
               border-radius:50%;"></i> ปานกลาง (Medium)<br>
            <i style="background:green; width:12px; height:12px; display:inline-block;
               border-radius:50%;"></i> ต่ำ (Low)
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        map_path = "output/fire_risk_map.html"
        m.save(map_path)
        generated_files.append(map_path)

        # --- สร้างรายงาน ---
        report_lines = [
            "=" * 60,
            "รายงาน Visualizations (VISUALIZATION REPORT)",
            "=" * 60,
            "",
            "ไฟล์ที่สร้าง:",
            f"  1. {ts_path} — กราฟอนุกรมเวลาจุดความร้อน (2015-2024)",
            f"  2. {heatmap_path} — Heatmap ความเข้มจุดความร้อนรายเดือน (เดือน x ปี)",
            f"  3. {map_path} — แผนที่ interactive ความเสี่ยงไฟป่า 9 จังหวัดภาคเหนือ",
            "",
            "รายละเอียดแผนที่:",
            "  - 9 จังหวัด พร้อม marker แสดงระดับความเสี่ยง",
            "  - ระดับความเสี่ยงแต่ละจังหวัด:",
        ]
        for province, info in NORTHERN_PROVINCES.items():
            report_lines.append(f"    {province}: {info['risk']} (เฉลี่ย {info['avg_hotspots']} จุด/เดือน)")
        report_lines.append("")
        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    except Exception as e:
        import traceback
        return f"เกิดข้อผิดพลาดระหว่างสร้างแผนที่: {str(e)}\n{traceback.format_exc()}"
