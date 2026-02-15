"""สคริปต์สร้างข้อมูลจำลอง (synthetic data) จุดความร้อนจังหวัดเชียงใหม่

สร้างข้อมูลรายเดือนจากดาวเทียมจำลอง ปี 2015-2024
จำลองรูปแบบตามฤดูกาลที่สมจริง สำหรับพยากรณ์ไฟป่าภาคเหนือของไทย
"""

import numpy as np
import pandas as pd

SEED = 42


def generate_sample_data(output_path: str = "data/sample_hotspot_data.csv") -> pd.DataFrame:
    """สร้างชุดข้อมูลจุดความร้อนจำลอง พร้อมรูปแบบตามฤดูกาลที่สมจริง

    Args:
        output_path: พาธสำหรับบันทึกไฟล์ CSV

    Returns:
        DataFrame ที่มีชุดข้อมูลที่สร้างขึ้น
    """
    np.random.seed(SEED)

    dates = pd.date_range(start="2015-01-01", end="2024-12-01", freq="MS")
    n = len(dates)

    months = dates.month
    years = dates.year

    # ตัวคูณความรุนแรงรายปี (บางปีไฟป่ารุนแรงกว่า)
    year_severity = {
        2015: 1.0, 2016: 1.3, 2017: 0.8, 2018: 1.1, 2019: 1.5,
        2020: 1.2, 2021: 0.9, 2022: 1.4, 2023: 1.6, 2024: 1.1,
    }
    severity = np.array([year_severity[y] for y in years])

    # --- จำนวนจุดความร้อน ---
    # พีคช่วง ก.พ.-เม.ย. (ฤดูเผาของไทย)
    hotspot_base = np.zeros(n)
    monthly_hotspot = {
        1: 80, 2: 350, 3: 600, 4: 400, 5: 50,
        6: 5, 7: 2, 8: 1, 9: 1, 10: 3, 11: 15, 12: 40,
    }
    for i, m in enumerate(months):
        hotspot_base[i] = monthly_hotspot[m] * severity[i]
    hotspot_count = np.clip(
        hotspot_base + np.random.normal(0, 30, n), 0, 800
    ).astype(int)

    # --- NDVI (แปรผกผันกับจุดความร้อน) ---
    ndvi_base = {
        1: 0.35, 2: 0.30, 3: 0.25, 4: 0.28, 5: 0.40,
        6: 0.55, 7: 0.65, 8: 0.72, 9: 0.75, 10: 0.68, 11: 0.55, 12: 0.42,
    }
    ndvi_mean = np.array([ndvi_base[m] for m in months]) + np.random.normal(0, 0.03, n)
    ndvi_mean = np.clip(ndvi_mean, 0.2, 0.8)

    # --- ปริมาณน้ำฝน (มม.) ---
    rainfall_base = {
        1: 10, 2: 5, 3: 15, 4: 50, 5: 150,
        6: 180, 7: 200, 8: 250, 9: 300, 10: 200, 11: 60, 12: 20,
    }
    rainfall_mm = np.array([rainfall_base[m] for m in months]) + np.random.normal(0, 20, n)
    rainfall_mm = np.clip(rainfall_mm, 0, 350)

    # --- อุณหภูมิ (องศาเซลเซียส) ---
    temp_base = {
        1: 22, 2: 25, 3: 29, 4: 32, 5: 30,
        6: 28, 7: 27, 8: 27, 9: 27, 10: 26, 11: 24, 12: 22,
    }
    temperature_c = np.array([temp_base[m] for m in months]) + np.random.normal(0, 1.5, n)
    temperature_c = np.clip(temperature_c, 20, 38)

    # --- ความชื้นดิน ---
    sm_base = {
        1: 0.12, 2: 0.08, 3: 0.07, 4: 0.10, 5: 0.20,
        6: 0.30, 7: 0.35, 8: 0.40, 9: 0.42, 10: 0.35, 11: 0.22, 12: 0.15,
    }
    soil_moisture = np.array([sm_base[m] for m in months]) + np.random.normal(0, 0.03, n)
    soil_moisture = np.clip(soil_moisture, 0.05, 0.45)

    # --- ความชื้นสัมพัทธ์ ---
    rh_base = {
        1: 45, 2: 38, 3: 35, 4: 42, 5: 60,
        6: 70, 7: 75, 8: 80, 9: 82, 10: 75, 11: 60, 12: 50,
    }
    relative_humidity = np.array([rh_base[m] for m in months]) + np.random.normal(0, 5, n)
    relative_humidity = np.clip(relative_humidity, 30, 85)

    # --- ความเร็วลม (กม./ชม.) ---
    ws_base = {
        1: 8, 2: 10, 3: 12, 4: 10, 5: 8,
        6: 7, 7: 6, 8: 5, 9: 5, 10: 6, 11: 7, 12: 8,
    }
    wind_speed_kmh = np.array([ws_base[m] for m in months]) + np.random.normal(0, 2, n)
    wind_speed_kmh = np.clip(wind_speed_kmh, 3, 25)

    # --- อุณหภูมิพื้นผิว (กลางวัน, องศาเซลเซียส) ---
    lst_base = {
        1: 28, 2: 32, 3: 38, 4: 40, 5: 36,
        6: 33, 7: 31, 8: 30, 9: 30, 10: 31, 11: 29, 12: 27,
    }
    lst_day_c = np.array([lst_base[m] for m in months]) + np.random.normal(0, 2, n)
    lst_day_c = np.clip(lst_day_c, 25, 45)

    df = pd.DataFrame({
        "date": dates,
        "province": "Chiang Mai",
        "hotspot_count": hotspot_count,
        "ndvi_mean": np.round(ndvi_mean, 4),
        "rainfall_mm": np.round(rainfall_mm, 1),
        "temperature_c": np.round(temperature_c, 1),
        "soil_moisture": np.round(soil_moisture, 4),
        "relative_humidity": np.round(relative_humidity, 1),
        "wind_speed_kmh": np.round(wind_speed_kmh, 1),
        "lst_day_c": np.round(lst_day_c, 1),
    })

    df.to_csv(output_path, index=False)
    print(f"สร้างข้อมูลจำลอง {len(df)} แถว -> {output_path}")
    return df


if __name__ == "__main__":
    generate_sample_data()
