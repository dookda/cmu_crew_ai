"""จุดเริ่มต้นสำหรับรัน Wildfire Prediction Pipeline"""

import time

from dotenv import load_dotenv
load_dotenv()  # โหลด API key จากไฟล์ .env

from wildfire_prediction.crew import WildfirePredictionCrew


def run() -> None:
    """รัน pipeline พยากรณ์ไฟป่า พร้อมจับเวลา"""

    # กำหนด inputs สำหรับ pipeline
    inputs = {
        "province": "Chiang Mai",
        "start_year": "2015",
        "end_year": "2024",
        "data_path": "data/sample_hotspot_data.csv",
    }

    print("=" * 60)
    print("ระบบพยากรณ์จุดความร้อน (WILDFIRE HOTSPOT PREDICTION)")
    print("=" * 60)
    print(f"จังหวัด: {inputs['province']}")
    print(f"ช่วงเวลา: {inputs['start_year']} - {inputs['end_year']}")
    print(f"ข้อมูล: {inputs['data_path']}")
    print("=" * 60)

    start_time = time.time()

    # สร้างทีม Crew แล้วเริ่มรัน pipeline
    crew = WildfirePredictionCrew()
    result = crew.crew().kickoff(inputs=inputs)

    # คำนวณเวลาที่ใช้
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    print("\n" + "=" * 60)
    print("รัน PIPELINE เสร็จสมบูรณ์")
    print(f"เวลาที่ใช้ทั้งหมด: {minutes} นาที {seconds:.1f} วินาที")
    print("=" * 60)
    print("\nผลลัพธ์สุดท้าย:")
    print(result)


if __name__ == "__main__":
    run()
