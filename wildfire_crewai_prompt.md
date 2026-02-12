# PROMPT: Wildfire Hotspot Prediction Pipeline — CrewAI Multi-Agent System

## Project Overview

สร้างระบบ Multi-Agent AI Pipeline สำหรับพยากรณ์จุดความร้อน (wildfire hotspot) ในจังหวัดภาคเหนือของประเทศไทย โดยใช้ CrewAI framework ระบบประกอบด้วย Agent หลายตัวทำงานร่วมกันแบบ sequential pipeline:

```
Data Collector → Data Analyst → ML Engineer → Cartographer → Report Writer
```

## Tech Stack

- Python 3.12
- CrewAI (latest version)
- Anthropic Claude as LLM (claude-sonnet-4-5-20250929)
- pandas, numpy สำหรับ data processing
- scikit-learn สำหรับ baseline models
- tensorflow/keras สำหรับ LSTM
- matplotlib, seaborn สำหรับ visualization
- geopandas, folium สำหรับ geospatial mapping
- ใช้ `uv` เป็น package manager (CrewAI standard)

## Project Structure

สร้างโปรเจคด้วย `crewai create crew wildfire_prediction` แล้วปรับโครงสร้างให้เป็น:

```
wildfire_prediction/
├── src/wildfire_prediction/
│   ├── config/
│   │   ├── agents.yaml          # Agent definitions
│   │   └── tasks.yaml           # Task definitions
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── data_fetcher.py      # Satellite data fetching tool
│   │   ├── data_processor.py    # Data preprocessing & feature engineering tool
│   │   ├── ml_trainer.py        # ML model training tool
│   │   ├── map_generator.py     # Risk map generation tool
│   │   └── report_tool.py       # Report formatting tool
│   ├── crew.py                  # Crew assembly with @CrewBase
│   └── main.py                  # Entry point
├── data/                        # Sample/mock data
│   └── sample_hotspot_data.csv  # Generated sample dataset
├── output/                      # Pipeline outputs
├── tests/
│   └── test_crew.py
├── pyproject.toml
├── .env.example                 # API key template
└── README.md
```

## Detailed Specifications

### 1. Sample Data Generation (`data/sample_hotspot_data.csv`)

เนื่องจากไม่สามารถเข้าถึง Google Earth Engine API ได้โดยตรง ให้สร้าง **realistic synthetic dataset** ที่จำลองข้อมูลจริงสำหรับจังหวัดเชียงใหม่ปี 2015-2024 (120 months):

| Column | Description | Range |
|--------|-------------|-------|
| date | วันที่ (monthly, YYYY-MM-01) | 2015-01 ถึง 2024-12 |
| province | ชื่อจังหวัด | Chiang Mai |
| hotspot_count | จำนวน hotspot รายเดือน | 0-800 (สูงสุดช่วง Feb-Apr) |
| ndvi_mean | ค่าเฉลี่ย NDVI | 0.2-0.8 (ต่ำช่วง dry season) |
| rainfall_mm | ปริมาณน้ำฝนรายเดือน | 0-350 mm |
| temperature_c | อุณหภูมิเฉลี่ย | 20-38°C |
| soil_moisture | ความชื้นดิน | 0.05-0.45 |
| relative_humidity | ความชื้นสัมพัทธ์ | 30-85% |
| wind_speed_kmh | ความเร็วลม | 3-25 km/h |
| lst_day_c | Land Surface Temperature (day) | 25-45°C |

**Important patterns to simulate:**
- Hotspot season: peak ช่วง February-April (Thai burning season) มีค่าสูง 200-800
- NDVI inversely correlated with hotspot — ต่ำช่วง dry season (Dec-Apr)
- Rainfall near zero ช่วง Dec-Feb, สูงสุดช่วง Aug-Sep
- Temperature สูงช่วง Mar-May
- เพิ่ม random noise เพื่อความสมจริง
- เพิ่ม year-to-year variation (บางปีไฟป่ารุนแรงกว่า)

สร้าง Python script `data/generate_sample_data.py` ที่สามารถรันแยกต่างหากเพื่อสร้างข้อมูลจำลองนี้

### 2. Agents Configuration (`config/agents.yaml`)

กำหนด 5 Agents:

#### Agent 1: data_collector
- role: "Remote Sensing Data Specialist"
- goal: รวบรวมและตรวจสอบคุณภาพข้อมูลดาวเทียม
- backstory: ผู้เชี่ยวชาญ Google Earth Engine 10 ปี เชี่ยวชาญ MODIS, Sentinel-2, CHIRPS สำหรับ Southeast Asia
- tools: [data_fetcher_tool]

#### Agent 2: data_analyst
- role: "Geospatial Data Analyst"
- goal: วิเคราะห์ spatio-temporal patterns และสร้าง features สำหรับ ML
- backstory: นักวิเคราะห์ข้อมูลเชิงพื้นที่ เชี่ยวชาญ time series analysis, feature engineering, EDA สำหรับ environmental data
- tools: [data_processor_tool]

#### Agent 3: ml_engineer
- role: "Deep Learning Engineer for Environmental Prediction"
- goal: สร้าง LSTM model ที่มีความแม่นยำสูง (R² > 0.8) พร้อมเปรียบเทียบกับ baseline
- backstory: วิศวกร ML เชี่ยวชาญ LSTM, GRU สำหรับ time series forecasting ด้าน environmental science
- tools: [ml_trainer_tool]

#### Agent 4: cartographer
- role: "GIS Cartographer and Spatial Visualization Expert"
- goal: สร้าง interactive fire risk map และ visualizations
- backstory: ผู้เชี่ยวชาญด้านการทำแผนที่ดิจิทัล เชี่ยวชาญ Folium, GeoPandas, Leaflet สำหรับ web mapping
- tools: [map_generator_tool]

#### Agent 5: report_writer
- role: "Academic Research Writer — Geoinformatics"
- goal: เขียนรายงานวิจัยฉบับสมบูรณ์ภาษาอังกฤษ พร้อม figures และ tables
- backstory: นักเขียนงานวิจัยที่ตีพิมพ์ในวารสาร Remote Sensing, IJRS สามารถเขียนได้ทั้งภาษาไทยและอังกฤษ
- tools: [report_formatting_tool]

**ทุก Agent ใช้ LLM**: `anthropic/claude-sonnet-4-5-20250929`
**ทุก Agent**: `verbose: true`, `memory: true`

### 3. Tasks Configuration (`config/tasks.yaml`)

#### Task 1: collect_and_validate_data
- agent: data_collector
- description: โหลดข้อมูลจาก data/sample_hotspot_data.csv ตรวจสอบ missing values, outliers, data types ทำ summary statistics และ data quality report
- expected_output: "Data quality report ที่ระบุ: จำนวนแถว, missing values ต่อ column, basic statistics (mean, std, min, max), outlier detection results, และ data readiness score (0-100)"

#### Task 2: analyze_and_engineer_features
- agent: data_analyst
- description: 
  1. วิเคราะห์ correlation matrix ระหว่างทุก features กับ hotspot_count
  2. วิเคราะห์ seasonal decomposition ของ hotspot time series
  3. สร้าง lag features: hotspot_count, ndvi, rainfall สำหรับ t-1, t-2, t-3
  4. สร้าง rolling statistics: mean และ std ของ rainfall, temperature (window=3 months)
  5. สร้าง month indicator (sine/cosine encoding)
  6. สร้าง dry_season flag (1 ถ้า Dec-Apr, 0 ถ้าอื่น)
  7. บันทึก processed dataset เป็น CSV
  8. สร้าง correlation heatmap เป็น PNG
- expected_output: "Feature engineering report พร้อมรายชื่อ features ทั้งหมด, correlation analysis, seasonal patterns ที่ค้นพบ, และ path ไปยัง processed CSV file และ heatmap image"
- context: [collect_and_validate_data]

#### Task 3: train_and_evaluate_model
- agent: ml_engineer
- description:
  1. แบ่งข้อมูล train/test โดย 80% แรก (ตามเวลา) เป็น train, 20% หลังเป็น test
  2. Normalize features ด้วย MinMaxScaler
  3. สร้าง sequence data สำหรับ LSTM (lookback = 6 months)
  4. Train LSTM: Input → LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(16) → Dense(1)
  5. Train baseline models: Linear Regression, Random Forest
  6. ประเมินทุก model ด้วย: RMSE, MAE, R², NSE (Nash-Sutcliffe Efficiency)
  7. สร้าง comparison table และ actual vs predicted plot เป็น PNG
  8. บันทึก trained model
- expected_output: "Model evaluation report มี comparison table ของทั้ง 3 models, แสดง metric ทั้ง 4 ตัว, path ไปยัง prediction plot, และ recommendation ว่า model ไหนดีที่สุดพร้อมเหตุผล"
- context: [analyze_and_engineer_features]

#### Task 4: generate_risk_maps
- agent: cartographer
- description:
  1. สร้าง static time series plot: hotspot count vs predicted (matplotlib)
  2. สร้าง monthly heatmap: hotspot intensity by month x year
  3. สร้าง interactive HTML map ด้วย Folium แสดงพื้นที่เสี่ยงจังหวัดภาคเหนือ 9 จังหวัด (Chiang Mai, Chiang Rai, Lampang, Lamphun, Mae Hong Son, Nan, Phayao, Phrae, Uttaradit)
     - ใส่ marker แสดงระดับความเสี่ยง (color-coded: green/yellow/orange/red)
     - เพิ่ม popup แสดงข้อมูลสรุปของแต่ละจังหวัด
  4. บันทึกทุก output ไปยัง output/ directory
- expected_output: "สรุป visualizations ที่สร้างทั้งหมด พร้อม file paths, คำอธิบายแต่ละ map/chart ว่าแสดงอะไร"
- context: [train_and_evaluate_model]

#### Task 5: write_research_report
- agent: report_writer
- description: เขียนรายงานวิจัยฉบับสมบูรณ์ภาษาอังกฤษ ในรูปแบบ Markdown บันทึกเป็น output/research_report.md ประกอบด้วย:
  1. **Title**: "LSTM-based Wildfire Hotspot Prediction in Northern Thailand: A Multi-Source Remote Sensing Approach"
  2. **Abstract** (250 words): สรุปวัตถุประสงค์ วิธีการ ผลลัพธ์หลัก
  3. **1. Introduction**: ปัญหาไฟป่าภาคเหนือไทย, ผลกระทบ PM2.5, literature review (อ้างอิงงานวิจัยที่เกี่ยวข้อง)
  4. **2. Study Area**: จังหวัดเชียงใหม่ — ที่ตั้ง สภาพภูมิประเทศ ภูมิอากาศ
  5. **3. Data and Methods**: 
     - 3.1 Data Sources (ตารางสรุป dataset)
     - 3.2 Feature Engineering
     - 3.3 LSTM Architecture (อธิบาย architecture พร้อม diagram)
     - 3.4 Baseline Models
     - 3.5 Evaluation Metrics
  6. **4. Results**:
     - 4.1 Exploratory Analysis (อ้างอิง correlation heatmap)
     - 4.2 Model Comparison (ตาราง metrics)
     - 4.3 Prediction Performance (อ้างอิง prediction plot)
     - 4.4 Fire Risk Assessment (อ้างอิง risk map)
  7. **5. Discussion**: ข้อค้นพบสำคัญ, เปรียบเทียบกับงานวิจัยอื่น, limitations
  8. **6. Conclusion**: สรุปและข้อเสนอแนะสำหรับงานวิจัยในอนาคต
  9. **References**: อ้างอิงในรูปแบบ APA
- expected_output: "Research report ฉบับสมบูรณ์ในรูปแบบ Markdown ที่อ้างอิง figures/tables จาก tasks ก่อนหน้า บันทึกที่ output/research_report.md"
- context: [collect_and_validate_data, analyze_and_engineer_features, train_and_evaluate_model, generate_risk_maps]

### 4. Custom Tools Implementation

#### tools/data_fetcher.py
สร้าง CrewAI tool ที่:
- อ่าน CSV file จาก path ที่กำหนด
- ตรวจสอบ schema (column names, data types)
- รายงาน missing values, duplicates
- คำนวณ basic statistics
- Return summary string

#### tools/data_processor.py
สร้าง CrewAI tool ที่:
- รับ CSV path → อ่านด้วย pandas
- สร้าง lag features, rolling statistics, seasonal encoding
- คำนวณ correlation matrix
- สร้าง heatmap ด้วย seaborn → บันทึก PNG
- บันทึก processed CSV
- Return feature list และ file paths

#### tools/ml_trainer.py
สร้าง CrewAI tool ที่:
- รับ processed CSV path
- Train/test split (temporal, 80/20)
- Train LSTM (Keras), Linear Regression, Random Forest
- ประเมินด้วย RMSE, MAE, R², NSE
- สร้าง actual vs predicted plot → บันทึก PNG
- Return comparison metrics as formatted string

#### tools/map_generator.py
สร้าง CrewAI tool ที่:
- สร้าง Folium map centered on Northern Thailand (lat=18.8, lon=99.0)
- วาง markers สำหรับ 9 จังหวัดภาคเหนือ พร้อม risk level (mock based on historical averages)
- Color-code: Critical (red), High (orange), Medium (yellow), Low (green)
- เพิ่ม popup ข้อมูลสรุปแต่ละจังหวัด
- สร้าง monthly heatmap plot (matplotlib)
- บันทึก HTML map + PNG plots ไปยัง output/
- Return file paths

#### tools/report_tool.py
สร้าง CrewAI tool ที่:
- รับ report content (Markdown string)
- ตรวจสอบว่ามีทุก section ที่กำหนด
- Format references ให้เป็น APA style
- บันทึกเป็น output/research_report.md
- Return confirmation + word count

### 5. Crew Assembly (`crew.py`)

ใช้ `@CrewBase` decorator pattern:
- Process: `sequential`
- Memory: `True`
- Planning: `True`
- Verbose: `True`

### 6. Entry Point (`main.py`)

```python
inputs = {
    "province": "Chiang Mai",
    "start_year": "2015",
    "end_year": "2024",
    "data_path": "data/sample_hotspot_data.csv"
}
```

รัน crew ด้วย `kickoff(inputs=inputs)` พร้อม timing measurement

### 7. Environment Configuration

#### .env.example
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

#### pyproject.toml
เพิ่ม dependencies: crewai, crewai[tools], pandas, numpy, scikit-learn, tensorflow, matplotlib, seaborn, geopandas, folium

### 8. README.md

เขียน README ภาษาอังกฤษ อธิบาย:
- Project description
- Architecture diagram (ASCII art)
- Prerequisites
- Installation steps
- How to run
- Expected outputs
- Project structure

## Important Notes

1. **ทุก tool ต้องทำงานได้จริง** — ไม่ใช่แค่ mock/placeholder ยกเว้น data fetching จาก GEE ที่ใช้ local CSV แทน
2. **Error handling** — ทุก tool ต้องมี try/except และ return meaningful error messages
3. **File I/O** — ทุก output บันทึกไปยัง `output/` directory, สร้าง directory ถ้ายังไม่มี
4. **Type hints** — ใช้ Python type hints ทุกที่
5. **Docstrings** — ทุก function มี docstring อธิบายการทำงาน
6. **ภาษา** — Code comments เป็นภาษาอังกฤษ, Agent backstory เป็นภาษาอังกฤษ, Report เป็นภาษาอังกฤษ
7. **Reproducibility** — ตั้ง random seed ทุกที่ที่มี randomness

## Expected Final Outputs

เมื่อรัน pipeline สำเร็จ จะได้ไฟล์ใน `output/`:

```
output/
├── data_quality_report.txt          # จาก Task 1
├── processed_features.csv           # จาก Task 2
├── correlation_heatmap.png          # จาก Task 2
├── model_comparison.txt             # จาก Task 3
├── prediction_plot.png              # จาก Task 3
├── fire_risk_map.html               # จาก Task 4 (interactive Folium map)
├── monthly_heatmap.png              # จาก Task 4
├── hotspot_timeseries.png           # จาก Task 4
└── research_report.md               # จาก Task 5
```
