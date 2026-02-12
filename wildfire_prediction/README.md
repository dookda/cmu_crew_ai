# Wildfire Hotspot Prediction Pipeline

A Multi-Agent AI Pipeline for predicting wildfire hotspots in Northern Thailand provinces using CrewAI framework.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Data Collector  │───▶│   Data Analyst   │───▶│   ML Engineer    │
│  (Validation)    │    │  (Features/EDA)  │    │  (LSTM/Baseline) │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                        │
                       ┌──────────────────┐    ┌────────▼─────────┐
                       │  Report Writer   │◀───│   Cartographer   │
                       │  (Research Paper) │    │  (Risk Maps)     │
                       └──────────────────┘    └──────────────────┘
```

**Pipeline:** Data Collector → Data Analyst → ML Engineer → Cartographer → Report Writer

## Tech Stack

- **Framework:** CrewAI with sequential pipeline
- **LLM:** Anthropic Claude (claude-sonnet-4-5-20250929)
- **ML:** scikit-learn (baselines), TensorFlow/Keras (LSTM)
- **Visualization:** matplotlib, seaborn, Folium
- **Geospatial:** GeoPandas, Folium
- **Package Manager:** uv

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Anthropic API key

## Installation

```bash
# Clone and enter project
cd wildfire_prediction

# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Generate sample data
uv run python data/generate_sample_data.py
```

## How to Run

```bash
uv run python -m wildfire_prediction.main
```

## Expected Outputs

After a successful pipeline run, the `output/` directory will contain:

| File | Description | Source |
|------|-------------|--------|
| `data_quality_report.txt` | Data validation and quality assessment | Task 1 |
| `processed_features.csv` | Engineered features dataset | Task 2 |
| `correlation_heatmap.png` | Feature correlation heatmap | Task 2 |
| `model_comparison.txt` | Model performance metrics | Task 3 |
| `prediction_plot.png` | Actual vs predicted comparison | Task 3 |
| `fire_risk_map.html` | Interactive Folium risk map | Task 4 |
| `monthly_heatmap.png` | Month × Year hotspot intensity | Task 4 |
| `hotspot_timeseries.png` | Hotspot count time series | Task 4 |
| `research_report.md` | Full research report | Task 5 |

## Project Structure

```
wildfire_prediction/
├── src/wildfire_prediction/
│   ├── config/
│   │   ├── agents.yaml          # 5 Agent definitions
│   │   └── tasks.yaml           # 5 Task definitions
│   ├── tools/
│   │   ├── data_fetcher.py      # Data loading & validation
│   │   ├── data_processor.py    # Feature engineering
│   │   ├── ml_trainer.py        # LSTM & baseline training
│   │   ├── map_generator.py     # Risk map generation
│   │   └── report_tool.py       # Report formatting
│   ├── crew.py                  # Crew assembly (@CrewBase)
│   └── main.py                  # Entry point
├── data/
│   ├── generate_sample_data.py  # Synthetic data generator
│   └── sample_hotspot_data.csv  # Generated dataset
├── output/                      # Pipeline outputs
├── tests/
│   └── test_crew.py
├── pyproject.toml
├── .env.example
└── README.md
```
