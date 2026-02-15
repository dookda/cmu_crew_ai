"""เครื่องมือฝึกโมเดล Machine Learning สำหรับพยากรณ์จุดความร้อน"""

import os
import warnings
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from crewai.tools import tool

warnings.filterwarnings("ignore")


def _nash_sutcliffe(observed: np.ndarray, predicted: np.ndarray) -> float:
    """คำนวณค่า Nash-Sutcliffe Efficiency (NSE)

    Args:
        observed: อาร์เรย์ค่าจริง
        predicted: อาร์เรย์ค่าพยากรณ์

    Returns:
        ค่า NSE อยู่ระหว่าง -inf ถึง 1.0 (1.0 = สมบูรณ์แบบ)
    """
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    if denominator == 0:
        return 0.0
    return 1.0 - numerator / denominator


@tool("ml_trainer_tool")
def ml_trainer_tool(processed_csv_path: str) -> str:
    """ฝึกโมเดล LSTM และ baseline models บนข้อมูลจุดความร้อน แล้วประเมินผล

    Args:
        processed_csv_path: พาธไปยังไฟล์ CSV ที่ประมวลผลแล้ว

    Returns:
        รายงานเปรียบเทียบโมเดล พร้อม metrics และพาธไฟล์กราฟ
    """
    try:
        np.random.seed(42)

        df = pd.read_csv(processed_csv_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # เลือกคอลัมน์ features (ไม่รวมค่าที่ไม่ใช่ตัวเลขและ target)
        exclude_cols = ["date", "province", "hotspot_count", "month"]
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and df[c].dtype in ["float64", "int64", "float32", "int32"]
        ]
        target_col = "hotspot_count"

        X = df[feature_cols].values
        y = df[target_col].values

        # แบ่งข้อมูลตามเวลา (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_test = df["date"].iloc[split_idx:].values

        # ปรับสเกลข้อมูลด้วย MinMaxScaler
        from sklearn.preprocessing import MinMaxScaler
        scaler_X = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        # --- Baseline 1: Linear Regression ---
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)
        y_pred_lr = np.clip(y_pred_lr, 0, None)

        # --- Baseline 2: Random Forest ---
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        y_pred_rf = rf.predict(X_test_scaled)
        y_pred_rf = np.clip(y_pred_rf, 0, None)

        # --- LSTM ---
        lookback = 6  # ใช้ข้อมูลย้อนหลัง 6 เดือน

        def create_sequences(X_data: np.ndarray, y_data: np.ndarray, lookback: int):
            """สร้าง sequences สำหรับ LSTM input"""
            Xs, ys = [], []
            for i in range(lookback, len(X_data)):
                Xs.append(X_data[i - lookback:i])
                ys.append(y_data[i])
            return np.array(Xs), np.array(ys)

        # สร้าง sequences จากข้อมูลทั้งหมดที่ปรับสเกลแล้ว
        X_full_scaled = scaler_X.transform(X)
        y_full_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()

        X_seq, y_seq = create_sequences(X_full_scaled, y_full_scaled, lookback)

        # แบ่ง sequences ตามเวลาเดียวกัน (ปรับตาม lookback)
        seq_split_idx = split_idx - lookback
        X_seq_train, X_seq_test = X_seq[:seq_split_idx], X_seq[seq_split_idx:]
        y_seq_train, y_seq_test = y_seq[:seq_split_idx], y_seq[seq_split_idx:]

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.random.set_seed(42)
        from tensorflow import keras

        # สร้างโมเดล LSTM: LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(16) → Dense(1)
        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True, input_shape=(lookback, len(feature_cols))),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            X_seq_train, y_seq_train,
            epochs=50, batch_size=8, verbose=0,
            validation_split=0.1,
        )

        # พยากรณ์และแปลงกลับเป็นค่าจริง
        y_pred_lstm_scaled = model.predict(X_seq_test, verbose=0).flatten()
        y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()
        y_pred_lstm = np.clip(y_pred_lstm, 0, None)

        # จัดให้ค่าจริงตรงกับค่าพยากรณ์ LSTM (ปรับตาม lookback)
        y_test_lstm = y[split_idx:]
        min_len = min(len(y_pred_lstm), len(y_test_lstm))
        y_pred_lstm = y_pred_lstm[:min_len]
        y_test_lstm = y_test_lstm[:min_len]

        # --- ประเมินผลทุกโมเดล ---
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
            """ประเมินโมเดลด้วย RMSE, MAE, R², NSE"""
            return {
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "MAE": mean_absolute_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred),
                "NSE": _nash_sutcliffe(y_true, y_pred),
            }

        metrics_lr = evaluate(y_test, y_pred_lr)
        metrics_rf = evaluate(y_test, y_pred_rf)
        metrics_lstm = evaluate(y_test_lstm, y_pred_lstm)

        # --- ตารางเปรียบเทียบ ---
        comparison = pd.DataFrame({
            "Linear Regression": metrics_lr,
            "Random Forest": metrics_rf,
            "LSTM": metrics_lstm,
        }).T

        # --- สร้างกราฟ: ค่าจริง vs ค่าพยากรณ์ ---
        os.makedirs("output", exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

        axes[0].plot(dates_test, y_test, "b-", label="Actual", linewidth=1.5)
        axes[0].plot(dates_test, y_pred_lr, "r--", label="Predicted (LR)", linewidth=1.5)
        axes[0].set_title("Linear Regression: Actual vs Predicted")
        axes[0].set_ylabel("Hotspot Count")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(dates_test, y_test, "b-", label="Actual", linewidth=1.5)
        axes[1].plot(dates_test, y_pred_rf, "g--", label="Predicted (RF)", linewidth=1.5)
        axes[1].set_title("Random Forest: Actual vs Predicted")
        axes[1].set_ylabel("Hotspot Count")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        dates_lstm = dates_test[:min_len]
        axes[2].plot(dates_lstm, y_test_lstm, "b-", label="Actual", linewidth=1.5)
        axes[2].plot(dates_lstm, y_pred_lstm, "m--", label="Predicted (LSTM)", linewidth=1.5)
        axes[2].set_title("LSTM: Actual vs Predicted")
        axes[2].set_ylabel("Hotspot Count")
        axes[2].set_xlabel("Date")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.suptitle("Model Comparison: Wildfire Hotspot Prediction", fontsize=14, y=1.01)
        plt.tight_layout()
        plot_path = "output/prediction_plot.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # บันทึกโมเดล LSTM
        model.save("output/lstm_model.keras")

        # บันทึกตารางเปรียบเทียบ
        comparison_path = "output/model_comparison.txt"
        with open(comparison_path, "w") as f:
            f.write("รายงานเปรียบเทียบโมเดล (MODEL COMPARISON REPORT)\n")
            f.write("=" * 60 + "\n\n")
            f.write(comparison.to_string())
            f.write("\n\n")
            best_model = comparison["R2"].idxmax()
            f.write(f"โมเดลที่ดีที่สุดตาม R²: {best_model}\n")
            f.write(f"ค่า R² สูงสุด: {comparison.loc[best_model, 'R2']:.4f}\n")

        # --- สร้างรายงาน ---
        report_lines = [
            "=" * 60,
            "รายงานประเมินโมเดล (MODEL EVALUATION REPORT)",
            "=" * 60,
            "",
            "--- ตารางเปรียบเทียบโมเดล ---",
            comparison.to_string(),
            "",
            f"โมเดลที่ดีที่สุดตาม R²: {best_model} (R² = {comparison.loc[best_model, 'R2']:.4f})",
            "",
            f"สถาปัตยกรรม LSTM: Input → LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(16) → Dense(1)",
            f"Lookback: {lookback} เดือน",
            f"ข้อมูล train: {len(X_train)} ตัวอย่าง, test: {len(X_test)} ตัวอย่าง",
            f"LSTM sequences — Train: {len(X_seq_train)}, Test: {len(X_seq_test)}",
            "",
            f"บันทึกกราฟพยากรณ์ที่: {plot_path}",
            f"บันทึกตารางเปรียบเทียบที่: {comparison_path}",
            f"บันทึกโมเดล LSTM ที่: output/lstm_model.keras",
            "=" * 60,
        ]

        return "\n".join(report_lines)

    except Exception as e:
        import traceback
        return f"เกิดข้อผิดพลาดระหว่างฝึกโมเดล: {str(e)}\n{traceback.format_exc()}"
