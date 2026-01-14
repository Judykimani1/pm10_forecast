import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PM10 Forecast App (Gradient Boosting)
# ============================================================

DATA_DIR = "data"
MODEL_DIR = "models"

DEFAULT_DATA_FILE = "pm10_model_dataset_t1_t24_final.csv"
DEFAULT_MODEL_1H = "gb_pm10_t_plus_1h.pkl"
DEFAULT_MODEL_24H = "gb_pm10_t_plus_24h.pkl"

st.set_page_config(page_title="PM10 Forecast (GB)", layout="wide")
st.title("PM10 Forecasting Interface (Gradient Boosting)")
st.caption("Forecast PM10 for t+1h and t+24h. Includes latest forecast and historical demo mode.")

def require_file(path: str, label: str):
    if not os.path.exists(path):
        st.error(f"Missing {label}: `{path}`")
        st.stop()

@st.cache_resource
def load_models(m1_path: str, m24_path: str):
    m1 = joblib.load(m1_path)
    m24 = joblib.load(m24_path)
    return m1, m24

@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    return df

# -----------------------------
# Paths
# -----------------------------
data_path = os.path.join(DATA_DIR, DEFAULT_DATA_FILE)
m1_path = os.path.join(MODEL_DIR, DEFAULT_MODEL_1H)
m24_path = os.path.join(MODEL_DIR, DEFAULT_MODEL_24H)

require_file(data_path, "dataset CSV")
require_file(m1_path, "t+1h model file")
require_file(m24_path, "t+24h model file")

gb_1h, gb_24h = load_models(m1_path, m24_path)
df = load_data(data_path)

# -----------------------------
# Feature columns (must match training)
# -----------------------------
if not hasattr(gb_1h, "feature_names_in_"):
    st.error("Model missing `feature_names_in_`. Please retrain using pandas DataFrame.")
    st.stop()

FEATURE_COLS = list(gb_1h.feature_names_in_)

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    st.error("Dataset is missing required feature columns:")
    st.write(missing)
    st.stop()

if "station" not in df.columns or "datetime" not in df.columns:
    st.error("Dataset must contain `station` and `datetime` columns.")
    st.stop()

df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

# -----------------------------
# Sidebar UI
# -----------------------------
st.sidebar.header("Controls")

stations = sorted(df["station"].astype(str).unique().tolist())
station = st.sidebar.selectbox("Station", stations, index=0)

mode = st.sidebar.radio(
    "Mode",
    ["Latest forecast", "Historical demo (compare with actual)"],
    index=0
)

scenario = st.sidebar.checkbox("Scenario mode (adjust key inputs)", value=False)

# Filter station
df_s = df[df["station"].astype(str) == str(station)].copy().sort_values("datetime")

# Pick row
if mode == "Latest forecast":
    row = df_s.iloc[-1].copy()
else:
    N = 2000 if len(df_s) > 2000 else len(df_s)
    df_recent = df_s.tail(N).copy()
    ts_list = df_recent["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    picked = st.sidebar.selectbox("Pick a timestamp (recent window)", ts_list, index=len(ts_list) - 1)
    picked_dt = pd.to_datetime(picked)
    row = df_s[df_s["datetime"] == picked_dt].iloc[0].copy()

x_row = row[FEATURE_COLS].to_frame().T

# Scenario sliders
if scenario:
    st.info("Scenario mode: adjust a few inputs and observe prediction changes.")
    c1, c2, c3 = st.columns(3)

    def slider_if_exists(colname, label, ui_col, minv, maxv):
        if colname in x_row.columns:
            cur = float(x_row[colname].iloc[0])
            x_row[colname] = ui_col.slider(label, min_value=minv, max_value=maxv, value=cur)

    slider_if_exists("temp", "Temperature", c1, -20.0, 45.0)
    slider_if_exists("humidity", "Humidity", c1, 0.0, 100.0)
    slider_if_exists("windspeed", "Wind speed", c2, 0.0, 25.0)
    slider_if_exists("windgust", "Wind gust", c2, 0.0, 40.0)
    slider_if_exists("sealevelpressure", "Sea level pressure", c3, 950.0, 1050.0)
    slider_if_exists("visibility", "Visibility", c3, 0.0, 30.0)

# Predict
p1 = float(gb_1h.predict(x_row[FEATURE_COLS])[0])
p24 = float(gb_24h.predict(x_row[FEATURE_COLS])[0])

# Display
st.subheader("Selected context")
a, b, c = st.columns(3)
a.write(f"**Station:** {station}")
b.write(f"**Timestamp used:** {row['datetime']}")
c.write(f"**Mode:** {mode}")

st.subheader("Forecast outputs")
m1, m2 = st.columns(2)
m1.metric("PM10 t+1h (µg/m³)", f"{p1:.2f}")
m2.metric("PM10 t+24h (µg/m³)", f"{p24:.2f}")

# Historical demo: compare if target columns exist
y1_col = "y_pm10_t_plus_1h"
y24_col = "y_pm10_t_plus_24h"

if mode.startswith("Historical") and y1_col in df.columns and y24_col in df.columns:
    st.subheader("Actual vs Predicted (backtest demo)")
    actual_1h = row.get(y1_col, np.nan)
    actual_24h = row.get(y24_col, np.nan)

    d1, d2 = st.columns(2)
    if pd.notna(actual_1h):
        d1.metric("Actual t+1h", f"{float(actual_1h):.2f}", delta=f"{(p1-float(actual_1h)):+.2f}")
    else:
        d1.write("Actual t+1h not available.")

    if pd.notna(actual_24h):
        d2.metric("Actual t+24h", f"{float(actual_24h):.2f}", delta=f"{(p24-float(actual_24h)):+.2f}")
    else:
        d2.write("Actual t+24h not available.")

    # ============================================================
    # % Error distribution (station backtest window)
    # ============================================================
    st.subheader("Backtest error distribution (% error)")

    # choose a backtest window (keeps app fast)
    WINDOW = 500
    df_bt = df_s.tail(WINDOW).copy()

    # Ensure actuals exist and drop rows where actual is missing
    df_bt = df_bt.dropna(subset=[y1_col, y24_col])

    if len(df_bt) < 50:
        st.warning("Not enough backtest rows with actual values to plot error distribution.")
    else:
        X_bt = df_bt[FEATURE_COLS]
        y_true_1h = df_bt[y1_col].astype(float).values
        y_true_24h = df_bt[y24_col].astype(float).values

        y_pred_1h = gb_1h.predict(X_bt)
        y_pred_24h = gb_24h.predict(X_bt)

        def pct_error(y_true, y_pred, eps=1e-6):
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
            return 100.0 * (y_pred - y_true) / (np.abs(y_true) + eps)

        def mape(y_true, y_pred, eps=1e-6):
            return float(np.mean(np.abs(pct_error(y_true, y_pred, eps=eps))))

        pe_1h = pct_error(y_true_1h, y_pred_1h)
        pe_24h = pct_error(y_true_24h, y_pred_24h)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("MAPE (t+1h)", f"{mape(y_true_1h, y_pred_1h):.2f}%")
        with col2:
            st.metric("MAPE (t+24h)", f"{mape(y_true_24h, y_pred_24h):.2f}%")

        fig = plt.figure(figsize=(10, 4))
        plt.hist(pe_1h, bins=35, alpha=0.6, label="t+1h % error")
        plt.hist(pe_24h, bins=35, alpha=0.6, label="t+24h % error")
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.xlabel("Percentage error (%)")
        plt.ylabel("Count")
        plt.title("Distribution of percentage error (Predicted vs Actual)")
        plt.legend()
        st.pyplot(fig)

        summary = pd.DataFrame({
            "Horizon": ["t+1h", "t+24h"],
            "Median %Err": [np.median(pe_1h), np.median(pe_24h)],
            "Mean %Err": [np.mean(pe_1h), np.mean(pe_24h)],
            "P5 %Err": [np.percentile(pe_1h, 5), np.percentile(pe_24h, 5)],
            "P95 %Err": [np.percentile(pe_1h, 95), np.percentile(pe_24h, 95)],
        })

        st.dataframe(summary, use_container_width=True)


with st.expander("Show features used (one row)"):
    st.dataframe(x_row[FEATURE_COLS].T.rename(columns={x_row.index[0]: "value"}))

# Download report
st.subheader("Download")
report = {
    "station": [station],
    "timestamp_used": [row["datetime"]],
    "mode": [mode],
    "pred_pm10_t_plus_1h": [p1],
    "pred_pm10_t_plus_24h": [p24],
}
if y1_col in df.columns:
    report["actual_pm10_t_plus_1h"] = [row.get(y1_col, np.nan)]
if y24_col in df.columns:
    report["actual_pm10_t_plus_24h"] = [row.get(y24_col, np.nan)]

report_df = pd.DataFrame(report)
st.download_button(
    "⬇️ Download prediction report (CSV)",
    data=report_df.to_csv(index=False).encode("utf-8"),
    file_name="pm10_prediction_report.csv",
    mime="text/csv",
)
