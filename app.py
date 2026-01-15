import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# PM Forecast App (PM10 + PM2.5) — Gradient Boosting
# ============================================================

DATA_DIR = "data"
MODEL_DIR = "models"

# Data files (keep both in repo: data/)
PM10_DATA_FILE = "pm10_model_dataset_t1_t24_final.csv"
PM25_DATA_FILE = "pm25_model_dataset_t1_t24_final.csv"

# Model files (keep both in repo: models/)
PM10_MODEL_1H = "gb_pm10_t_plus_1h.pkl"
PM10_MODEL_24H = "gb_pm10_t_plus_24h.pkl"
PM25_MODEL_1H = "gb_pm25_t_plus_1h.pkl"
PM25_MODEL_24H = "gb_pm25_t_plus_24h.pkl"

st.set_page_config(page_title="PM Forecast (GB)", layout="wide")
st.title("PM10 & PM2.5 Forecasting Interface (Gradient Boosting)")
st.caption("Forecast PM10 and PM2.5 for t+1h and t+24h. Includes latest forecast, historical demo, scenario mode, and % error diagnostics.")

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

def safe_pct_error(pred: float, actual: float) -> float:
    """Percentage error: 100*(pred-actual)/actual, safely handling 0/NaN."""
    if actual is None or pd.isna(actual):
        return np.nan
    actual = float(actual)
    if actual == 0:
        return np.nan
    return 100.0 * (float(pred) - actual) / actual

# -----------------------------
# Sidebar UI
# -----------------------------
st.sidebar.header("Controls")
pollutant = st.sidebar.selectbox("Pollutant", ["PM10", "PM2.5"], index=0)

mode = st.sidebar.radio(
    "Mode",
    ["Latest forecast", "Historical demo (compare with actual)"],
    index=0
)

scenario = st.sidebar.checkbox("Scenario mode (adjust key inputs)", value=False)

show_error_hist = st.sidebar.checkbox("Show % error distribution plots (Historical demo)", value=True)

# -----------------------------
# Select data + models based on pollutant
# -----------------------------
if pollutant == "PM10":
    data_path = os.path.join(DATA_DIR, PM10_DATA_FILE)
    m1_path = os.path.join(MODEL_DIR, PM10_MODEL_1H)
    m24_path = os.path.join(MODEL_DIR, PM10_MODEL_24H)

    y1_col = "y_pm10_t_plus_1h"
    y24_col = "y_pm10_t_plus_24h"
else:
    data_path = os.path.join(DATA_DIR, PM25_DATA_FILE)
    m1_path = os.path.join(MODEL_DIR, PM25_MODEL_1H)
    m24_path = os.path.join(MODEL_DIR, PM25_MODEL_24H)

    y1_col = "y_pm25_t_plus_1h"
    y24_col = "y_pm25_t_plus_24h"

require_file(data_path, f"{pollutant} dataset CSV")
require_file(m1_path, f"{pollutant} t+1h model file")
require_file(m24_path, f"{pollutant} t+24h model file")

gb_1h, gb_24h = load_models(m1_path, m24_path)
df = load_data(data_path)

# -----------------------------
# Feature columns (must match training)
# -----------------------------
if not hasattr(gb_1h, "feature_names_in_"):
    st.error("Model missing `feature_names_in_`. Please retrain using pandas DataFrame so feature order is preserved.")
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
# Station selection
# -----------------------------
stations = sorted(df["station"].astype(str).unique().tolist())
station = st.sidebar.selectbox("Station", stations, index=0)

df_s = df[df["station"].astype(str) == str(station)].copy().sort_values("datetime")

# Pick row
if mode == "Latest forecast":
    row = df_s.iloc[-1].copy()
else:
    # Use a recent window to keep dropdown reasonable
    N = 2000 if len(df_s) > 2000 else len(df_s)
    df_recent = df_s.tail(N).copy()
    ts_list = df_recent["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    picked = st.sidebar.selectbox(
        "Pick a timestamp (recent window)",
        ts_list,
        index=len(ts_list) - 1
    )
    picked_dt = pd.to_datetime(picked)
    row = df_s[df_s["datetime"] == picked_dt].iloc[0].copy()

x_row = row[FEATURE_COLS].to_frame().T

# -----------------------------
# Scenario sliders (optional)
# -----------------------------
if scenario:
    st.info("Scenario mode: adjust a few inputs and observe prediction changes.")
    c1, c2, c3 = st.columns(3)

    def slider_if_exists(colname, label, ui_col, minv, maxv):
        if colname in x_row.columns:
            cur = float(x_row[colname].iloc[0])
            x_row[colname] = ui_col.slider(label, min_value=float(minv), max_value=float(maxv), value=float(cur))

    slider_if_exists("temp", "Temperature", c1, -20.0, 45.0)
    slider_if_exists("humidity", "Humidity", c1, 0.0, 100.0)
    slider_if_exists("windspeed", "Wind speed", c2, 0.0, 25.0)
    slider_if_exists("windgust", "Wind gust", c2, 0.0, 40.0)
    slider_if_exists("sealevelpressure", "Sea level pressure", c3, 950.0, 1050.0)
    slider_if_exists("visibility", "Visibility", c3, 0.0, 30.0)
    slider_if_exists("precip", "Precipitation", c3, 0.0, 30.0)

# -----------------------------
# Predict
# -----------------------------
p1 = float(gb_1h.predict(x_row[FEATURE_COLS])[0])
p24 = float(gb_24h.predict(x_row[FEATURE_COLS])[0])

# -----------------------------
# Display context
# -----------------------------
st.subheader("Selected context")
a, b, c = st.columns(3)
a.write(f"**Pollutant:** {pollutant}")
b.write(f"**Station:** {station}")
c.write(f"**Timestamp used:** {row['datetime']}")

st.write(f"**Mode:** {mode}")

# -----------------------------
# Forecast outputs
# -----------------------------
st.subheader("Forecast outputs")
m1, m2 = st.columns(2)
m1.metric(f"{pollutant} t+1h (µg/m³)", f"{p1:.2f}")
m2.metric(f"{pollutant} t+24h (µg/m³)", f"{p24:.2f}")

# -----------------------------
# Historical demo: compare with actual (if present)
# -----------------------------
if mode.startswith("Historical") and y1_col in df.columns and y24_col in df.columns:
    st.subheader("Actual vs Predicted (backtest demo)")
    actual_1h = row.get(y1_col, np.nan)
    actual_24h = row.get(y24_col, np.nan)

    d1, d2 = st.columns(2)

    if pd.notna(actual_1h):
        d1.metric(
            "Actual t+1h",
            f"{float(actual_1h):.2f}",
            delta=f"{(p1 - float(actual_1h)):+.2f}"
        )
    else:
        d1.write("Actual t+1h not available.")

    if pd.notna(actual_24h):
        d2.metric(
            "Actual t+24h",
            f"{float(actual_24h):.2f}",
            delta=f"{(p24 - float(actual_24h)):+.2f}"
        )
    else:
        d2.write("Actual t+24h not available.")

    # -----------------------------
    # % Error diagnostics (requested)
    # -----------------------------
    if show_error_hist:
        st.subheader("Percentage error diagnostics (%)")

        err_1h = safe_pct_error(p1, actual_1h)
        err_24h = safe_pct_error(p24, actual_24h)

        e1, e2 = st.columns(2)

        with e1:
            if pd.notna(err_1h):
                fig, ax = plt.subplots()
                ax.hist([err_1h], bins=20)
                ax.axvline(0, linestyle="--")
                ax.set_title(f"{pollutant} t+1h % Error (single-point demo)")
                ax.set_xlabel("Percentage error (%)")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
                st.caption(f"t+1h % error: {err_1h:.2f}%")
            else:
                st.write("t+1h % error not available (actual is missing or zero).")

        with e2:
            if pd.notna(err_24h):
                fig, ax = plt.subplots()
                ax.hist([err_24h], bins=20)
                ax.axvline(0, linestyle="--")
                ax.set_title(f"{pollutant} t+24h % Error (single-point demo)")
                ax.set_xlabel("Percentage error (%)")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
                st.caption(f"t+24h % error: {err_24h:.2f}%")
            else:
                st.write("t+24h % error not available (actual is missing or zero).")

else:
    if mode.startswith("Historical"):
        st.info("Historical demo selected, but actual target columns were not found in the dataset.")

# -----------------------------
# Show features
# -----------------------------
with st.expander("Show features used (one row)"):
    st.dataframe(x_row[FEATURE_COLS].T.rename(columns={x_row.index[0]: "value"}))

# -----------------------------
# Download report
# -----------------------------
st.subheader("Download")
report = {
    "pollutant": [pollutant],
    "station": [station],
    "timestamp_used": [row["datetime"]],
    "mode": [mode],
    "pred_t_plus_1h": [p1],
    "pred_t_plus_24h": [p24],
}

if y1_col in df.columns:
    report["actual_t_plus_1h"] = [row.get(y1_col, np.nan)]
if y24_col in df.columns:
    report["actual_t_plus_24h"] = [row.get(y24_col, np.nan)]

report_df = pd.DataFrame(report)

st.download_button(
    "⬇️ Download prediction report (CSV)",
    data=report_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{pollutant.lower()}_prediction_report.csv",
    mime="text/csv",
)
