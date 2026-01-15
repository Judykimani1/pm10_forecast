import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# Multi-Pollutant Forecast App (PM10 + PM2.5) — Gradient Boosting
# Includes:
#   - Latest forecast
#   - Historical demo (compare with actual)
#   - Scenario mode sliders
#   - % error distribution plots (true distribution over recent window)
#   - MAE / MAPE per station (over recent window)
# ============================================================

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data"
MODEL_DIR = "models"

# Data files (you can keep one combined dataset if it contains both targets)
DEFAULT_DATA_FILE = "pm10_model_dataset_t1_t24_final.csv"   # must contain station, datetime, features, and y targets for PM10
# If PM2.5 has its own modeling dataset, set it here:
DEFAULT_DATA_FILE_PM25 = "pm25_model_dataset_t1_t24_final.csv"

# Model files
PM10_MODEL_1H = "gb_pm10_t_plus_1h.pkl"
PM10_MODEL_24H = "gb_pm10_t_plus_24h.pkl"

PM25_MODEL_1H = "gb_pm25_t_plus_1h.pkl"
PM25_MODEL_24H = "gb_pm25_t_plus_24h.pkl"

# Target column names
TARGETS = {
    "PM10": ("y_pm10_t_plus_1h", "y_pm10_t_plus_24h"),
    "PM2.5": ("y_pm25_t_plus_1h", "y_pm25_t_plus_24h"),
}

st.set_page_config(page_title="PM Forecast (PM10 + PM2.5)", layout="wide")
st.title("PM Forecasting Interface (Gradient Boosting)")
st.caption("Forecast PM10 and PM2.5 for t+1h and t+24h, with diagnostics (error distributions + MAE/MAPE).")


# -----------------------------
# Helpers
# -----------------------------
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


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPE with protection against divide-by-zero.
    Returns percentage (0-100).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    pe = np.abs((y_pred - y_true) / denom)
    return float(np.nanmean(pe) * 100.0)


def compute_error_distribution(
    df_station: pd.DataFrame,
    feature_cols: list,
    model_1h,
    model_24h,
    y1_col: str,
    y24_col: str,
    window: int = 168,  # last 7 days of hourly data
):
    """
    Compute % error arrays over a rolling historical window for one station.
    Returns (err_1h_array, err_24h_array) or (None, None) if insufficient data.
    """
    needed = set(feature_cols + [y1_col, y24_col])
    if not needed.issubset(set(df_station.columns)):
        return None, None

    df_win = (
        df_station.dropna(subset=feature_cols + [y1_col, y24_col])
        .sort_values("datetime")
        .tail(window)
    )

    if len(df_win) < 10:
        return None, None

    X = df_win[feature_cols]
    y1 = df_win[y1_col].values.astype(float)
    y24 = df_win[y24_col].values.astype(float)

    p1 = model_1h.predict(X).astype(float)
    p24 = model_24h.predict(X).astype(float)

    # % error = 100*(pred-actual)/actual ; protect actual==0
    denom1 = np.where(np.abs(y1) < 1e-9, np.nan, y1)
    denom24 = np.where(np.abs(y24) < 1e-9, np.nan, y24)

    err1 = 100.0 * (p1 - y1) / denom1
    err24 = 100.0 * (p24 - y24) / denom24

    return err1[~np.isnan(err1)], err24[~np.isnan(err24)]


def compute_station_mae_mape(
    df_station: pd.DataFrame,
    feature_cols: list,
    model_1h,
    model_24h,
    y1_col: str,
    y24_col: str,
    window: int = 168
):
    """
    Compute MAE and MAPE for the station over a recent window.
    """
    needed = set(feature_cols + [y1_col, y24_col])
    if not needed.issubset(set(df_station.columns)):
        return None

    df_win = (
        df_station.dropna(subset=feature_cols + [y1_col, y24_col])
        .sort_values("datetime")
        .tail(window)
    )
    if len(df_win) < 10:
        return None

    X = df_win[feature_cols]
    y1 = df_win[y1_col].values.astype(float)
    y24 = df_win[y24_col].values.astype(float)

    p1 = model_1h.predict(X).astype(float)
    p24 = model_24h.predict(X).astype(float)

    mae_1h = float(np.mean(np.abs(p1 - y1)))
    mae_24h = float(np.mean(np.abs(p24 - y24)))

    mape_1h = safe_mape(y1, p1)
    mape_24h = safe_mape(y24, p24)

    return {
        "window_points": int(len(df_win)),
        "mae_1h": mae_1h,
        "mape_1h": mape_1h,
        "mae_24h": mae_24h,
        "mape_24h": mape_24h,
    }


# -----------------------------
# Load assets (PM10 + PM2.5)
# -----------------------------
# Paths
pm10_data_path = os.path.join(DATA_DIR, DEFAULT_DATA_FILE)
pm25_data_path = os.path.join(DATA_DIR, DEFAULT_DATA_FILE_PM25)

pm10_m1_path = os.path.join(MODEL_DIR, PM10_MODEL_1H)
pm10_m24_path = os.path.join(MODEL_DIR, PM10_MODEL_24H)

pm25_m1_path = os.path.join(MODEL_DIR, PM25_MODEL_1H)
pm25_m24_path = os.path.join(MODEL_DIR, PM25_MODEL_24H)

# Check files
require_file(pm10_data_path, "PM10 dataset CSV")
require_file(pm10_m1_path, "PM10 t+1h model file")
require_file(pm10_m24_path, "PM10 t+24h model file")

require_file(pm25_data_path, "PM2.5 dataset CSV")
require_file(pm25_m1_path, "PM2.5 t+1h model file")
require_file(pm25_m24_path, "PM2.5 t+24h model file")

# Load
gb10_1h, gb10_24h = load_models(pm10_m1_path, pm10_m24_path)
gb25_1h, gb25_24h = load_models(pm25_m1_path, pm25_m24_path)

df10 = load_data(pm10_data_path)
df25 = load_data(pm25_data_path)

# Ensure station/datetime exist
for name, df_ in [("PM10", df10), ("PM2.5", df25)]:
    if "station" not in df_.columns or "datetime" not in df_.columns:
        st.error(f"{name} dataset must contain `station` and `datetime` columns.")
        st.stop()

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
show_err = st.sidebar.checkbox("Show % error distribution plots (Historical demo)", value=True)
show_metrics = st.sidebar.checkbox("Show MAE / MAPE per station (Historical demo)", value=True)

window_hours = st.sidebar.slider("Diagnostics window (hours)", 48, 720, 168, step=24)

# Choose dataset/models based on pollutant
if pollutant == "PM10":
    df = df10.copy()
    gb_1h, gb_24h = gb10_1h, gb10_24h
    y1_col, y24_col = TARGETS["PM10"]
else:
    df = df25.copy()
    gb_1h, gb_24h = gb25_1h, gb25_24h
    y1_col, y24_col = TARGETS["PM2.5"]

df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

# Feature columns from model
if not hasattr(gb_1h, "feature_names_in_"):
    st.error("Model missing `feature_names_in_`. Please retrain using pandas DataFrame.")
    st.stop()

FEATURE_COLS = list(gb_1h.feature_names_in_)

missing_features = [c for c in FEATURE_COLS if c not in df.columns]
if missing_features:
    st.error("Dataset is missing required feature columns:")
    st.write(missing_features)
    st.stop()

# Station selection
stations = sorted(df["station"].astype(str).unique().tolist())
station = st.sidebar.selectbox("Station", stations, index=0)

df_s = df[df["station"].astype(str) == str(station)].copy().sort_values("datetime")

# Timestamp selection (for Historical demo)
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

# -----------------------------
# Display: context + forecasts
# -----------------------------
st.subheader("Selected context")
a, b, c = st.columns(3)
a.write(f"**Pollutant:** {pollutant}")
b.write(f"**Station:** {station}")
c.write(f"**Timestamp used:** {row['datetime']}")

st.subheader("Forecast outputs")
m1, m2 = st.columns(2)
m1.metric(f"{pollutant} t+1h (µg/m³)", f"{p1:.2f}")
m2.metric(f"{pollutant} t+24h (µg/m³)", f"{p24:.2f}")

# -----------------------------
# Historical demo: actual vs predicted
# -----------------------------
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

    # --------------------------------------------------------
    # MAE / MAPE per station over recent window
    # --------------------------------------------------------
    if show_metrics:
        st.subheader("Station performance metrics (recent window)")
        stats = compute_station_mae_mape(
            df_s, FEATURE_COLS, gb_1h, gb_24h, y1_col, y24_col, window=window_hours
        )
        if stats is None:
            st.info("Not enough historical data to compute MAE/MAPE for this station.")
        else:
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Window points", f"{stats['window_points']}")
            k2.metric("MAE t+1h", f"{stats['mae_1h']:.2f}")
            k3.metric("MAPE t+1h", f"{stats['mape_1h']:.2f}%")
            k4.metric("MAE t+24h", f"{stats['mae_24h']:.2f}")
            k5.metric("MAPE t+24h", f"{stats['mape_24h']:.2f}%")

    # --------------------------------------------------------
    # % Error distribution plots (true distribution over window)
    # --------------------------------------------------------
    if show_err:
        st.subheader("Percentage error diagnostics (%)")
        err1, err24 = compute_error_distribution(
            df_s, FEATURE_COLS, gb_1h, gb_24h, y1_col, y24_col, window=window_hours
        )

        if err1 is None:
            st.info("Not enough historical data to compute error distributions.")
        else:
            c1, c2 = st.columns(2)

            with c1:
                fig, ax = plt.subplots()
                ax.hist(err1, bins=40, alpha=0.85)
                ax.axvline(0, linestyle="--", color="black")
                ax.set_title(f"{pollutant} t+1h % Error (last {window_hours}h)")
                ax.set_xlabel("Percentage error (%)")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

            with c2:
                fig, ax = plt.subplots()
                ax.hist(err24, bins=40, alpha=0.85)
                ax.axvline(0, linestyle="--", color="black")
                ax.set_title(f"{pollutant} t+24h % Error (last {window_hours}h)")
                ax.set_xlabel("Percentage error (%)")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

# -----------------------------
# Feature viewer
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
