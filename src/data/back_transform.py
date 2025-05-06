#!/usr/bin/env python3
# back_transform.py

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import STL

# 1) Paths & parameters
# ---------------------
project_root = Path().resolve().parent.parent
raw_path       = project_root / "data" / "raw" / "Macro_series_FS25.xlsx"
processed      = project_root / "data" / "processed"
results        = project_root / "results"
SPLIT_PARAM    = "0_8"

# --- Load raw data and parse dates (same as your cleaning script) ---
df_raw = pd.read_excel(raw_path)
df_raw.rename(columns={df_raw.columns[0]: "date"}, inplace=True)
df_raw.drop(index=0, inplace=True)
df_raw[["year","quarter"]] = df_raw["date"].str.split(expand=True)
df_raw["year"]  = df_raw["year"].astype(int)
df_raw["month"] = df_raw["quarter"].map({"Q1":1,"Q2":4,"Q3":7,"Q4":10})
df_raw["date_parsed"] = pd.to_datetime(df_raw[["year","month"]].assign(day=1))
df_raw.set_index("date_parsed", inplace=True)
exclude = {"year","month","quarter","date","date_parsed"}
relevant = [c for c in df_raw.columns if c not in exclude]
df_raw[relevant] = df_raw[relevant].apply(pd.to_numeric, errors="coerce").ffill()

# --- Compute (and cache) the raw-seasonal component via STL ---
raw_seasonal = {}
for v in relevant:
    res = STL(df_raw[v], period=4).fit()
    raw_seasonal[v] = res.seasonal

# --- Load the last‐step (deseasoned+diffed+scaled) series & diff orders ---
df_stat       = pd.read_pickle(processed/"cleaned_macro_series2.pkl")
df_stat["date_parsed"] = pd.to_datetime(df_stat["date_parsed"])
df_stat.set_index("date_parsed", inplace=True)

# if you saved diff_order.json, load it; otherwise hard-code:
diff_order_file = processed/"diff_order.json"
if diff_order_file.exists():
    diff_order = json.load(open(diff_order_file))
else:
    diff_order = {"gdp":1,"cpi":1,"lrate":1,"srate":0}

# --- Load the ORIGINAL scaler (on raw data!) ---
orig_scaler = joblib.load(processed/"scaler_original.pkl")
orig_feats  = list(orig_scaler.feature_names_in_)
orig_scale  = orig_scaler.scale_
orig_mean   = orig_scaler.mean_

# Prepare output folders
forecast_in  = results/"forecasts"/"forecast_data"/SPLIT_PARAM
forecast_out = results/"forecasts"/"forecast_data"/f"{SPLIT_PARAM}_orig"
plot_out     = results/"figures"/"forecast_plots"/f"{SPLIT_PARAM}_orig"
for d in (forecast_out, plot_out):
    d.mkdir(exist_ok=True, parents=True)

# --- Back-transform loop ---
for var, d in diff_order.items():
    print(f"\n=== {var.upper()} ===")
    # 1) load your PCA/VAR forecast (scaled+diffed+deseasoned)
    df_fc  = pd.read_pickle(forecast_in/f"{var}_forecast_{SPLIT_PARAM}.pkl")
    dates  = pd.to_datetime(df_fc["date"])
    y_stat = df_fc["forecast"].to_numpy()
    print(" raw y_stat (first 5):", y_stat[:5])

    # 2) invert differencing
    if d > 0:
        last_stat = df_stat[var].iloc[-1]
        y_des     = last_stat + np.cumsum(y_stat)
        print(" last_stat:", last_stat)
    else:
        y_des = y_stat.copy()
        print(" no differencing to invert")
    print(" y_des (first 5):", y_des[:5])

    # 3) invert ORIGINAL scaling (on raw data)
    idx      = orig_feats.index(var)
    y_deseas = y_des * orig_scale[idx] + orig_mean[idx]
    print(f" y_deseas (first 5):", y_deseas[:5],
          f"(scale={orig_scale[idx]:.4f}, mean={orig_mean[idx]:.4f})")

    # 4) re-add the RAW seasonal component
    svals  = raw_seasonal[var].reindex(dates).to_numpy()
    print(" seasonals (first 5):", svals[:5])
    y_orig = y_deseas + svals
    print(" y_orig (first 5):", y_orig[:5])

    # 5) save & plot against THE RAW actuals
    df_out = pd.DataFrame({
        "date": dates,
        "forecast_orig": y_orig
    }).set_index("date")
    df_out.to_pickle (forecast_out / f"{var}_forecast_{SPLIT_PARAM}_orig.pkl")
    df_out.to_excel  (forecast_out / f"{var}_forecast_{SPLIT_PARAM}_orig.xlsx")

    df_act  = df_raw[[var]].rename(columns={var:"actual"}).reindex(dates)
    df_plot = df_act.join(df_out)
    print(" df_plot head:\n", df_plot.head())

    plt.figure(figsize=(10,4))
    plt.plot(df_plot.index, df_plot["actual"],       label="Actual")
    plt.plot(df_plot.index, df_plot["forecast_orig"],label="Forecast", ls="--")
    plt.title(f"{var.upper()} Forecast (original scale)")
    plt.xlabel("Date"); plt.ylabel(var.upper())
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(plot_out / f"{var}_forecast_{SPLIT_PARAM}_orig.png")
    plt.close()

    print(f"{var} done.")