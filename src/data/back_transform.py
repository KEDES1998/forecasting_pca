# === IMPORTS ===
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.seasonal import STL
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# === 1 Pfade & Parameter ===
SPLIT_PARAM = "0_3"
project_root = Path().resolve().parent.parent
raw_folder = project_root / "data" / "raw"
processed_folder = project_root / "data" / "processed"
results_path = project_root / "results"

raw_data_path = raw_folder / "Macro_series_FS25.xlsx"
forecast_data_folder = results_path / "forecasts" / "forecast_data" / SPLIT_PARAM
output_file = results_path / "forecasts" / "forecast_data_transformed.csv"
scaler_path = processed_folder / "pca_outputs" / "scaler_original.pkl"

diff_order = {"gdp": 1, "cpi": 1, "lrate": 1, "srate": 0}

# === 2 Scaler & Forecasts laden ===
scaler = joblib.load(scaler_path)
feature_names = list(scaler.feature_names_in_)

all_forecasts = {}
for var in diff_order.keys():
    pkl_path = forecast_data_folder / f"{var}_forecast_top20_{SPLIT_PARAM}.pkl"
    df_fc = pd.read_pickle(pkl_path)
    dates = pd.to_datetime(df_fc["date"])
    fvals = df_fc["forecast"].values
    all_forecasts[var] = {"dates": dates, "forecast": fvals}

# === 3 Originaldaten + Transformation wie in tscleaning ===
df = pd.read_excel(raw_data_path)
df.rename(columns={df.columns[0]: "date"}, inplace=True)
df.drop(index=0, inplace=True)
df[["year", "quarter"]] = df["date"].str.split(expand=True)
df["year"] = df["year"].astype(int)
df["month"] = df["quarter"].map({"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10})
df["date_parsed"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
df.set_index("date_parsed", inplace=True)

exclude = {"year", "month", "quarter", "date", "date_parsed"}
relevant = [c for c in df.columns if c not in exclude]
df[relevant] = df[relevant].apply(pd.to_numeric, errors="coerce").ffill()

df_std = df.copy()
for c in relevant:
    df_std[c] = (df_std[c] - df_std[c].mean()) / df_std[c].std()

# === Differenzierungen ===
ns1 = [c for c in relevant if adfuller(df_std[c].dropna())[1] > 0.05]
df_diff1 = df_std.copy()
for c in ns1:
    df_diff1[c] = df_std[c].diff()

ns2 = [c for c in relevant if adfuller(df_diff1[c].dropna())[1] > 0.05]
df_diff2 = df_diff1.copy()
for c in ns2:
    df_diff2[c] = df_diff1[c].diff()

# 🔁 Speichere df_diff2 VOR STL zur Rücktransformation
df_before_stl = df_diff2.copy()

df_diff2 = df_diff2.iloc[2:]  # durch 2x diff
df_before_stl = df_before_stl.iloc[2:]

# === Saisonkomponenten mit STL ===
seasonals = {}
for c in relevant:
    s = df_diff2[c].dropna()
    res = STL(s, period=4).fit()
    seasonals[c] = res.seasonal

# === 4 Rücktransformation Forecasts ===
records = []
for var, info in all_forecasts.items():
    dates = info["dates"]
    fvals = info["forecast"]
    print(f"{var}: {np.isnan(fvals).sum()} NaNs im Forecast")

    # 1) Saisonkomponente hinzufügen (noch standardisiert!)
    last_seasonals = seasonals[var].values[-4:]
    seas_fore = np.tile(last_seasonals, int(np.ceil(len(fvals) / 4)))[:len(fvals)]
    y_diff2 = fvals + seas_fore

    # 2) Differenzen rückrechnen (noch standardisiert!)
    d = diff_order[var]
    if d == 2:
        last_vals = df_before_stl[var].iloc[-2:].values
        y_level_std = np.r_[last_vals, y_diff2].cumsum()[2:]
    elif d == 1:
        last_val = df_before_stl[var].iloc[-1]
        y_level_std = np.r_[last_val, y_diff2].cumsum()[1:]
    else:
        y_level_std = y_diff2

    # 3) Entstandardisieren mit ORIGINAL-SCALER
    dummy = np.zeros((len(y_level_std), len(feature_names)))
    idx = feature_names.index(var)
    dummy[:, idx] = y_level_std
    y_level = scaler.inverse_transform(dummy)[:, idx]

    # 4) Plot gegen echte Rohdaten
    y_true = df[var]
    plt.figure(figsize=(10, 4))
    plt.plot(y_true.index, y_true, label=f"Original {var.upper()}", color="black")
    plt.plot(dates, y_level, label=f"Forecast {var.upper()}", color="red")
    plt.title(f"{var.upper()} – Forecast vs. Original")
    plt.xlabel("Datum")
    plt.ylabel(var.upper())
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    for dt, y in zip(dates, y_level):
        records.append({"date": dt, "variable": var, "forecast_original": y})

# === 5 Exportieren ===
df_out = pd.DataFrame.from_records(records)
df_out.to_csv(output_file, index=False)
print(f"Forecasts in Original-Einheiten gespeichert unter: {output_file}")