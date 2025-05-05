# In[Imports]

import pandas as pd
import numpy as np
from pathlib import Path

from matplotlib.pyplot import figure
from statsmodels.tsa.seasonal import STL
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# In[Parameters]
SPLIT_PARAM = "0_3"
diff_order = {"gdp": 1, "cpi": 1, "lrate": 1, "srate": 0}

# In[Paths]

project_root         = Path().resolve().parent.parent
raw_folder           = project_root / "data" / "raw"
processed_folder     = project_root / "data" / "processed"
results_path         = project_root / "results"

# inputs
raw_data_path        = raw_folder / "Macro_series_FS25.xlsx"
cleaned_path         = processed_folder / "cleaned_macro_series2.xlsx"
forecast_data_folder = results_path / "forecasts" / "forecast_data" / SPLIT_PARAM
scaler_path          = processed_folder  / "scaler_original.pkl"
seasonal_path        = processed_folder / "seasonal_components.pkl"

# outputs
output_data_folder   = results_path / "forecasts" / "forecast_data" / f"{SPLIT_PARAM}_orig"
output_plot_folder   = results_path / "figures"   / "forecast_plots"   / f"{SPLIT_PARAM}_orig"
output_data_folder.mkdir(parents=True, exist_ok=True)
output_plot_folder.mkdir(parents=True, exist_ok=True)



# In[ raw series (for inverting differencing & seasonality)]
# df_raw.drop(index=0, inplace=True)

df_raw = pd.read_excel(raw_data_path)
df_raw.rename(columns={df_raw.columns[0]: "date"}, inplace=True)
df_raw.drop(index=0, inplace=True)
df_raw[["year", "quarter"]] = df_raw["date"].str.split(expand=True)
df_raw["year"]  = df_raw["year"].astype(int)
df_raw["month"] = df_raw["quarter"].map({"Q1":1, "Q2":4, "Q3":7, "Q4":10})
df_raw["date_parsed"] = pd.to_datetime(df_raw[["year","month"]].assign(day=1))
df_raw.set_index("date_parsed", inplace=True)

df_cleaned = pd.read_excel(cleaned_path)
df_cleaned["date_parsed"] = pd.to_datetime(df_cleaned["date_parsed"])

# In[seasonal components]
seasonals = pd.read_pickle(seasonal_path)

# In[original scaler]

scaler = joblib.load(scaler_path)
feat_names = list(scaler.feature_names_in_)
scale_      = scaler.scale_
mean_       = scaler.mean_

# ----------------------------------------
# Back-transform each forecast
# ----------------------------------------
# In[Back-transform each forecast]

for var, dorder in diff_order.items():
    # load forecasted (scaled & stationary & de-seasonalized)
    df_fc = pd.read_pickle(forecast_data_folder / f"{var}_forecast_{SPLIT_PARAM}.pkl")
    dates = pd.to_datetime(df_fc["date"])
    y_scaled_fc = df_fc["forecast"].values

    # 1) inverse standardization
    idx = feat_names.index(var)
    y_stat = y_scaled_fc * scale_[idx] + mean_[idx]
    # now y_stat is the differenced & season-adjusted series

    # 2) add back seasonal component
    #    seasonal may have same index as df_3; ensure alignment
    seasonal_vals = seasonals[var].reindex(dates).values
    y_diff = y_stat + seasonal_vals

    # 3) invert differencing
    if dorder == 0:
        y_orig = y_diff
    elif dorder == 1:
        # need last observed original value just before first forecast date
        first_date = dates[0]
        # assuming quarterly frequency: subtract 3 months
        last_date = first_date - pd.DateOffset(months=3)
        y_last    = df_raw[var].loc[last_date]
        # cumulative sum of diffs + last observed value
        y_orig = np.cumsum(y_diff) + y_last
    else:
        raise ValueError(f"Unsupported diff order {dorder} for variable {var}")

    # assemble DataFrame
    df_out = pd.DataFrame({
        "date":    dates,
        "forecast_orig": y_orig
    }).set_index("date")

    # 4) merge with actual original series for plotting/comparison
    df_actual = df_raw[[var]].reindex(dates)
    df_plot   = df_actual.rename(columns={var:"actual"}) \
                  .join(df_out)

    # 5) save back-transformed forecasts
    df_out.to_pickle(output_data_folder / f"{var}_forecast_{SPLIT_PARAM}_orig.pkl")
    df_out.to_excel  (output_data_folder / f"{var}_forecast_{SPLIT_PARAM}_orig.xlsx")

    # 6) plot
    plt.figure(figsize=(10,4))
    plt.plot(df_plot.index, df_plot["actual"], label=f"Actual {var.upper()}")
    plt.plot(df_plot.index, df_plot["forecast_orig"],
             linestyle="--", label=f"Forecast {var.upper()}")
    plt.title(f"{var.upper()} Forecast Back-Transformed ({SPLIT_PARAM})")
    plt.xlabel("Datum")
    plt.ylabel(var.upper())
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot_folder / f"{var}_forecast_{SPLIT_PARAM}_orig.png")
    plt.close()

    print(f"[{var}] Back-transformed forecast saved and plotted.")