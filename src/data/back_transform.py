# In[ IMPORT ]

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.seasonal import STL

# In[=== 1 Pfade und Parameter ===]

project_root = Path().resolve().parent.parent
raw_folder = project_root / "data" / "raw"
results_path = project_root / "results"

raw_data_path = raw_folder / "Macro_series_FS25.xlsx"                  # Dein Original-Excel vor der Bereinigung
processed_folder = project_root / "data" / "processed"                   # Enthält cleaned_macro_series2.xlsx
forecast_data_folder = results_path / "forecasts" / "forecast_data"   # Enthält die *.pkl Forecast-Outputs
output_file = results_path / "forecasts" / "forecast_data_transformed"  # Wo die finalen Werte hin sollen

# Definiere hier, wie oft jede Variable differenziert wurde (d = 0,1 oder 2)
diff_order = {"gdp":1, "cpi":1, "lrate": 1, "srate": 0}


# In[=== 2 Forecasts laden ===]

all_forecasts = {}
for var in diff_order.keys():
    pkl = forecast_data_folder / f"{var}_forecast_top20.pkl"
    df_fc = pd.read_pickle(pkl)

    # 2) Spalte "date" in Pandas-Timestamps umwandeln
    dates = pd.to_datetime(df_fc["date"])

    # 3) Forecast-Werte als NumPy-Array extrahieren
    fvals = df_fc["forecast"].values

    # 4) Im Dictionary speichern
    all_forecasts[var] = {
        "dates":   dates,
        "forecast": fvals
    }

# In[=== 3 Den Vor-Processing-Pipeline rekonstruieren ===]

# Wir laden noch einmal das Original und bauen df_3_input exakt so nach,

df = pd.read_excel(raw_data_path)
df.rename(columns={df.columns[0]: "date"}, inplace=True)
df.drop(index=0, inplace=True)

# Datum parsen
df[["year","quarter"]] = df["date"].str.split(expand=True)
df["year"]    = df["year"].astype(int)
df["month"]   = df["quarter"].map({"Q1":1,"Q2":4,"Q3":7,"Q4":10})
df["date_parsed"] = pd.to_datetime(df[["year","month"]].assign(day=1))

# Numerics & NaN
exclude = {"year","month","quarter","date","date_parsed"}
relevant = [c for c in df.columns if c not in exclude]
for c in relevant:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c].fillna(method="ffill", inplace=True)

# 1. Differenzierung für non_stationary1
from statsmodels.tsa.stattools import adfuller
def identify_ns(df, cols, alpha=0.05):
    out = []
    for c in cols:
        p = adfuller(df[c].dropna())[1]
        if p > alpha: out.append(c)
    return out

ns1 = identify_ns(df, relevant)
df2 = df.copy()
for c in ns1:
    df2[c] = df[c].diff()

# 2. Differenzierung für non_stationary2
ns2 = identify_ns(df2, relevant)
df3_input = df2.copy()
for c in ns2:
    df3_input[c] = df2[c].diff()

# 3. Saisonale Komponente aus df3_input extrahieren
seasonals = {}
for c in relevant:
    s = df3_input[c].dropna()
    res = STL(s, period=4).fit()
    seasonals[c] = res.seasonal

# 4. Aufräumen: die ersten 2 Zeilen fallen weg durch bis zu 2 diffs
df3_input = df3_input.iloc[2:].set_index("date_parsed")


# In[=== 4 Rücktransformation der Forecasts ===]

records = []
for var, info in all_forecasts.items():
    dates = info["dates"]
    fvals = np.array(info["forecast"])
    print(f"{var}: {np.isnan(fvals).sum()} NaNs im Forecast")

    # 1) Saisonkomponente holen – OHNE reindex()
    seas_vals = seasonals[var].values
    if len(seas_vals) < 4:
        raise ValueError(f"Nicht genug saisonale Werte für {var}")
    last_four = seas_vals[-4:]

    # 2) auf Forecast-Horizont ausdehnen
    seas_fore = np.tile(last_four, int(np.ceil(len(fvals)/4)))[:len(fvals)]
    y_diff2 = fvals + seas_fore

    # 3) Differenzinvertierung
    d = diff_order[var]
    if d > 0:
        last_vals = df3_input[var].iloc[-d:].values
        y_level = np.r_[last_vals, y_diff2].cumsum()[d:]
    else:
        y_level = y_diff2

    if np.isnan(y_level).all():
        print(f"Achtung: y_level für {var} ist immer noch komplett NaN!")

    for dt, y in zip(dates, y_level):
        records.append({"date": dt, "variable": var, "forecast_original": y})



# 5) Ergebnis abspeichern
df_out = pd.DataFrame.from_records(records)
df_out.to_csv(output_file, index=False)
print(f"Original-Einheiten Forecasts gespeichert in {output_file}")
