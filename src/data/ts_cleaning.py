# In[1]
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
import joblib

########################################
##########     SETUP PFAD     ##########
########################################

# In[Pfad Setup]
project_root = Path().resolve().parent.parent
raw_folder = project_root / "data" / "raw"
raw_data = raw_folder / "Macro_series_FS25.xlsx"
processed_folder = project_root / "data" / "processed"
pca_output_folder = processed_folder / "pca_outputs"
pca_output_folder.mkdir(parents=True, exist_ok=True)

########################################
##########     RAW DATA LOAD    ########
########################################

# In[Rohdaten laden & vorbereiten]
df_raw = pd.read_excel("/Users/Mshaise/Desktop/Office/forecasting_pca/data/raw/Macro_series_FS25.xlsx")
df_raw.rename(columns={df_raw.columns[0]: "date"}, inplace=True)
df_raw.drop(index=0, inplace=True)
df_raw[["year", "quarter"]] = df_raw["date"].str.split(expand=True)
df_raw["year"] = df_raw["year"].astype(int)
df_raw["month"] = df_raw["quarter"].map({"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10})
df_raw["date_parsed"] = pd.to_datetime(df_raw[["year", "month"]].assign(day=1))

exclude_cols = {"year", "month", "quarter", "date", "date_parsed"}
relevant_cols = [c for c in df_raw.columns if c not in exclude_cols]

for c in relevant_cols:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
df_raw[relevant_cols] = df_raw[relevant_cols].ffill()

########################################
##########   SCALER ORIGINAL   #########
########################################

# In[Originalscaler speichern]
scaler_original = StandardScaler()
scaler_original.fit(df_raw[relevant_cols])
scaler_path_original = pca_output_folder / "scaler_original.pkl"
joblib.dump(scaler_original, scaler_path_original)
print(f"Original-Scaler gespeichert unter {scaler_path_original}")

########################################
##########  TRANSFORMIERTES DF  ########
########################################

# In[Standardisierung + Stationarität]
df = df_raw.copy()
df[relevant_cols] = scaler_original.transform(df[relevant_cols])

# ADF-Test
non_stationary1 = [c for c in relevant_cols if adfuller(df[c].dropna())[1] > 0.05]

# In[Differenzierung 1]
df_2 = df.copy()
for col in non_stationary1:
    df_2[col] = df[col].diff()

# Check Stationarität
non_stationary2 = [c for c in relevant_cols if adfuller(df_2[c].dropna())[1] > 0.05]

# In[Differenzierung 2]
df_3 = df_2.copy()
for col in non_stationary2:
    df_3[col] = df_2[col].diff()

df_3 = df_3.iloc[2:]  # wegen zwei Differenzierungen

# In[Saisonalität entfernen]
for col in df_3.columns:
    if col not in exclude_cols:
        series = df_3[col].dropna()
        res = STL(series, period=4).fit()
        df_3[col] = series - res.seasonal

########################################
##########     SPEICHERN       #########
########################################

# In[Speichern der verarbeiteten Daten]
processed_file_xlsx = processed_folder / "cleaned_macro_series2.xlsx"
processed_file_pkl = processed_folder / "cleaned_macro_series2.pkl"
df_3.to_excel(processed_file_xlsx, index=False)
df_3.to_pickle(processed_file_pkl)

########################################
##########   SCALER TRANSFORM   ########
########################################

# In[Skalierer auf transformierten Daten speichern]
scaler_transformed = StandardScaler()
scaler_transformed.fit(df_3[relevant_cols])
scaler_path_transformed = pca_output_folder / "scaler_transformed.pkl"
joblib.dump(scaler_transformed, scaler_path_transformed)
print(f"Transformierter Scaler gespeichert unter {scaler_path_transformed}")
