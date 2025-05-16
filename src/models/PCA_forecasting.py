# In[IMPORTS]
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
# In[===== 1) Parameter & Pfade =====]
project_root = Path().resolve()
processed_folder = project_root / "data" / "processed" / "test_train"
TRAIN_DATA_PATH = processed_folder / "train_splits.xlsx"
TEST_DATA_PATH = processed_folder / "test_splits.xlsx"
# Originaldaten (nicht differenziert, nicht transformiert)
ORIGINAL_DATA_PATH = project_root / "data" / "raw" / "Macro_series_FS25.xlsx"



group_number = 0
split_number = 5

sheet_name_train = f"train_{group_number}_{split_number}"
sheet_name_test = f"test_{group_number}_{split_number}"


# In[===== 2) Daten einlesen =====]
# === Originaldaten einlesen ===
# === 1) Originaldaten laden und vorbereiten ===
df_raw = pd.read_excel(
    ORIGINAL_DATA_PATH,
    sheet_name=0,
    header=None
)

# Erste Zeile als Spaltennamen verwenden
df_raw.columns = ["date"] + df_raw.iloc[0, 1:].tolist()
df_raw = df_raw.drop(index=0).reset_index(drop=True)

# Nur Zeilen behalten mit Quartalsdatum (z. B. "2020 Q1")
df_raw = df_raw[df_raw["date"].notna() & df_raw["date"].astype(str).str.contains("Q")]

# Jahr und Quartal extrahieren
df_raw["year"] = df_raw["date"].astype(str).str.split(" ").str[0].astype(int)
df_raw["quarter"] = df_raw["date"].astype(str).str.split(" ").str[1]
quarter_to_month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
df_raw["month"] = df_raw["quarter"].map(quarter_to_month)

# Zeitstempel erzeugen
df_raw["date_parsed"] = pd.to_datetime(df_raw[["year", "month"]].assign(day=1), errors="coerce")
df_raw = df_raw.dropna(subset=["date_parsed"])

# Numerische Umwandlung aller Spalten außer Datum
for col in df_raw.columns:
    if col not in {"date", "year", "quarter", "month", "date_parsed"}:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

# Finales DataFrame aufbauen
df = df_raw.set_index("date_parsed").drop(columns=["date", "year", "quarter", "month"])
df = df.asfreq("QS").dropna()

# === 2) Log-Differenzierte Zielvariablen erzeugen ===
if "cpi" not in df.columns or "gdpos" not in df.columns:
    raise ValueError(f"Benötigte Spalten fehlen. Verfügbar: {df.columns.tolist()}")

df["inflation"] = np.log(df["cpi"]).diff()
df["g_gdpos"] = np.log(df["gdpos"]).diff()



# === 3) Trainings- und Testdaten laden ===
df_train = pd.read_excel(
    TRAIN_DATA_PATH,
    sheet_name=sheet_name_train,
    parse_dates=['date_parsed']
).set_index('date_parsed').asfreq('QS').dropna()

df_test = pd.read_excel(
    TEST_DATA_PATH,
    sheet_name=sheet_name_test,
    parse_dates=['date_parsed']
).set_index('date_parsed').asfreq('QS').dropna()

split_timestamp = df_test.index.min()

# === 4) Vorbereitung für PCA: relevante Spalten extrahieren ===
exclude_cols = {"date_parsed"}
relevant_cols = [col for col in df_train.columns if col not in exclude_cols]

PCAdf_train = df_train[relevant_cols]
PCAdf_test = df_test[relevant_cols]

scaler = StandardScaler()
PCAdf_train_scaled = scaler.fit_transform(PCAdf_train)
PCAdf_test_scaled = scaler.transform(PCAdf_test)

# In[===== PCA (nur auf Train) =====]
n_components = min(20, PCAdf_train.shape[0], PCAdf_train.shape[1])
pca = PCA(n_components=n_components, svd_solver="full")
X_pca_train = pca.fit_transform(PCAdf_train_scaled)
X_pca_test  = pca.transform(PCAdf_test_scaled)

X_pca_train_df = pd.DataFrame(X_pca_train, index=df_train.index, columns=[f"PC{i+1}" for i in range(n_components)])
X_pca_test_df = pd.DataFrame(X_pca_test, index=df_test.index, columns=[f"PC{i+1}" for i in range(n_components)])


# In[===== Forecasting Model (Train) =====]
n_pcs = 5
lags = 1
PC_cols = [f"PC{i+1}" for i in range(n_pcs)]

X_train = X_pca_train_df[PC_cols].copy()
for i in range(1, lags + 1):
    X_train[f"inflation_lag{i}"] = df_train["inflation"].shift(i)

y_train = df_train["inflation"]
valid_train = X_train.notnull().all(axis=1) & y_train.notnull()
X_train_clean = X_train[valid_train]
y_train_clean = y_train[valid_train]

model = LinearRegression()
model.fit(X_train_clean, y_train_clean)


# In[===== Forecast auf vollen Zeitverlauf (Train + Test) =====]
X_pca_full_df = pd.concat([X_pca_train_df, X_pca_test_df])
df_full = pd.concat([df_train, df_test])

X_full = X_pca_full_df[PC_cols].copy()
for i in range(1, lags + 1):
    X_full[f"inflation_lag{i}"] = df_full["inflation"].shift(i)

y_full = df_full["inflation"]
valid_full = X_full.notnull().all(axis=1) & y_full.notnull()
X_full_clean = X_full[valid_full]
y_full_clean = y_full[valid_full]

y_full_pred = model.predict(X_full_clean)

forecast_df = pd.DataFrame({
    "y_true": y_full_clean.values,
    "y_pred": y_full_pred
}, index=X_full_clean.index)


# In[===== Plot =====]
plt.figure(figsize=(12, 5))
plt.plot(forecast_df.index, forecast_df["y_true"], label="True Inflation", color="black")
plt.plot(forecast_df.index, forecast_df["y_pred"], label="Predicted Inflation", linestyle="--", color="orange")
plt.axvline(x=split_timestamp, color="red", linestyle="dotted", label="Train/Test Split")
plt.title("Forecast: Train + Test (einfaches Modell)")
plt.legend()
plt.tight_layout()
plt.show()


# In[===== Bestes Modell (Train) =====]
#scheint nicht skaliert zu sein
# Parameter
# === Parameter ===
n_pcs = 16
target_vars = ['inflation', 'g_gdpos', 'srate', 'lrate']
min_obs = 30
diff_vars = {"srate", "lrate"}
pc_cols = [f"PC{i + 1}" for i in range(n_pcs)]

pcs_train = X_pca_train_df[pc_cols]
pcs_full = pd.concat([X_pca_train_df[pc_cols], X_pca_test_df[pc_cols]])
df_full = pd.concat([df_train, df_test])

# === Modellschätzung ===
results = {}
for target_var in target_vars:
    y_train = df_train[target_var]
    h = 1
    model_infos = []

    while True:
        y_lag = y_train.shift(h).rename(f"{target_var}_lag{h}")
        pcs_lag = pcs_train.shift(h)
        pcs_lag.columns = [f"{col}_lag{h}" for col in pc_cols]
        X = pd.concat([y_lag, pcs_lag], axis=1)
        y = y_train

        valid = X.notnull().all(axis=1) & y.notnull()
        X_valid = X[valid]
        y_valid = y[valid]

        if len(X_valid) < min_obs:
            break

        model = LinearRegression().fit(X_valid, y_valid)
        mse = mean_squared_error(y_valid, model.predict(X_valid))

        model_infos.append({
            "h": h,
            "mse": mse,
            "n_obs": len(X_valid),
            "model": model,
            "X_cols": X.columns.tolist(),
            "X_valid": X_valid,
            "y_valid": y_valid
        })

        h += 1

    results[target_var] = model_infos


######=============================================######
######===============  ALARMALARM  ===============######
######===============  ALARMALARM  ===============######
######===============  ALARMALARM  ===============######
######=============================================######


######=============================================######
######===============  ALARMALARM  ===============######
######===============  ALARMALARM  ===============######
######===============  ALARMALARM  ===============######
######=============================================######




# In[===== Plot3  =====]

# Alle Zielvariablen, die im Dictionary forecast_results_dict gespeichert wurden
for target_var in ['inflation', 'g_gdpos', 'srate', 'lrate']:
    result = forecast_results_dict[target_var]
    forecast_df = result["forecast_df"]
    mse = result["mse"]

    # Wahre Werte zum Vergleich
    y_true = df_test.loc[forecast_df.index, target_var]

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(y_true.index, y_true, label="True Values", color="black")
    plt.plot(forecast_df.index, forecast_df["forecast"], label="Bhut-Model Forecast", linestyle="--", color="green")

    plt.title(f"{target_var.upper()}: Bhut-Style Forecast vs. True | MSE = {mse:.4f}")
    plt.xlabel("Zeit")
    plt.ylabel(target_var)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# In[===== PlotMSE  =====]


# MSE-Tabelle vorbereiten
mse_data = []

for target_var in forecast_results_dict:
    mse_bhut = forecast_results_dict[target_var]["mse"]
    mse_ar1 = ar_results[target_var]["ar1"]["mse"]
    mse_ar2 = ar_results[target_var]["ar2"]["mse"]

    mse_data.append({
        "target_var": target_var,
        "Bhut-Model": mse_bhut,
        "AR(1)": mse_ar1,
        "AR(2)": mse_ar2
    })

mse_df = pd.DataFrame(mse_data)
mse_df.set_index("target_var", inplace=True)

# Plot 1: inflation & g_gdpos
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for idx, var in enumerate(["inflation", "g_gdpos"]):
    mse_series = mse_df.loc[var]
    axes[idx].bar(mse_series.index, mse_series.values, color=["green", "blue", "purple"])
    axes[idx].set_title(f"{var.upper()}")
    axes[idx].set_ylabel("Mean Squared Error")
    axes[idx].set_xticks(range(len(mse_series.index)))
    axes[idx].set_xticklabels(mse_series.index, rotation=45)
    axes[idx].grid(True, axis="y")

fig.suptitle("MSE Comparison: Inflation & g_gdpos", fontsize=14)
plt.tight_layout()
plt.show()

# Plot 2: srate & lrate
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for idx, var in enumerate(["srate", "lrate"]):
    mse_series = mse_df.loc[var]
    axes[idx].bar(mse_series.index, mse_series.values, color=["green", "blue", "purple"])
    axes[idx].set_title(f"{var.upper()}")
    axes[idx].set_ylabel("Mean Squared Error")
    axes[idx].set_xticks(range(len(mse_series.index)))
    axes[idx].set_xticklabels(mse_series.index, rotation=45)
    axes[idx].grid(True, axis="y")

fig.suptitle("MSE Comparison: srate & lrate", fontsize=14)
plt.tight_layout()
plt.show()






# In[===== 1) Rücktransform =====]

diff_vars = {"srate", "lrate"}

for var in target_vars:
    forecast_df = forecast_results_dict[var]["forecast_df"]
    forecast_diff = forecast_df["forecast"].values
    forecast_index = forecast_df.index

    if var in diff_vars:
        last_train_date = forecast_index.min() - pd.DateOffset(months=3)
        y0 = df[var].asof(last_train_date)  # Robust: nehme letzten bekannten Wert vor Forecast-Beginn
        forecast_level = y0 + np.cumsum(forecast_diff)
    else:
        forecast_level = forecast_diff

    # Forecast-Level als Series mit Index speichern
    forecast_results_dict[var]["forecast_df"]["forecast_level"] = pd.Series(
        forecast_level, index=forecast_index
    )


# In[===== 2) Plot: Forecast vs. Originaldaten (Differenz & Level) =====]
for var in target_vars:
    forecast_df = forecast_results_dict[var]["forecast_df"]

    if "forecast_level" in forecast_df.columns:
        forecast_level = forecast_df["forecast_level"]
        original_series = df[var]

        plt.figure(figsize=(12, 5))
        plt.plot(original_series.index, original_series.values, label="Original Data (Level)", color="black")
        plt.plot(forecast_level.index, forecast_level.values, label="Forecast (Level)", linestyle="--", color="green")
        plt.axvline(x=forecast_level.index.min(), color="red", linestyle=":", label="Forecast-Beginn")
        plt.title(f"{var.upper()}: Forecast Level vs. Original Data")
        plt.xlabel("Zeit")
        plt.ylabel(var)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()