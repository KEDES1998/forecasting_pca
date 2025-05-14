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

group_number = 0
split_number = 5

sheet_name_train = f"train_{group_number}_{split_number}"
sheet_name_test = f"test_{group_number}_{split_number}"


# In[===== 2) Daten einlesen =====]
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

exclude_cols = {"date_parsed"}
relevant_cols = [col for col in df_train.columns if col not in exclude_cols]

PCAdf_train = df_train[relevant_cols]
PCAdf_test = df_test[relevant_cols]

scaler = StandardScaler()
PCAdf_train_scaled = scaler.fit_transform(PCAdf_train.values)
PCAdf_test_scaled  = scaler.transform(PCAdf_test.values)


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

# In[===== AR(1) Modell (nur Lag, keine PCs) =====]
X_ar_train = pd.DataFrame()
X_ar_train["inflation_lag1"] = df_train["inflation"].shift(1)
y_ar_train = df_train["inflation"]
valid_ar_train = X_ar_train.notnull().all(axis=1) & y_ar_train.notnull()
X_ar_train_clean = X_ar_train[valid_ar_train]
y_ar_train_clean = y_ar_train[valid_ar_train]

model_ar1 = LinearRegression()
model_ar1.fit(X_ar_train_clean, y_ar_train_clean)

# In[===== Forecast auf vollen Zeitverlauf (AR1) =====]
X_ar_full = pd.DataFrame()
X_ar_full["inflation_lag1"] = df_full["inflation"].shift(1)
y_ar_full = df_full["inflation"]
valid_ar_full = X_ar_full.notnull().all(axis=1) & y_ar_full.notnull()
X_ar_full_clean = X_ar_full[valid_ar_full]
y_ar_full_clean = y_ar_full[valid_ar_full]
y_ar_pred = model_ar1.predict(X_ar_full_clean)

# Forecast-DataFrame kombinieren
forecast_df["ar1_pred"] = y_ar_pred
# In[===== Forecast auf vollen Zeitverlauf (AR2) =====]
# AR(2): inflation_t ~ inflation_{t-1} + inflation_{t-2}
X_ar2_full = pd.DataFrame()
X_ar2_full["inflation_lag1"] = df_full["inflation"].shift(1)
X_ar2_full["inflation_lag2"] = df_full["inflation"].shift(2)
y_ar2_full = df_full["inflation"]

# Gültige Beobachtungen auswählen
valid_ar2 = X_ar2_full.notnull().all(axis=1) & y_ar2_full.notnull()
X_ar2_clean = X_ar2_full[valid_ar2]
y_ar2_clean = y_ar2_full[valid_ar2]

# Modell schätzen
model_ar2 = LinearRegression().fit(X_ar2_clean, y_ar2_clean)

# Vorhersage erstellen
y_ar2_pred = model_ar2.predict(X_ar2_clean)

# Forecast-DataFrame sicher erweitern (mit Index-Ausrichtung)
forecast_df.loc[X_ar2_clean.index, "ar2_pred"] = y_ar2_pred

# In[===== Plot: Vergleich Forecast vs. AR(1) =====]
plt.figure(figsize=(12, 5))
plt.plot(forecast_df.index, forecast_df["y_true"], label="True Inflation", color="black")
plt.plot(forecast_df.index, forecast_df["y_pred"], label="Predicted (PCs + Lag)", linestyle="--", color="orange")
plt.plot(forecast_df.index, forecast_df["ar1_pred"], label="AR(1) Prediction", linestyle=":", color="blue")
plt.axvline(x=split_timestamp, color="red", linestyle="dotted", label="Train/Test Split")
plt.title("Forecast Comparison: PCs + Lag vs. AR(1)")
plt.xlabel("Zeit")
plt.ylabel("Inflation")
plt.legend()
plt.tight_layout()
plt.show()

# In[===== Selektion Bestes Modell (Train) =====]
# Parameter
# Parameter
target_var = "inflation"
max_lags = 5
max_pcs =20

# Vorbereitung
results = []
best_mse = float("inf")
best_config = None

for n_lags in range(1, max_lags + 1):
    for n_pcs in range(1, max_pcs + 1):
        pc_cols = [f"PC{i + 1}" for i in range(n_pcs)]
        df_pcs_full = pd.concat([X_pca_train_df[pc_cols], X_pca_test_df[pc_cols]])
        df_target_full = pd.concat([df_train[target_var], df_test[target_var]])
        horizon = len(df_test)

        preds, trues = [], []

        for i in range(1, horizon + 1):
            features = []

            for lag in range(1, n_lags + 1):
                y_lag = df_target_full.shift(i + lag - 1).rename(f"{target_var}_lag{lag}")
                pcs_lag = df_pcs_full[pc_cols].shift(i + lag - 1)
                pcs_lag.columns = [f"{col}_lag{lag}" for col in pc_cols]
                features.append(y_lag)
                features.append(pcs_lag)

            X_train = pd.concat(features, axis=1)
            y_train = df_target_full

            valid = X_train.notnull().all(axis=1) & y_train.notnull()
            X_valid = X_train[valid]
            y_valid = y_train[valid]

            if len(X_valid) < 30:
                continue

            model = LinearRegression().fit(X_valid, y_valid)

            # Forecast: y_{t+i} using info at t
            try:
                t_i = df_test.index[i - 1]
                X_input = []

                for lag in range(1, n_lags + 1):
                    lookup = t_i - pd.DateOffset(months=3 * (i + lag - 1))
                    y_val = df_target_full.loc[lookup]
                    pcs_val = df_pcs_full.loc[lookup, pc_cols]
                    if pd.isna(y_val) or pcs_val.isna().any():
                        raise ValueError
                    X_input.append(y_val)
                    X_input.extend(pcs_val.tolist())

                feature_names = [f"{target_var}_lag{l}" for l in range(1, n_lags + 1)]
                for l in range(1, n_lags + 1):
                    feature_names += [f"{pc}_lag{l}" for pc in pc_cols]

                X_pred_df = pd.DataFrame([X_input], columns=feature_names)
                y_hat = model.predict(X_pred_df)[0]
                y_true = df_target_full.loc[t_i]

                preds.append(y_hat)
                trues.append(y_true)

            except:
                continue

        if preds and len(preds) == len(trues):
            mse = mean_squared_error(trues, preds)
            results.append((n_lags, n_pcs, mse))
            if mse < best_mse:
                best_mse = mse
                best_config = (n_lags, n_pcs)

# Output
print(f"Best MSE: {best_mse:.6f} at n_lags={best_config[0]}, n_pcs={best_config[1]}")

results_df = pd.DataFrame(results, columns=["n_lags", "n_pcs", "mse"])

# In[===== Bestes Modell (Train) =====]

n_pcs = 16
n_lags = 1
target_var = "inflation"
pc_cols = [f"PC{i + 1}" for i in range(n_pcs)]

# Kombinierte Daten
df_pcs_full = pd.concat([X_pca_train_df[pc_cols], X_pca_test_df[pc_cols]])
df_target_full = pd.concat([df_train[target_var], df_test[target_var]])
horizon = len(df_test)

forecast_results = []

for i in range(1, horizon + 1):
    lagged_features = []

    for lag in range(1, n_lags + 1):
        y_lag = df_target_full.shift(i + lag - 1).rename(f"{target_var}_lag{lag}")
        pcs_lag = df_pcs_full[pc_cols].shift(i + lag - 1)
        pcs_lag.columns = [f"{col}_lag{lag}" for col in pc_cols]

        lagged_features.append(y_lag)
        lagged_features.append(pcs_lag)

    X_train = pd.concat(lagged_features, axis=1)
    y_train = df_target_full

    valid = X_train.notnull().all(axis=1) & y_train.notnull()
    X_valid = X_train[valid]
    y_valid = y_train[valid]

    if len(X_valid) < 30:
        print(f"Skipping i={i}: only {len(X_valid)} valid observations.")
        continue

    model = LinearRegression().fit(X_valid, y_valid)

    # Prognosezeitpunkt
    t_i_date = df_test.index[i - 1]
    offset_base = pd.DateOffset(months=3 * i)

    try:
        X_input_vals = []
        for lag in range(1, n_lags + 1):
            offset = pd.DateOffset(months=3 * (i + lag - 1))
            date_lookup = t_i_date - offset
            y_val = df_target_full.loc[date_lookup]
            pcs_val = df_pcs_full.loc[date_lookup, pc_cols]
            if pd.isna(y_val) or pcs_val.isna().any():
                raise ValueError("Missing value")
            X_input_vals.append(y_val)
            X_input_vals.extend(pcs_val.tolist())
    except:
        continue

    feature_names = []
    for lag in range(1, n_lags + 1):
        feature_names.append(f"{target_var}_lag{lag}")
        feature_names.extend([f"{col}_lag{lag}" for col in pc_cols])

    X_input_df = pd.DataFrame([X_input_vals], columns=feature_names)
    y_hat = model.predict(X_input_df)[0]
    forecast_results.append((t_i_date, i, y_hat))

# Ergebnis als DataFrame
forecast_bhut_df = pd.DataFrame(
    forecast_results,
    columns=["date", "horizon", "forecast_custom"]
).set_index("date")

# Wahre Zielwerte
true_inflation_values = df_test.loc[forecast_bhut_df.index, "inflation"].values

# Prognosen des spezifischen Modells (Bhut-Modell)
forecast_bhut_predictions = forecast_bhut_df["forecast_custom"].values

# MSE berechnen
mse_bhut_model = mean_squared_error(true_inflation_values, forecast_bhut_predictions)
print(f"Mean Squared Error (Bhut-Modell): {mse_bhut_model:.6f}")


# In[===== Plot3  =====]

# Forecast-Werte aus Bhut-Modell ergänzen
forecast_df.loc[forecast_bhut_df.index, "forecast_custom"] = forecast_bhut_df["forecast_custom"].values
# Plot
plt.figure(figsize=(12, 5))

# Wahre Werte
plt.plot(forecast_df.index, forecast_df["y_true"], label="True Inflation", color="black")

# Originalmodell: PCs + y_{t-1}
plt.plot(forecast_df.index, forecast_df["y_pred"], label="Predicted (PCs + y_{t-1})", linestyle="--", color="orange")

# AR(1)
plt.plot(forecast_df.index, forecast_df["ar1_pred"], label="AR(1) Prediction", linestyle=":", color="blue")

# AR(2)
plt.plot(forecast_df.index, forecast_df["ar2_pred"], label="AR(2) Prediction", linestyle=":", color="purple")

# Neues Modell: y_t ~ y_{t-i} + PCs_{t-i}, angewandt auf t+i
plt.plot(forecast_df.index, forecast_df["forecast_custom"], label="Forecast: y_t ~ PCs + y_{t-i}", linestyle="-.", color="green")

# Split-Linie
plt.axvline(x=split_timestamp, color="red", linestyle="dotted", label="Train/Test Split")

# Formatierung
plt.title("Forecast Comparison: AR(1), AR(2), PCs + y_{t-1}, and Bhut-Style Model y_{t-i}")
plt.xlabel("Zeit")
plt.ylabel("Inflation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()