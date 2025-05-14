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

# In[===== AR(1)/AR(2) Modell (nur Lag, keine PCs) =====]
target_vars = ['inflation', 'g_gdpos', 'srate', 'lrate']
ar_results = {}

for target_var in target_vars:
    print(f"\n=== {target_var.upper()} ===")

    # AR(1): Train
    df_ar1 = pd.DataFrame()
    df_ar1[f"{target_var}_lag1"] = df_train[target_var].shift(1)
    df_ar1[target_var] = df_train[target_var]
    df_ar1_clean = df_ar1.dropna()
    X_ar1 = df_ar1_clean[[f"{target_var}_lag1"]]
    y_ar1 = df_ar1_clean[target_var]
    model_ar1 = LinearRegression().fit(X_ar1, y_ar1)

    # AR(2): Train
    df_ar2 = pd.DataFrame()
    df_ar2[f"{target_var}_lag1"] = df_train[target_var].shift(1)
    df_ar2[f"{target_var}_lag2"] = df_train[target_var].shift(2)
    df_ar2[target_var] = df_train[target_var]
    df_ar2_clean = df_ar2.dropna()
    X_ar2 = df_ar2_clean[[f"{target_var}_lag1", f"{target_var}_lag2"]]
    y_ar2 = df_ar2_clean[target_var]
    model_ar2 = LinearRegression().fit(X_ar2, y_ar2)

    # AR(1): Forecast auf Testdaten
    df_test_ar1 = pd.DataFrame()
    df_test_ar1[f"{target_var}_lag1"] = df_full[target_var].shift(1)
    df_test_ar1[target_var] = df_full[target_var]
    df_test_ar1 = df_test_ar1.loc[df_test.index]
    df_test_ar1_clean = df_test_ar1.dropna()
    X_test_ar1 = df_test_ar1_clean[[f"{target_var}_lag1"]]
    y_test_ar1 = df_test_ar1_clean[target_var]
    y_pred_ar1 = model_ar1.predict(X_test_ar1)
    mse_ar1 = mean_squared_error(y_test_ar1, y_pred_ar1)

    # AR(2): Forecast auf Testdaten
    df_test_ar2 = pd.DataFrame()
    df_test_ar2[f"{target_var}_lag1"] = df_full[target_var].shift(1)
    df_test_ar2[f"{target_var}_lag2"] = df_full[target_var].shift(2)
    df_test_ar2[target_var] = df_full[target_var]
    df_test_ar2 = df_test_ar2.loc[df_test.index]
    df_test_ar2_clean = df_test_ar2.dropna()
    X_test_ar2 = df_test_ar2_clean[[f"{target_var}_lag1", f"{target_var}_lag2"]]
    y_test_ar2 = df_test_ar2_clean[target_var]
    y_pred_ar2 = model_ar2.predict(X_test_ar2)
    mse_ar2 = mean_squared_error(y_test_ar2, y_pred_ar2)

    # MSE-Ausgabe
    print(f"MSE AR(1) (Test): {mse_ar1:.6f}")
    print(f"MSE AR(2) (Test): {mse_ar2:.6f}")

    # Speichern
    ar_results[target_var] = {
        "ar1": {"mse": mse_ar1, "y_true": y_test_ar1, "y_pred": y_pred_ar1, "index": X_test_ar1.index},
        "ar2": {"mse": mse_ar2, "y_true": y_test_ar2, "y_pred": y_pred_ar2, "index": X_test_ar2.index}
    }

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(df_test.index, df_test[target_var], label="True Series", color="black")
    plt.plot(X_test_ar1.index, y_pred_ar1, label=f"AR(1) | MSE={mse_ar1:.4f}", linestyle=":", color="blue")
    plt.plot(X_test_ar2.index, y_pred_ar2, label=f"AR(2) | MSE={mse_ar2:.4f}", linestyle="--", color="purple")
    plt.title(f"{target_var.upper()}: Forecast (Test) vs. True")
    plt.xlabel("Zeit")
    plt.ylabel(target_var)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[===== Selektion Bestes Modell (Train) =====]
# Zielvariablen
target_vars = ['inflation', 'g_gdpos', 'srate', 'lrate']
max_lags = 5
max_pcs = 20

final_forecasts = {}

for target_var in target_vars:
    print(f"\n=== {target_var.upper()} ===")

    # Grid Search zur Auswahl des besten Modells
    best_mse = float("inf")
    best_config = (None, None)
    results = []

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

    # Bestes Modell ausgeben
    best_n_lags, best_n_pcs = best_config
    print(f"Best config: lags={best_n_lags}, pcs={best_n_pcs}, MSE={best_mse:.6f}")

    # Forecast mit bestem Modell
    pc_cols = [f"PC{i + 1}" for i in range(best_n_pcs)]
    df_pcs_full = pd.concat([X_pca_train_df[pc_cols], X_pca_test_df[pc_cols]])
    df_target_full = pd.concat([df_train[target_var], df_test[target_var]])
    horizon = len(df_test)

    forecast_results = []

    for i in range(1, horizon + 1):
        features = []
        for lag in range(1, best_n_lags + 1):
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

        t_i_date = df_test.index[i - 1]

        try:
            X_input_vals = []
            for lag in range(1, best_n_lags + 1):
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
        for lag in range(1, best_n_lags + 1):
            feature_names.append(f"{target_var}_lag{lag}")
            feature_names.extend([f"{col}_lag{lag}" for col in pc_cols])

        X_input_df = pd.DataFrame([X_input_vals], columns=feature_names)
        y_hat = model.predict(X_input_df)[0]
        forecast_results.append((t_i_date, i, y_hat))

    forecast_df = pd.DataFrame(forecast_results, columns=["date", "horizon", "forecast"]).set_index("date")
    y_true_final = df_test.loc[forecast_df.index, target_var].values
    mse_final = mean_squared_error(y_true_final, forecast_df["forecast"].values)

    final_forecasts[target_var] = {
        "forecast_df": forecast_df,
        "mse": mse_final,
        "n_lags": best_n_lags,
        "n_pcs": best_n_pcs
    }

    print(f"Final MSE ({target_var}): {mse_final:.6f}")

# In[===== Bestes Modell (Train) =====]

n_lags = 1
n_pcs = 16
target_vars = ['inflation', 'g_gdpos', 'srate', 'lrate']

forecast_results_dict = {}

for target_var in target_vars:
    print(f"\n=== {target_var.upper()} ===")

    pc_cols = [f"PC{i + 1}" for i in range(n_pcs)]
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
            continue

        model = LinearRegression().fit(X_valid, y_valid)

        t_i_date = df_test.index[i - 1]

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

    # Forecast-DataFrame
    forecast_df = pd.DataFrame(
        forecast_results,
        columns=["date", "horizon", "forecast"]
    ).set_index("date")

    # Wahre Werte und MSE
    y_true = df_test.loc[forecast_df.index, target_var].values
    y_pred = forecast_df["forecast"].values
    mse = mean_squared_error(y_true, y_pred)

    print(f"MSE ({target_var}): {mse:.6f}")

    # Speichern
    forecast_results_dict[target_var] = {
        "forecast_df": forecast_df,
        "mse": mse
    }


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