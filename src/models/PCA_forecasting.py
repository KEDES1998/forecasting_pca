# In[IMPORTS]
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


# In[===== 1) Parameter & Pfade =====]
project_root = Path().resolve().parent.parent
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

exclude_cols = {"year", "month", "quarter", "date", "date_parsed", "pgdpos", "pgdpoi"}
relevant_cols = [col for col in df_train.columns if col not in exclude_cols]

PCAdf_train = df_train[relevant_cols]
PCAdf_test = df_test[relevant_cols]


# In[===== PCA (nur auf Train) =====]
n_components = min(20, PCAdf_train.shape[0], PCAdf_train.shape[1])
pca = PCA(n_components=n_components, svd_solver="full")
X_pca_train = pca.fit_transform(PCAdf_train.values)
X_pca_test = pca.transform(PCAdf_test.values)

X_pca_train_df = pd.DataFrame(X_pca_train, index=df_train.index, columns=[f"PC{i+1}" for i in range(n_components)])
X_pca_test_df = pd.DataFrame(X_pca_test, index=df_test.index, columns=[f"PC{i+1}" for i in range(n_components)])


# In[===== Forecasting Model (Train) =====]
n_pcs = 5
lags = 2
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