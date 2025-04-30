# In[IMPORTS]
import pickle
import joblib
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# In[Setting Parameters]
SPLIT_PARAM = "0_3"

# In[Setting Path's]
project_root = Path().resolve()
processed_folder = project_root / "data" / "processed"
data_train_path = processed_folder / "test_train" / "train_splits.xlsx"
data_test_path = processed_folder / "test_train" / "test_splits.xlsx"
pca_path = processed_folder / "pca_outputs" / f"eigenvectors_train_{SPLIT_PARAM}.pkl"
scaler_path = processed_folder / "pca_outputs" / f"scaler_train_{SPLIT_PARAM}.pkl"

results_path = project_root / "results"
forecast_plot_folder = results_path / "figures"
forecast_data_folder = results_path / "forecasts" / "forecast_data"
forecast_plot_folder.mkdir(parents=True, exist_ok=True)
forecast_data_folder.mkdir(parents=True, exist_ok=True)

# In[Loading PCA + Scaler]
eigen_full = pickle.load(open(pca_path, "rb"))
eigenvectors = eigen_full.iloc[:20, :]
scaler = joblib.load(scaler_path)

# In[Loading Data]
exclude_cols = {"year", "month", "quarter", "date", "date_parsed", "ngdp",
                "gdp_prod", "ngdpos", "pgdp", "gdpoi", "gdpos"}

df_train = pd.read_excel(data_train_path, sheet_name=f"train_{SPLIT_PARAM}")
X_raw = df_train[[col for col in df_train.columns if col not in exclude_cols]]

df_test = pd.read_excel(data_test_path, sheet_name=f"test_{SPLIT_PARAM}")
X_test = df_test[[col for col in df_test.columns if col not in exclude_cols]]
date_index = df_test["date_parsed"].values

# In[Standardisierung]
X = scaler.transform(X_raw)

# In[Forecast je Zielvariable basierend auf wichtigsten PCs]
forecast_vars = ["cpi", "gdp", "srate", "lrate"]
n_top_pcs = 5
all_forecasts = {}

for var in forecast_vars:
    important_pcs = eigenvectors[var].abs().sort_values(ascending=False).index[:n_top_pcs]
    V = eigenvectors.loc[important_pcs]

    factors_train = X @ V.T
    factors_df = pd.DataFrame(factors_train)

    model = VAR(factors_df)
    results = model.fit(maxlags=1)

    forecasted = []
    actual = X_test[var].values
    factors_history = factors_df.values.copy()

    for i in range(len(X_test)):
        forecast = results.forecast(factors_history[-results.k_ar:], steps=1)
        loading_vector = V[var].values
        forecast_std = forecast @ loading_vector

        col_idx = X_raw.columns.get_loc(var)
        mean = scaler.mean_[col_idx]
        scale = scaler.scale_[col_idx]
        unscaled_value = forecast_std[0] * scale + mean
        forecasted.append(unscaled_value)

        new_x_scaled = scaler.transform(X_test.iloc[[i]])
        new_factors = new_x_scaled @ V.T
        factors_history = np.vstack([factors_history, new_factors])

    all_forecasts[var] = forecasted

    plt.figure(figsize=(10, 4))
    plt.plot(date_index, actual, label=f"Actual {var.upper()}", color="black")
    plt.plot(date_index, forecasted, label=f"Forecast {var.upper()}", color="red")
    plt.title(f"{var.upper()} Forecast – Top {n_top_pcs} PCs")
    plt.xlabel("Datum")
    plt.ylabel(var.upper())
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = forecast_plot_folder / f"{var}_forecast_top{n_top_pcs}.png"
    plt.savefig(plot_path)
    plt.show()

    forecast_df = pd.DataFrame({
        "date": date_index,
        "actual": actual,
        "forecast": forecasted
    })
    forecast_df.to_pickle(forecast_data_folder / f"{var}_forecast_top{n_top_pcs}.pkl")

    mse = mean_squared_error(actual, forecasted)
    print(f"{var.upper()} – MSE: {mse:.4f} | Top PCs: {', '.join(important_pcs)}")

# In[Forecast auf Level zurückbringen (GDP)]
# === GDP-Level-Forecast rekonstruieren ===

######
#Habe ich noch nicht geschafft