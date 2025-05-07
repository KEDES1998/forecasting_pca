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

SPLIT_PARAMS = [
    "0_3",  "0_35",
    "0_4",  "0_45",
    "0_5",  "0_55",
    "0_6",  "0_65",
    "0_7",  "0_75",
    "0_8"
]

forecast_vars = ["cpi", "gdp", "srate", "lrate"]
n_top_pcs = 20
exclude_cols = {"year", "month", "quarter", "date", "date_parsed", "ngdp",
                    "gdp_prod", "ngdpos", "pgdp", "gdpoi", "gdpos"}

for SPLIT_PARAM in SPLIT_PARAMS:

# In[Setting Path's]
    project_root = Path().resolve().parent.parent
    processed_folder = project_root / "data" / "processed"
    data_train_path = processed_folder / "test_train" / "train_splits.xlsx"
    data_test_path = processed_folder / "test_train" / "test_splits.xlsx"
    pca_path = processed_folder / "pca_outputs" / f"eigenvectors_train_{SPLIT_PARAM}.pkl"
    scaler_path = processed_folder / "pca_outputs" / f"scaler_train_{SPLIT_PARAM}.pkl"

    results_path = project_root / "results"
    forecast_plot_folder = results_path / "figures" /"forecast_plots" / f"{SPLIT_PARAM}"
    forecast_data_folder = results_path / "forecasts" / "forecast_data" /f"{SPLIT_PARAM}"
    forecast_plot_folder.mkdir(parents=True, exist_ok=True)
    forecast_data_folder.mkdir(parents=True, exist_ok=True)

    # In[Loading PCA + Scaler]
    eigen_full = pickle.load(open(pca_path, "rb"))
    eigenvectors = eigen_full.iloc[:20, :]

    # In[Loading Data]
    exclude_cols = {"year", "month", "quarter", "date", "date_parsed", "ngdp",
                    "gdp_prod", "ngdpos", "pgdp", "gdpoi", "gdpos"}

    df_train = pd.read_excel(data_train_path, sheet_name=f"train_{SPLIT_PARAM}")
    X_raw    = df_train[[c for c in df_train.columns if c not in exclude_cols]]

    df_test   = pd.read_excel(data_test_path, sheet_name=f"test_{SPLIT_PARAM}")
    X_test_df = df_test[[c for c in df_test.columns if c not in exclude_cols]]
    date_index = df_test["date_parsed"].values

    # Convert to NumPy arrays (already scaled)
    X_train_vals = X_raw.values
    X_test_vals  = X_test_df.values

    # In[Forecast je Zielvariable basierend auf wichtigsten PCs]

    all_forecasts = {}

    for var in forecast_vars:
        # pick the most important PCs for this variable
        important_pcs = (
            eigenvectors[var]
            .abs()
            .sort_values(ascending=False)
            .index[:n_top_pcs]
        )
        V = eigenvectors.loc[important_pcs]           # shape: (n_top_pcs, n_vars)
        V_mat = V.values.T                             # shape: (n_vars, n_top_pcs)

        # project train data onto these PCs
        factors_train = X_train_vals @ V_mat           # shape: (n_train, n_top_pcs)
        factors_df    = pd.DataFrame(factors_train)

        # fit a VAR(1) on factors
        model   = VAR(factors_df)
        results = model.fit(maxlags=1)

        # prepare for recursive forecasting
        forecasted      = []
        actual          = X_test_df[var].values       # already scaled
        factors_history = factors_train.copy()

        for i in range(len(X_test_vals)):
            # 1-step ahead forecast of factors
            fcast = results.forecast(
                factors_history[-results.k_ar :], steps=1
            )  # shape (1, n_top_pcs)

            # rebuild forecast of the target var via its loading vector
            loading_vec = V[var].values                # length n_top_pcs
            yhat_scaled = fcast @ loading_vec          # shape (1,)
            forecasted.append(yhat_scaled[0])

            # append the next actual factors to the history
            new_x       = X_test_vals[[i]]             # shape (1, n_vars)
            new_factors = new_x @ V_mat                # shape (1, n_top_pcs)
            factors_history = np.vstack([factors_history, new_factors])

        all_forecasts[var] = forecasted

        # plot
        plt.figure(figsize=(10, 4))
        plt.plot(date_index, actual,    label=f"Actual {var.upper()}", color="black")
        plt.plot(date_index, forecasted, label=f"Forecast {var.upper()}", color="red")
        plt.title(f"{var.upper()} Forecast – Top {n_top_pcs} PCs {SPLIT_PARAM}")
        plt.xlabel("Datum")
        plt.ylabel(var.upper())
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # save figure
        plot_path = Path(forecast_plot_folder) / f"{var}_forecast_{SPLIT_PARAM}.png"
        plt.savefig(plot_path)
        plt.show()

        # save forecast data
        forecast_df = pd.DataFrame({
            "date":    date_index,
            "actual":  actual,
            "forecast":forecasted
        })
        forecast_df.to_pickle(Path(forecast_data_folder) / f"{var}_forecast_{SPLIT_PARAM}.pkl")

        # performance
        mse_pcas = mean_squared_error(actual, forecasted)
        print(f"{SPLIT_PARAM}: {var.upper()} – MSE: {mse_pcas:.4f} | Top PCs: {', '.join(important_pcs)}")
