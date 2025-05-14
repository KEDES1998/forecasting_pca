import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from sklearn.metrics import mean_squared_error
from pathlib import Path
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.ar_model import AutoReg

# In[===== 1) Parameter & Pfade =====]
project_root = Path().resolve().parent.parent
print(f"Projektroot: {project_root}")
processed_folder = project_root / "data" / "processed"
DATA_PATH = processed_folder / "cleaned_data.csv"
STEPS = 1
K_FACTORS = 1   # Anzahl latenter Faktoren
ORDER = 1       # Ordnung der AR-Dynamik im Faktor
MAXITER = 1000  # Max. Iterationen im Fit
min_train_periods = 10   # train period

# In[===== 2) Daten einlesen & vorbereiten =====]
df = pd.read_csv(
    DATA_PATH,
    parse_dates=['date_parsed'],
    index_col='date_parsed'
)

df = df.asfreq('QS')
endog = df[['inflation', 'g_gdpos', 'srate', 'lrate']].dropna()

# Expanding-Window Forecast (one-step ahead)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# DataFrames für DynamicFactor-Forecasts
pred_index    = endog.index[min_train_periods:]
pred_mean_exp = pd.DataFrame(index=pred_index, columns=endog.columns, dtype=float)

# DataFrame für AR(1)-Forecasts
pred_mean_ar  = pd.DataFrame(index=pred_index, columns=endog.columns, dtype=float)

# 3) Expanding-Window Loop
for i in range(min_train_periods, len(endog)):
    train = endog.iloc[:i]
    t_next = endog.index[i]

    # 3.1) DynamicFactor fitten & prognostizieren
    mod_df = DynamicFactor(train, k_factors=K_FACTORS, factor_order=ORDER)
    res_df = mod_df.fit(maxiter=MAXITER, disp=False)
    fcast_df = res_df.get_forecast(steps=STEPS)
    pm_df = fcast_df.predicted_mean.iloc[0]
    pred_mean_exp.loc[t_next] = pm_df.values

    # 3.2) AR(1)-Modell für jede Variable
    for var in endog.columns:
        series = train[var]
        try:
            model_ar = AutoReg(series, lags=1, old_names=False)
            res_ar = model_ar.fit()
            # Vorhersage für den nächsten Zeitpunkt (ein Schritt)
            fc_ar = res_ar.predict(start=len(series), end=len(series))
            pred_mean_ar.loc[t_next, var] = fc_ar.iloc[0]
        except Exception:
            pred_mean_ar.loc[t_next, var] = np.nan

# 4) Forecast-Accuracy berechnen
metrics_exp = pd.DataFrame(index=endog.columns, columns=['RMSE'], dtype=float)
metrics_ar  = pd.DataFrame(index=endog.columns, columns=['RMSE'], dtype=float)

for var in endog.columns:
    y_true = endog[var].loc[pred_mean_exp.index]
    # DynamicFactor
    y_pred_df = pred_mean_exp[var]
    metrics_exp.loc[var, 'RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred_df))
    # AR(1)
    y_pred_ar = pred_mean_ar[var]
    metrics_ar.loc[var, 'RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred_ar))

print("\nExpanding-Window One-Step-Ahead Forecast Accuracy (DynamicFactor):")
print(metrics_exp)
print("\nExpanding-Window One-Step-Ahead Forecast Accuracy (AR(1)):")
print(metrics_ar)

# 5) Vergleichsmatrix erstellen
metrics_compare = pd.concat([
    metrics_exp.rename(columns=lambda c: f"DF_{c}"),
    metrics_ar.rename(columns=lambda c: f"AR1_{c}")
], axis=1)
print("\nForecast Accuracy Comparison:")
print(metrics_compare)

# 6) Plot
for var in endog.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(endog.index, endog[var], label='Actual', color='black')
    plt.plot(pred_mean_exp.index, pred_mean_exp[var], label='DFM Forecast')
    plt.plot(pred_mean_ar.index, pred_mean_ar[var], label='AR(1) Forecast')
    plt.title(f'Expanding-Window {STEPS}-Step Forecast für {var}')
    plt.xlabel('Datum')
    plt.ylabel(var)
    plt.legend()
    plt.tight_layout()
    plt.show()
