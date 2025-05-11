import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

# In[===== 1) Parameter & Pfade =====]
project_root = Path().resolve().parent.parent
print(f"Projektroot: {project_root}")
processed_folder = project_root / "data" / "processed"
DATA_PATH = processed_folder / "cleaned_data.csv"
HORIZON  = 8    # Forecast-Horizont in Quartalen
K_FACTORS = 1   # Anzahl latenter Faktoren
ORDER = 1       # Ordnung der AR-Dynamik im Faktor
MAXITER = 1000  # Max. Iterationen im Fit

# In[===== 2) Daten einlesen & vorbereiten =====]
df = pd.read_csv(
    DATA_PATH,
    parse_dates=['date_parsed'],
    index_col='date_parsed'
)

# explizit Quartals-Anfang
df = df.asfreq('QS')

# Endogene Variablen auswählen und fehlende Zeilen entfernen
endog = df[['inflation', 'g_gdpos', 'srate', 'lrate']].dropna()

# In[===== 3) Train/Test-Split =====]
train = endog.iloc[:-HORIZON]
test  = endog.iloc[-HORIZON:]

# In[===== 4) Modell fitting =====]
mod    = DynamicFactor(train, k_factors=K_FACTORS, factor_order=ORDER)
res    = mod.fit(maxiter=MAXITER, disp=False)
print(res.summary())

# In[===== 5) Forecast auf Test-Periode =====]
fcast     = res.get_forecast(steps=HORIZON)
pred_mean = fcast.predicted_mean


ci = fcast.conf_int()

flat_cols = []
for col in ci.columns.to_flat_index():
    if isinstance(col, tuple):
        flat_cols.append("_".join(str(c) for c in col))
    else:
        flat_cols.append(str(col).replace(" ", "_"))
ci.columns = flat_cols

# In[===== 6) Forecast-Accuracy messen =====]
metrics = pd.DataFrame(index=endog.columns, columns=['RMSE','MAE'])
for var in endog.columns:
    y_true = test[var]
    y_pred = pred_mean[var]
    metrics.loc[var, 'RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics.loc[var, 'MAE']  = mean_absolute_error(y_true, y_pred)

print("\nForecast Accuracy (Test-Set):")
print(metrics)

# In[===== 7) Plots: Train vs Test vs Forecast =====]
for var in endog.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(train.index, train[var], label='Train (historic)')
    plt.plot(test.index,  test[var],  label='Test (real)')
    plt.plot(pred_mean.index, pred_mean[var], label='Forecast')
    # falls CI-Spalten vorhanden sind:
    lower_col = f'lower_{var}'
    upper_col = f'upper_{var}'
    if lower_col in ci.columns and upper_col in ci.columns:
        plt.fill_between(
            pred_mean.index,
            ci[lower_col],
            ci[upper_col],
            color='gray', alpha=0.3,
            label='95% CI'
        )
    plt.title(f'DFM Forecast vs. Reality for {var}')
    plt.xlabel('Datum')
    plt.ylabel(var)
    plt.legend()
    plt.tight_layout()
    plt.show()