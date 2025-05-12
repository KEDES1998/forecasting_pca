import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

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


# In[Expanding-Window Forecast (one-step ahead) ===]

warnings.filterwarnings("ignore", category=ConvergenceWarning)

pred_index       = endog.index[min_train_periods:]
pred_mean_exp    = pd.DataFrame(index=pred_index, columns=endog.columns, dtype=float)
ci_lower_exp     = pd.DataFrame(index=pred_index, columns=endog.columns, dtype=float)
ci_upper_exp     = pd.DataFrame(index=pred_index, columns=endog.columns, dtype=float)

# 2) Expanding-Window Loop
for i in range(min_train_periods, len(endog)):
    train_exp = endog.iloc[:i]

    # Modell fitten
    mod_exp = DynamicFactor(train_exp, k_factors=K_FACTORS, factor_order=ORDER)
    res_exp = mod_exp.fit(maxiter=MAXITER, disp=False)

    # Ein-Schritt-Prognose
    fcast_exp = res_exp.get_forecast(steps=STEPS)
    t_next    = fcast_exp.predicted_mean.index[0]

    # Predicted mean
    pm = fcast_exp.predicted_mean.iloc[0]
    pred_mean_exp.loc[t_next] = pm.values

    # CI
    ci_df = fcast_exp.conf_int().loc[t_next]  # Series mit Keys "lower var", "upper var"
    for var in endog.columns:
        ci_lower_exp .loc[t_next, var] = ci_df[f"lower {var}"]
        ci_upper_exp .loc[t_next, var] = ci_df[f"upper {var}"]

# 3) Forecast-Accuracy (one-step ahead)
metrics_exp = pd.DataFrame(index=endog.columns, columns=['RMSE','MAE'], dtype=float)
for var in endog.columns:
    y_true = endog[var].loc[pred_mean_exp.index]
    y_pred = pred_mean_exp[var]
    metrics_exp.loc[var, 'RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics_exp.loc[var, 'MAE']  = mean_absolute_error(y_true, y_pred)

print("\nExpanding-Window One-Step-Ahead Forecast Accuracy:")
print(metrics_exp)

# 4) Plot
for var in endog.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(endog.index,          endog[var],         label='Actual')
    plt.plot(pred_mean_exp.index,  pred_mean_exp[var], label=f'{STEPS}-Step Forecast')
    plt.fill_between(
        pred_mean_exp.index,
        ci_lower_exp[var],
        ci_upper_exp[var],
        alpha=0.3, label='95 % CI'
    )
    plt.title(f'Expanding-Window {STEPS}-Step Forecast für {var}')
    plt.xlabel('Datum')
    plt.ylabel(var)
    plt.legend()
    plt.tight_layout()
    plt.show()