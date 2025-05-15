import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from sklearn.metrics import mean_squared_error
from pathlib import Path
import warnings
from itertools import product

# 1) Parameter & Pfade
project_root       = Path().resolve().parent.parent
processed_folder   = project_root / "data" / "processed"
DATA_PATH          = processed_folder / "cleaned_data.csv"

STEPS              = 1
K_FACTORS_LIST     = [1, 2, 3]
ORDER_LIST         = [1, 2, 3]
MAXITER            = 1000
min_train_periods  = 10

# 2) Daten einlesen & vorbereiten
df = pd.read_csv(
    DATA_PATH,
    parse_dates=['date_parsed'],
    index_col='date_parsed'
).asfreq('QS')
endog = df[['inflation', 'g_gdpos', 'srate', 'lrate']].dropna()

# 3) Warnings unterdrücken
warnings.filterwarnings("ignore")

# 4) Grid-Search
results = []
for k_factors, order in product(K_FACTORS_LIST, ORDER_LIST):
    print(f"Teste Modell: k_factors={k_factors}, factor_order={order}")
    pred_index = endog.index[min_train_periods:]
    pred_mean  = pd.DataFrame(index=pred_index, columns=endog.columns, dtype=float)

    # Expanding-Window Forecast
    for i in range(min_train_periods, len(endog)):
        train  = endog.iloc[:i]
        t_next = endog.index[i]
        try:
            mod = DynamicFactor(train, k_factors=k_factors, factor_order=order)
            res = mod.fit(maxiter=MAXITER, disp=False)
            fcast = res.get_forecast(steps=STEPS)
            pred_mean.loc[t_next] = fcast.predicted_mean.iloc[0].values
        except Exception as e:
            pred_mean.loc[t_next] = np.nan

    # RMSE pro Variable (NaNs rausfiltern)
    rmses = {}
    for var in endog.columns:
        y_true = endog[var].loc[pred_mean.index]
        y_pred = pred_mean[var]
        mask   = y_pred.notna()
        if mask.sum() == 0:
            rmses[var] = np.nan
        else:
            rmses[var] = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    avg_rmse = np.nanmean(list(rmses.values()))

    results.append({
        'k_factors':    k_factors,
        'factor_order': order,
        **rmses,
        'avg_rmse':     avg_rmse
    })

# 5) Beste Kombination finden
results_df = pd.DataFrame(results).sort_values('avg_rmse').reset_index(drop=True)
best       = results_df.iloc[0]
print("\nGrid-Search Ergebnisse:")
print(results_df)
print("\nBeste Parameterkombination:")
print(best)

# 6) Finale Forecasts mit den besten Parametern
bf_k = int(best['k_factors'])
bf_o = int(best['factor_order'])
print(f"\nErstelle finale Forecasts mit k_factors={bf_k}, factor_order={bf_o}")

pred_index         = endog.index[min_train_periods:]
pred_mean_best     = pd.DataFrame(index=pred_index, columns=endog.columns, dtype=float)
for i in range(min_train_periods, len(endog)):
    train  = endog.iloc[:i]
    t_next = endog.index[i]
    mod    = DynamicFactor(train, k_factors=bf_k, factor_order=bf_o)
    res    = mod.fit(maxiter=MAXITER, disp=False)
    fcast  = res.get_forecast(steps=STEPS)
    pred_mean_best.loc[t_next] = fcast.predicted_mean.iloc[0].values

# 7) Plots
for var in endog.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(endog.index,          endog[var],          label='Actual', color='black')
    plt.plot(pred_mean_best.index, pred_mean_best[var], label='DFM Forecast')
    plt.title(f'Expanding-Window {STEPS}-Step Forecast für {var}\n'
              f'(k_factors={bf_k}, factor_order={bf_o})')
    plt.xlabel('Datum')
    plt.ylabel(var)
    plt.legend()
    plt.tight_layout()
    plt.show()
