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
OUTPUT_PATH_PLOTS = project_root / "results" / "figures" / "forecast_plots" / "DFM"

####################
#### PARAMETER #####
####################

STEPS = 8       # Forecast Horizont (Q)
K_FACTORS = 1   # Anzahl latenter Faktoren (Q)
ORDER = 2       # AR-Dynamik -> AR(ORDER) (Q)
MAXITER = 1000  # Max. Iterationen im Fit
min_train_periods = 60   # train period
MAX_HORIZON = 8

AR_LAG = 1

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

#######################################
#### EXPANDING WINDOW DFM FORECAST ####
#######################################

for i in range(min_train_periods, len(endog)):
    train = endog.iloc[:i]
    t_next = endog.index[i]

    # 3.1) DynamicFactor fitten & prognostizieren
    mod_df = DynamicFactor(train, k_factors=K_FACTORS, factor_order=ORDER)
    res_df = mod_df.fit(maxiter=MAXITER, disp=False)
    fcast_df = res_df.get_forecast(steps=STEPS)
    pm_df = fcast_df.predicted_mean.iloc[0]
    pred_mean_exp.loc[t_next] = pm_df.values

    #### AR() ####
    for var in endog.columns:
        series = train[var]
        try:
            model_ar = AutoReg(series, lags=AR_LAG, old_names=False)
            res_ar = model_ar.fit()
            # Vorhersage für den nächsten Zeitpunkt (ein Schritt)
            fc_ar = res_ar.predict(start=len(series), end=len(series))
            pred_mean_ar.loc[t_next, var] = fc_ar.iloc[0]
        except Exception:
            pred_mean_ar.loc[t_next, var] = np.nan

# 4) Forecast-Accuracy
metrics_exp = pd.DataFrame(index=endog.columns, columns=['RMSE'], dtype=float)
metrics_ar  = pd.DataFrame(index=endog.columns, columns=['RMSE'], dtype=float)

for var in endog.columns:
    y_true = endog[var].loc[pred_mean_exp.index]
    #### EXPANDING WINDOW DFM  ####
    y_pred_df = pred_mean_exp[var]
    metrics_exp.loc[var, 'RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred_df))
    #### AR() ####
    y_pred_ar = pred_mean_ar[var]
    metrics_ar.loc[var, 'RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred_ar))

print("\nExpanding-Window One-Step-Ahead Forecast Accuracy (DynamicFactor):")
print(metrics_exp)
print(f"\nExpanding-Window One-Step-Ahead Forecast Accuracy (AR({AR_LAG})):")
print(metrics_ar)

# 5) Vergleichsmatrix erstellen
metrics_compare = pd.concat([
    metrics_exp.rename(columns=lambda c: f"DF_{c}"),
    metrics_ar.rename(columns=lambda c: f"AR{AR_LAG}_{c}")
], axis=1)
print("\nForecast Accuracy Comparison:")
print(metrics_compare)

################################################################
#######################  FORECAST PLOTS ########################
################################################################

for var in endog.columns:
    fig = plt.figure(figsize=(10, 4))
    # Forecast-Kurven
    plt.plot(pred_mean_exp.index, pred_mean_exp[var],
             label='DFM Forecast', color='red', linestyle='dashed')
    plt.plot(pred_mean_ar.index, pred_mean_ar[var],
             label=f'AR({AR_LAG}) Forecast', color='blue', linestyle='dashed')
    # Actual nur ab Forecast-Beginn
    forecast_start = pred_mean_exp.index.min()
    actual_fc = endog.loc[forecast_start:, var]
    plt.plot(actual_fc.index, actual_fc.values,
             label='Actual', color='black')

    plt.title(f'Expanding-Window {STEPS}-Step Forecast for {var}')
    plt.xlabel('Datum')
    plt.ylabel(var)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(OUTPUT_PATH_PLOTS / f"{STEPS}-Step Forecast for {var}_DFM.png")
    plt.show()


################################################################
#########################  RMSE Plots ##########################
################################################################

pred_index = endog.index[min_train_periods:]

# Storage für alle Horizonte und beide Modelle
preds_exp = {h: pd.DataFrame(index=pred_index, columns=endog.columns, dtype=float)
             for h in range(1, MAX_HORIZON+1)}
preds_ar  = {h: pd.DataFrame(index=pred_index, columns=endog.columns, dtype=float)
             for h in range(1, MAX_HORIZON+1)}

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Expanding‐Window Loop über Zeitpunkte
for i in range(min_train_periods, len(endog)):
    train = endog.iloc[:i]
    # 1) DynamicFactor Mehrschritt‐Forecast
    mod_df = DynamicFactor(train, k_factors=K_FACTORS, factor_order=ORDER)
    res_df = mod_df.fit(maxiter=MAXITER, disp=False)
    fcast = res_df.get_forecast(steps=MAX_HORIZON).predicted_mean

    for h in range(1, MAX_HORIZON+1):
        # falls Horizon h verfügbar
        if h <= fcast.shape[0]:
            preds_exp[h].iloc[i-min_train_periods] = fcast.iloc[h-1].values

    # 2) AR(1) Mehrschritt‐Forecast
    for var in endog.columns:
        series = train[var]
        try:
            model_ar = AutoReg(series, lags=AR_LAG, old_names=False)
            res_ar = model_ar.fit()
            # dynamische Mehrschritt‐Prognose
            fc_ar = res_ar.predict(start=len(series),
                                   end=len(series)+MAX_HORIZON-1)
            for h in range(1, MAX_HORIZON+1):
                preds_ar[h].iloc[i-min_train_periods, preds_ar[h].columns.get_loc(var)] = fc_ar.iloc[h-1]
        except Exception:
            # fehlgeschlagene Prognose => NaN
            preds_ar[h].loc[preds_ar[h].index[i-min_train_periods], var] = np.nan

# 4) RMSE berechnen pro Horizon und Gesamt‐RMSE
rmse_exp = []
rmse_ar  = []
for h in range(1, MAX_HORIZON+1):
    # für dieses h alle Variablen und Zeitpunkte auf einen Vektor abflachen
    y_true = []
    y_e = []
    y_a = []
    for var in endog.columns:
        idx = preds_exp[h].index
        y_true.extend(endog[var].loc[idx].values)
        y_e.extend(preds_exp[h][var].values)
        y_a.extend(preds_ar[h][var].values)
    # rechnen
    rmse_exp.append(np.sqrt(mean_squared_error(y_true, y_e)))
    rmse_ar.append( np.sqrt(mean_squared_error(y_true, y_a)) )

# Gesamt‐RMSE über alle Horizonte
y_true_all = []
y_e_all    = []
y_a_all    = []
for h in range(1, MAX_HORIZON+1):
    for var in endog.columns:
        idx = preds_exp[h].index
        y_true_all.extend(endog[var].loc[idx].values)
        y_e_all.extend(preds_exp[h][var].values)
        y_a_all.extend(preds_ar[h][var].values)
total_rmse_exp = np.sqrt(mean_squared_error(y_true_all, y_e_all))
total_rmse_ar  = np.sqrt(mean_squared_error(y_true_all, y_a_all))

# Zusammenstellen ins DataFrame für Plot
labels = ['Total'] + [f'{h}-Step' for h in range(1, MAX_HORIZON+1)]
df_plot = pd.DataFrame({
    'DFM': [total_rmse_exp] + rmse_exp,
    f'AR({AR_LAG})': [total_rmse_ar] + rmse_ar
}, index=labels)

# 5) RMSE‐Plots pro Variable
horizons = list(range(1, MAX_HORIZON+1))
# für jede Variable einzeln
for var in endog.columns:
    # RMSE‐Listen initialisieren
    rmse_var_exp = []
    rmse_var_ar  = []
    # für jeden Forecast‐Horizon berechnen
    for h in horizons:
        idx = preds_exp[h].index
        y_true = endog[var].loc[idx]
        y_e    = preds_exp[h][var]
        y_a    = preds_ar[h][var]
        rmse_var_exp.append(np.sqrt(mean_squared_error(y_true, y_e)))
        rmse_var_ar.append(np.sqrt(mean_squared_error(y_true, y_a)))

    # DataFrame für den Plot
    df_plot_var = pd.DataFrame({
        'DFM':    rmse_var_exp,
        f'AR({AR_LAG})':  rmse_var_ar
    }, index=[f'{h}-Step' for h in horizons])

    # Balkendiagramm erstellen
    fig, ax = plt.subplots(figsize=(8,5))
    x     = np.arange(len(df_plot_var.index))
    width = 0.35
    ax.bar(x - width/2, df_plot_var['DFM'],    width, label='DynamicFactor')
    ax.bar(x + width/2, df_plot_var[f'AR({AR_LAG})'],  width, label=f'AR({AR_LAG})')

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot_var.index)
    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel('RMSE')
    ax.set_title(f'RMSE‐Comparison for Variable "{var}"')
    ax.legend(loc= "center right")
    plt.grid(True)
    plt.tight_layout()

    # optional: speichern
    fname = OUTPUT_PATH_PLOTS / f'RMSE_{var}_DFM_vs_AR{AR_LAG}.png'
    fig.savefig(fname)
    plt.show()
