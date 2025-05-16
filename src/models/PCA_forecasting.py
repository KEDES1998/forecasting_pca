import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import root_mean_squared_error
from pathlib import Path
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# In[Warning Surpression]

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*Series\.__getitem__ treating keys as positions is deprecated.*"
)

# 2) UserWarnings from statsmodels about unsupported Index types
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Only PeriodIndexes, DatetimeIndexes with a frequency set.*"
)

# 3) UndefinedMetricWarning from scikit-learn (R² with < 2 samples)
warnings.filterwarnings(
    "ignore",
    category=UndefinedMetricWarning
)

warnings.filterwarnings(
    "ignore",
    category=ValueWarning,
    message=r"No frequency information was provided, so inferred frequency QS-OCT will be used.*"
)

# In[# Pfade und Parameter definieren]

project_root = Path().resolve().parent.parent
processed_folder = project_root / "data" / "processed"
cleaned_data_path = processed_folder / 'cleaned_data.csv'

####################
#### PARAMETER #####
####################

max_lag = 2 # Lags for AR
initial_train_periods = 60 - max_lag
forecast_horizon = 1
n_components = 0.90  # PCA-Komponenten, that explain 95% of variance
target_vars = ["inflation", "g_gdpos"]

# Daten laden
df = pd.read_csv(cleaned_data_path, index_col=0, parse_dates=True)
print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
print("Variablen:", df.columns.tolist())

# In[Missing Data handling -> Backfill commonly used in timeseries]
df_filled = df.fillna(method='ffill').fillna(method='bfill')

# Feature-Set erstellen (alle Spalten außer den Zielvariablen)
feature_vars = [col for col in df_filled.columns if col not in target_vars]

df_with_lags = df_filled.copy()

#In[Creating Lags]
for target in target_vars:
    for lag in range(1, max_lag + 1):
        lag_name = f"{target}_lag{lag}"
        df_with_lags[lag_name] = df_filled[target].shift(lag)
        feature_vars.append(lag_name)  # Lag-variables for features

# Removing first lagged rows
df_with_lags = df_with_lags.iloc[max_lag:]

# Expanding Window with PCA and AR
# ============================================

# Saving results
results = {}
for target in target_vars:
    results[target] = {
        'true_values': [],
        'predictions': [],
        'rmse_scores': [],
        'test_indices': [],
        'coefficients': [],  # Für jede Periode die Koeffizienten speichern
        'pca_loadings': []  # Für jede Periode die PCA-Loadings speichern
    }

# DataFrame für PCA-Features und Zielvariablen vorbereiten
X = df_with_lags[feature_vars]
y_dict = {target: df_with_lags[target] for target in target_vars}

# Expanding Window Loop
for i in range(initial_train_periods, len(df_with_lags) - forecast_horizon + 1):
    # Trainings- und Testdaten definieren
    X_train = X.iloc[:i]
    X_test = X.iloc[i:i + forecast_horizon]

    # Skalieren der Trainingsdaten
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA auf Trainingsdaten anwenden (ohne Information aus Testdaten!)
    pca = PCA(n_components=n_components)
    pca.fit(X_train_scaled)

    # PCA-transformierte Daten
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Nummer der verwendeten Komponenten und erklärte Varianz ausgeben
    if i == initial_train_periods:
        print(f"\nAnzahl PCA-Komponenten: {pca.n_components_}")
        print(f"Erklärte Varianz: {sum(pca.explained_variance_ratio_):.4f}")

    # Für jede Zielvariable ein Modell trainieren
    for target in target_vars:
        y_train = y_dict[target].iloc[:i]
        y_test = y_dict[target].iloc[i:i + forecast_horizon]

        # print("XXXXXXXXXXXXXXXXXXX IMPUT AUTOREG -> Y_TRAIN " + str(len(y_train)))

        # AutoReg Modell
        model = AutoReg(y_train, lags=max_lag, exog=X_train_pca)
        fit_model = model.fit()

        # Vorhersage mit exogenen Variablen
        y_pred = fit_model.forecast(steps=forecast_horizon, exog=X_test_pca)

        # Ergebnisse speichern
        results[target]['true_values'].extend(y_test.values)
        results[target]['predictions'].extend(y_pred)
        results[target]['test_indices'].extend(y_test.index)

        # RMSE calculation
        rmse = root_mean_squared_error(y_test, y_pred)
        results[target]['rmse_scores'].append(rmse)


        # Coef and Pca loadings saving for later prediction
        ar_coefs = fit_model.params[1:max_lag + 1]  # AR-Coeff
        exog_coefs = fit_model.params[max_lag + 1:]  # exogene (PCA) coeff
        results[target]['coefficients'].append(exog_coefs)
        results[target]['pca_loadings'].append(pca.components_)

        # Regressions-equation at the moment
        if i == initial_train_periods or i == len(df_with_lags) - forecast_horizon:
            period_label = "Erstes Modell" if i == initial_train_periods else "Letztes Modell"
            print(f"\n=== {period_label}: Regressionsgleichung für {target} ===")
            equation = f"{target} = {fit_model.params[0]:.6f}"  # Konstante

            # AR-terms
            for j, coef in enumerate(ar_coefs):
                equation += f" + ({coef:.6f} × {target}_lag{j + 1})"

            # PCA-temrs
            for j, coef in enumerate(exog_coefs):
                equation += f" + ({coef:.6f} × PC{j + 1})"

            print(equation)

            # Analysis of pca components
            n_top_components = min(3, pca.n_components_)
            for j in range(n_top_components):
                print(f"\nPC{j + 1} (erklärt {pca.explained_variance_ratio_[j]:.2%} der Varianz):")
                # Sortiere Features nach absolutem Loading-Wert
                sorted_loadings = sorted(zip(feature_vars, pca.components_[j]),
                                         key=lambda x: abs(x[1]), reverse=True)
                # Top 5 feautures
                for feature, loading in sorted_loadings[:5]:
                    print(f"  {feature}: {loading:.4f}")

# Result visualisation
# =======================

# Metrik-Evolution über die Zeit darstellen
plt.figure(figsize=(18, 10))

for i, target in enumerate(target_vars):
    # Indizes für x-Achse - die Testperioden
    test_periods = range(initial_train_periods, len(df_with_lags) - forecast_horizon + 1)

    # Subplot für MSE
    plt.subplot(2, 3, 2 + i * 3)
    plt.plot(test_periods, results[target]['rmse_scores'], marker='o')
    plt.axhline(y= np.mean(results[target]['rmse_scores']), color='r', linestyle='--',
                label=f'Mittelwert: {np.mean(results[target]["rmse_scores"]):.3f}')
    plt.title(f'RMSE Evolution - {target}')
    plt.xlabel('Test period')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.suptitle('Evolution of RMSE for testperiod', fontsize=16)
plt.subplots_adjust(top=0.9)
plt.show()

# Tatsächliche vs. prognostizierte Werte visualisieren
plt.figure(figsize=(15, 10))

for i, target in enumerate(target_vars):
    plt.subplot(2, 1, i + 1)

    # Zeitindex für den Plot erstellen
    time_idx = results[target]['test_indices']

    # Tatsächliche und prognostizierte Werte plotten
    plt.plot(time_idx, results[target]['true_values'], 'b-', label='Tatsächliche Werte')
    plt.plot(time_idx, results[target]['predictions'], 'r--', label='Prognosen')

    plt.title(f'Prognose vs. tatsächliche Werte - {target}')
    plt.xlabel('Datum')
    plt.ylabel(target)
    plt.legend()
    plt.grid(True)

    # RMSE und MAE im Plot anzeigen
    rmse = np.sqrt(np.mean(results[target]['rmse_scores']))

    plt.annotate(f'RMSE: {rmse:.4f}',
                 xy=(0.05, 0.9), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()


# Abschließende Statistiken und Zusammenfassung
print("\n===== Zusammenfassung der Modellperformance =====")
for target in target_vars:
    print(f"\nZielvariable: {target}")
    print("\n\n")
    print(f"Durchschnittliches MSE: {np.sqrt(np.mean(results[target]['rmse_scores'])):.4f}")


# Analyse der wichtigsten Features für die Prognose
print("\n===== Wichtigste Features für die Prognose =====")
for target in target_vars:
    print(f"\nFür {target}:")

    # Letztes Modell analysieren
    last_coefs = results[target]['coefficients'][-1]
    last_loadings = results[target]['pca_loadings'][-1]

    # Gewicht jedes Features berechnen (Kombination von PCA-Loadings und Koeffizienten)
    feature_importance = np.zeros(len(feature_vars))

    for i, coef in enumerate(last_coefs):
        if i < len(last_loadings):  # Sicherstellen, dass wir nur gültige Indizes verwenden
            feature_importance += abs(coef) * abs(last_loadings[i])

    # Features nach Wichtigkeit sortieren
    sorted_features = [(feature, importance)
                       for feature, importance in zip(feature_vars, feature_importance)]
    sorted_features.sort(key=lambda x: x[1], reverse=True)

    # Top 10 wichtigste Features ausgeben
    for feature, importance in sorted_features[:10]:
        print(f"  {feature}: {importance:.4f}")