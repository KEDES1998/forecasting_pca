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

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*SDataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.*"
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
OUTPUT_PATH_PLOTS = project_root / "results" / "figures" / "forecast_plots" / "PCA"

####################
#### PARAMETER #####
####################

max_lag = 2 # Lags for AR
initial_train_periods = 60 - max_lag
forecast_horizon = 1
n_components = 0.90  # PCA-Komponenten, that explain 95% of variance
target_vars = ["inflation", "g_gdpos", "srate", "lrate"]

AR_ORDER = 1
MAX_HORIZON = 8

# In[Data loading]
df = pd.read_csv(cleaned_data_path, index_col=0, parse_dates=True)
print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
print("Variablen:", df.columns.tolist())

# In[Missing Data handling -> Backfill commonly used in timeseries]
df_filled = df.fillna(method='ffill').fillna(method='bfill')

# Feature-Set erstellen (alle Spalten außer den Zielvariablen)
feature_vars = [col for col in df_filled.columns if col not in target_vars]
df_with_lags = df_filled.copy()

# In[Creating Lags]
for target in target_vars:
    for lag in range(1, max_lag + 1):
        lag_name = f"{target}_lag{lag}"
        df_with_lags[lag_name] = df_filled[target].shift(lag)
        feature_vars.append(lag_name)  # Lag-variables for features

# Removing first lagged rows
df_with_lags = df_with_lags.iloc[max_lag:]

#######################################
#### EXPANDING WINDOW PCA FORCAST #####
#######################################

# Saving results
results = {}
for target in target_vars:
    results[target] = {
        'true_values': [],
        'predictions': [],
        'rmse_scores': [],
        'test_indices': [],
        'coefficients': [],
        'pca_loadings': []
    }

# DF for pca features and target vars
X = df_with_lags[feature_vars]
y_dict = {target: df_with_lags[target] for target in target_vars}

# Expanding Window Loop
for i in range(initial_train_periods, len(df_with_lags) - forecast_horizon + 1):
    # Trainings- und Testdaten definieren
    X_train = X.iloc[:i]
    X_test = X.iloc[i:i + forecast_horizon]

    # scale
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA ON TRAINIGDATA !!!
    pca = PCA(n_components=n_components)
    pca.fit(X_train_scaled)

    # PCA-transformed data
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # No of used pca's adn variance explained
    if i == initial_train_periods:
        print(f"\nAnzahl PCA-Komponenten: {pca.n_components_}")
        print(f"Erklärte Varianz: {sum(pca.explained_variance_ratio_):.4f}")

    # modell fpr each targetr var
    for target in target_vars:
        y_train = y_dict[target].iloc[:i]
        y_test = y_dict[target].iloc[i:i + forecast_horizon]

        # print("XXXXXXXXXXXXXXXXXXX INPUT AUTOREG -> Y_TRAIN " + str(len(y_train)))

        # (PCA) AutoReg Modell
        model = AutoReg(y_train, lags=max_lag, exog=X_train_pca)
        fit_model = model.fit()

        # Prediction with exogenous variables
        y_pred = fit_model.forecast(steps=forecast_horizon, exog=X_test_pca)

        # Saving results
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

####################
###### PLOTS #######
####################

################ RMSE Evolution ##################

for i, target in enumerate(target_vars):
    fig = plt.figure(figsize=(10, 4))
    # Indizes x-axis testperiod
    test_periods = range(initial_train_periods, len(df_with_lags) - forecast_horizon + 1)

    plt.plot(test_periods, results[target]['rmse_scores'], marker='o')
    plt.axhline(y= np.mean(results[target]['rmse_scores']), color='r', linestyle='--',
                label=f'Mittelwert: {np.mean(results[target]["rmse_scores"]):.3f}')
    plt.title(f'RMSE Evolution (Testperiod)- {target}')
    plt.xlabel('Test period')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

    fig.savefig(OUTPUT_PATH_PLOTS / f"{target}-RMSE_evo.png")
    plt.tight_layout()
    plt.show()

################# Model Summary ##################
print("\n===== Zusammenfassung der Modellperformance =====")
for target in target_vars:
    print(f"\nZielvariable: {target}")
    print("\n\n")
    print(f"Durchschnittliches RMSE: {np.mean(results[target]['rmse_scores']):.4f}")


# Analysis of the most important features
print("\n===== Wichtigste Features für die Prognose =====")
for target in target_vars:
    print(f"\nFür {target}:")

    # Last modell analysis
    last_coefs = results[target]['coefficients'][-1]
    last_loadings = results[target]['pca_loadings'][-1]

    # calculation importance of each feature
    feature_importance = np.zeros(len(feature_vars))

    for i, coef in enumerate(last_coefs):
        if i < len(last_loadings):  # Sicherstellen, dass wir nur gültige Indizes verwenden
            feature_importance += abs(coef) * abs(last_loadings[i])

    # sort after importance
    sorted_features = [(feature, importance)
                       for feature, importance in zip(feature_vars, feature_importance)]
    sorted_features.sort(key=lambda x: x[1], reverse=True)

    # Top 10 Features
    for feature, importance in sorted_features[:10]:
        print(f"  {feature}: {importance:.4f}")

#########################
######### AR() ##########
#########################

preds_pca = {h: {t: [] for t in target_vars}
             for h in range(1, MAX_HORIZON+1)}
preds_ar = {h: {t: [] for t in target_vars}
             for h in range(1, MAX_HORIZON+1)}


# AR(1) Forecasts PREP
ar_forecasts = {}

for target in target_vars:
    y_full = df_with_lags[target].values
    ar_predictions = []

    for i in range(initial_train_periods, len(df_with_lags) - forecast_horizon + 1):
        y_train = y_full[:i]
        model_ar = AutoReg(y_train, lags=AR_ORDER, old_names=False)
        fit_ar = model_ar.fit()
        y_pred_ar = fit_ar.forecast(steps=forecast_horizon)
        ar_predictions.extend(y_pred_ar)

    ar_forecasts[target] = ar_predictions

################ Prediction Plot ##################

for i, target in enumerate(target_vars):

    fig = plt.figure(figsize=(10, 4))

    time_idx = results[target]['test_indices']

    plt.plot(time_idx, results[target]['true_values'], color = "black", label='time series', linestyle='-')
    plt.plot(time_idx, results[target]['predictions'], 'r--', label='PCA AR')
    plt.plot(time_idx, ar_forecasts[target], color = "blue", label=f'AR {AR_ORDER}', linestyle='--')

    plt.title(f'Target Variable - {target}')
    plt.xlabel('date')
    plt.ylabel(target)
    plt.legend()
    plt.grid(True)

    rmse = np.sqrt(np.mean(results[target]['rmse_scores']))
    rmse_ar = root_mean_squared_error(results[target]['true_values'], ar_forecasts[target])

    plt.annotate(f'RMSE PCA AR: {rmse:.4f}', xy=(0.05, 0.88), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    plt.annotate(f'RMSE AR {AR_ORDER}: {rmse_ar:.4f}',
                xy=(0.05, 0.80), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH_PLOTS / f"PCA_forecast_{target}_comparison_ar{AR_ORDER}.png", dpi=300)
    plt.show()

##############################################################
### RMSE Horizon Comparison Plot (PCA-AR vs AR) ###
##############################################################

# We need to generate forecasts for multiple horizons first
# For both PCA-AR and AR models

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height * 1.02,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# Dictionary to store RMSE values for different horizons
rmse_by_horizon = {
    'PCA-AR': {t: [] for t in target_vars},
    'AR': {t: [] for t in target_vars}
}

# Loop through each horizon
for horizon in range(1, MAX_HORIZON + 1):
    print(f"Computing forecasts for horizon h={horizon}...")

    # For each target variable
    for target in target_vars:
        y_full = df_with_lags[target].values
        pca_predictions = []
        ar_predictions = []
        true_values = []

        # Use expanding window approach
        for i in range(initial_train_periods, len(df_with_lags) - horizon + 1):
            # Training data up to point i
            X_train = X.iloc[:i]
            y_train = y_full[:i]

            # Test point is i + horizon - 1
            if i + horizon <= len(df_with_lags):
                true_values.append(y_full[i + horizon - 1])

                # PCA-AR model
                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)

                # Apply PCA
                pca = PCA(n_components=n_components)
                X_train_pca = pca.fit_transform(X_train_scaled)

                # Fit AR model with PCA components
                model_pca = AutoReg(y_train, lags=max_lag, exog=X_train_pca)
                fit_pca = model_pca.fit()

                # For multi-step forecasts, we need to forecast step by step
                # For simplicity, we'll use a direct forecasting approach here
                # This is a simplification - in practice, you might want recursive forecasting
                if horizon == 1:
                    # One-step forecast - use the last X_test directly
                    X_test = X.iloc[i:i + 1]
                    X_test_scaled = scaler.transform(X_test)
                    X_test_pca = pca.transform(X_test_scaled)
                    y_pred_pca = fit_pca.forecast(steps=1, exog=X_test_pca)
                else:
                    # Direct h-step forecast - train model to predict h steps ahead
                    # This is simplified - in a full implementation you'd use a more sophisticated approach
                    y_train_h = y_full[:(i - horizon + 1)]  # Shift target to create direct h-step predictions
                    if len(y_train_h) > max_lag + 5:  # Ensure enough data
                        model_pca_h = AutoReg(y_train_h, lags=max_lag, exog=X_train_pca[:len(y_train_h)])
                        fit_pca_h = model_pca_h.fit()
                        X_test = X.iloc[i:i + 1]
                        X_test_scaled = scaler.transform(X_test)
                        X_test_pca = pca.transform(X_test_scaled)
                        y_pred_pca = fit_pca_h.forecast(steps=1, exog=X_test_pca)
                    else:
                        # Fall back to a simpler approach if not enough data
                        y_pred_pca = [y_train[-1]]  # Use last value as prediction

                pca_predictions.append(y_pred_pca[0])

                # Simple AR model
                model_ar = AutoReg(y_train, lags=AR_ORDER, old_names=False)
                fit_ar = model_ar.fit()

                if horizon == 1:
                    y_pred_ar = fit_ar.forecast(steps=1)
                else:
                    # Simplified multi-step forecasting for AR model
                    y_pred_ar = fit_ar.forecast(steps=horizon)

                ar_predictions.append(y_pred_ar[-1])  # Take the last prediction (h-step ahead)

        # Calculate RMSE for this horizon and target
        if true_values and pca_predictions and ar_predictions:
            rmse_pca = root_mean_squared_error(true_values, pca_predictions)
            rmse_ar = root_mean_squared_error(true_values, ar_predictions)

            rmse_by_horizon['PCA-AR'][target].append(rmse_pca)
            rmse_by_horizon['AR'][target].append(rmse_ar)

# Now create the bar chart comparison for each target variable
for target in target_vars:
    fig = plt.figure(figsize=(12, 6))
    bar_width = 0.35
    horizons = list(range(1, MAX_HORIZON + 1))
    x = np.arange(len(horizons))

    # Create bars
    bars1 = plt.bar(x - bar_width / 2, rmse_by_horizon['PCA-AR'][target],
                    bar_width, label='PCA-AR', color='darkred', alpha=0.8)
    bars2 = plt.bar(x + bar_width / 2, rmse_by_horizon['AR'][target],
                    bar_width, label=f'AR({AR_ORDER})', color='darkblue', alpha=0.8)

    # Add improvement percentages
    for i, h in enumerate(horizons):
        pca_rmse = rmse_by_horizon['PCA-AR'][target][i]
        ar_rmse = rmse_by_horizon['AR'][target][i]
        improvement = ((ar_rmse - pca_rmse) / ar_rmse) * 100

        # Only add text if significant improvement
        if abs(improvement) > 1:
            y_pos = min(pca_rmse, ar_rmse) - 0.03 * max(pca_rmse, ar_rmse)
            color = 'green' if improvement > 0 else 'red'
            label = f"{improvement:.1f}%" if improvement > 0 else f"{-improvement:.1f}%"

            plt.text(i, y_pos, label, ha='center', color=color, fontweight='bold', fontsize=8)

    add_labels(bars1)
    add_labels(bars2)

    # Labels and title
    plt.xlabel('Forecast Horizon (h)')
    plt.ylabel('RMSE')
    plt.title(f'RMSE by Forecast Horizon: {target}')
    plt.xticks(x, horizons)
    plt.legend(loc='upper left')

    # Add a horizontal grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH_PLOTS / f"rmse_horizon_comparison_{target}.png", dpi=300)
    plt.show()

# Create a summary table of RMSE values across horizons
print("\n===== RMSE by Horizon Comparison Summary =====")
for target in target_vars:
    print(f"\nTarget: {target}")
    print(f"{'Horizon':<8} {'PCA-AR':<10} {'AR(' + str(AR_ORDER) + ')':<10} {'Improvement':<10}")
    print("-" * 45)

    for h in range(1, MAX_HORIZON + 1):
        idx = h - 1
        if idx < len(rmse_by_horizon['PCA-AR'][target]) and idx < len(rmse_by_horizon['AR'][target]):
            pca_rmse = rmse_by_horizon['PCA-AR'][target][idx]
            ar_rmse = rmse_by_horizon['AR'][target][idx]
            improvement = ((ar_rmse - pca_rmse) / ar_rmse) * 100

            improvement_str = f"{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
            print(f"h = {h:<5} {pca_rmse:.4f}    {ar_rmse:.4f}    {improvement_str}")
        else:
            print(f"h = {h:<5} No data available")

print("\nPositive improvement percentage means PCA-AR outperforms AR model")