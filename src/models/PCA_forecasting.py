import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from pathlib import Path

# Pfade und Parameter definieren
project_root = Path().resolve().parent.parent
processed_folder = project_root / "data" / "processed"
cleaned_data_path = processed_folder / 'cleaned_data.csv'

# Daten laden
df = pd.read_csv(cleaned_data_path, index_col=0, parse_dates=True)
print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
print("Variablen:", df.columns.tolist())

# Fehlende Werte überprüfen und behandeln
missing_values = df.isnull().sum()
print("\nFehlende Werte pro Spalte:")
print(missing_values)

# Für dieses Beispiel füllen wir fehlende Werte auf
# In einer realen Zeitreihenanalyse könntest du andere Methoden wie Interpolation verwenden
df_filled = df.fillna(df.mean())

# Die zu prognostizierenden Zielvariablen definieren
target_vars = ["inflation", "g_gdpos"]

# Überprüfen, ob diese Variablen im Datensatz vorhanden sind
for var in target_vars:
    if var not in df_filled.columns:
        raise ValueError(f"Zielvariable '{var}' nicht im Datensatz gefunden!")

# Feature-Set erstellen (alle Spalten außer den Zielvariablen)
feature_vars = [col for col in df_filled.columns if col not in target_vars]

# Expanding Window Modell mit PCA
# ==============================

# Parameter
initial_train_periods = 60  # Erste 60 Quartale als Trainingsdaten
forecast_horizon = 1  # Ein Schritt voraus prognostizieren (kann angepasst werden)
n_components = 0.95  # PCA-Komponenten, die 95% der Varianz erklären

# Ergebnisse speichern
results = {}
for target in target_vars:
    results[target] = {
        'true_values': [],
        'predictions': [],
        'r2_scores': [],
        'mse_scores': [],
        'mae_scores': [],
        'test_indices': []
    }

# DataFrame für PCA-Features vorbereiten
X = df_filled[feature_vars]
y_dict = {target: df_filled[target] for target in target_vars}

# Scaler für Features vorbereiten
scaler = StandardScaler()

# Expanding Window Loop
for i in range(initial_train_periods, len(df_filled) - forecast_horizon + 1):
    # Trainings- und Testdaten definieren
    X_train = X.iloc[:i]
    X_test = X.iloc[i:i + forecast_horizon]

    # Skalieren der Trainingsdaten
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA auf Trainingsdaten anwenden
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

        # Modell trainieren
        model = LinearRegression()
        model.fit(X_train_pca, y_train)

        # Vorhersage
        y_pred = model.predict(X_test_pca)

        # Ergebnisse speichern
        results[target]['true_values'].extend(y_test.values)
        results[target]['predictions'].extend(y_pred)
        results[target]['test_indices'].extend(y_test.index)

        # Metriken berechnen
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results[target]['r2_scores'].append(r2)
        results[target]['mse_scores'].append(mse)
        results[target]['mae_scores'].append(mae)

# Ergebnisse visualisieren
# =======================

# Metrik-Evolution über die Zeit darstellen
plt.figure(figsize=(18, 10))

for i, target in enumerate(target_vars):
    # Indizes für x-Achse - die Testperioden
    test_periods = range(initial_train_periods, len(df_filled) - forecast_horizon + 1)

    # Subplot für R²
    plt.subplot(2, 3, 1 + i * 3)
    plt.plot(test_periods, results[target]['r2_scores'], marker='o')
    plt.axhline(y=np.mean(results[target]['r2_scores']), color='r', linestyle='--',
                label=f'Mittelwert: {np.mean(results[target]["r2_scores"]):.3f}')
    plt.title(f'R² Evolution - {target}')
    plt.xlabel('Testperiode')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)

    # Subplot für MSE
    plt.subplot(2, 3, 2 + i * 3)
    plt.plot(test_periods, results[target]['mse_scores'], marker='o')
    plt.axhline(y=np.mean(results[target]['mse_scores']), color='r', linestyle='--',
                label=f'Mittelwert: {np.mean(results[target]["mse_scores"]):.3f}')
    plt.title(f'MSE Evolution - {target}')
    plt.xlabel('Testperiode')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    # Subplot für MAE
    plt.subplot(2, 3, 3 + i * 3)
    plt.plot(test_periods, results[target]['mae_scores'], marker='o')
    plt.axhline(y=np.mean(results[target]['mae_scores']), color='r', linestyle='--',
                label=f'Mittelwert: {np.mean(results[target]["mae_scores"]):.3f}')
    plt.title(f'MAE Evolution - {target}')
    plt.xlabel('Testperiode')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.suptitle('Evolution der Modellmetriken über alle Testperioden', fontsize=16)
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
    rmse = np.sqrt(np.mean(results[target]['mse_scores']))
    mae = np.mean(results[target]['mae_scores'])
    r2_mean = np.mean(results[target]['r2_scores'])

    plt.annotate(f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2_mean:.4f}',
                 xy=(0.05, 0.9), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()

# Feature-Importance basierend auf den PCA-Komponenten analysieren
# ===============================================================

# PCA auf den gesamten Datensatz anwenden, um Feature-Importance zu berechnen
X_scaled = scaler.fit_transform(X)
final_pca = PCA(n_components=n_components)
final_pca.fit(X_scaled)

# Varianzanteil der einzelnen Komponenten
explained_variance = final_pca.explained_variance_ratio_

# PCA-Komponenten und Loadings visualisieren
plt.figure(figsize=(14, 8))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.8)
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', color='red')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Varianzgrenze')
plt.xlabel('Hauptkomponenten')
plt.ylabel('Erklärte Varianz')
plt.title('Scree Plot: Erklärte Varianz durch Hauptkomponenten')
plt.legend()
plt.grid(True)
plt.show()

# Korrelationsmatrix zwischen Features und den wichtigsten PCA-Komponenten
loadings = final_pca.components_
n_top_components = min(5, loadings.shape[0])  # Max. 5 Komponenten anzeigen

# Loadings als DataFrame für leichtere Handhabung
loadings_df = pd.DataFrame(
    loadings[:n_top_components].T,
    index=feature_vars,
    columns=[f'PC{i + 1} ({var:.2%})' for i, var in enumerate(explained_variance[:n_top_components])]
)

# Heatmap der Loadings
plt.figure(figsize=(12, 10))
sns.heatmap(loadings_df, annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title('Feature-Loadings der wichtigsten PCA-Komponenten')
plt.tight_layout()
plt.show()

# Abschließende Statistiken und Zusammenfassung
# ============================================
print("\n===== Zusammenfassung der Modellperformance =====")
for target in target_vars:
    print(f"\nZielvariable: {target}")
    print(f"Durchschnittliches R²: {np.mean(results[target]['r2_scores']):.4f}")
    print(f"Durchschnittliches MSE: {np.mean(results[target]['mse_scores']):.4f}")
    print(f"Durchschnittliches MAE: {np.mean(results[target]['mae_scores']):.4f}")
    print(f"RMSE: {np.sqrt(np.mean(results[target]['mse_scores'])):.4f}")

# Interpretation der wichtigsten PCA-Komponenten
print("\n===== Interpretation der PCA-Komponenten =====")
for i in range(min(3, loadings.shape[0])):  # Top 3 Komponenten
    print(f"\nPC{i + 1} (erklärt {explained_variance[i]:.2%} der Varianz):")

    # Sortiere Features nach absolutem Loading-Wert
    sorted_loadings = sorted(zip(feature_vars, loadings[i]), key=lambda x: abs(x[1]), reverse=True)

    # Top 5 einflussreichste Features
    for feature, loading in sorted_loadings[:5]:
        print(f"  {feature}: {loading:.4f}")