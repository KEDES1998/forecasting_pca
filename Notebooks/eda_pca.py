import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# In[]

project_root = Path().resolve().parent
print(f"Projektroot: {project_root}")

processed_folder = project_root / "data" / "processed"
processed_file = processed_folder / "cleaned_macro_series2.pkl"

df = pd.read_pickle(processed_file)

# In[] Probe PCA

# Abhängige Variablen definieren
dependent_vars = ["gdp_diff", "lrate_diff", "srate", "cpi_diff"]

# Unabhängige Spalten (alle numerischen, außer den abhängigen und 'date_parsed')
X_cols = [col for col in df.columns if col not in dependent_vars + ["date_parsed"]]

# Daten für PCA vorbereiten
X = df[X_cols].dropna()
X_std = StandardScaler().fit_transform(X)

# PCA durchführen
pca = PCA()
pca.fit(X_std)

# Eigenvektoren als DataFrame
eigenvectors = pd.DataFrame(pca.components_, columns=X.columns)
eigenvectors.index = [f"PC{i+1}" for i in range(len(eigenvectors))]

print("Erklärte Varianz je Komponente:")
print(pca.explained_variance_ratio_)

print("\nEigenvektoren (Ladungen):")
print(eigenvectors)

all_eigenvectors = processed_folder / "eigenvectors_full.xlsx"
eigenvectors.to_excel(all_eigenvectors)

all_eigenvectors_pkl = processed_folder / "eigenvectors_full.pkl"
eigenvectors.to_pickle(all_eigenvectors_pkl)


#In[] Scree-Plot
# Scree-Plot: Erklärte Varianz der ersten 10 Hauptkomponenten
explained_variance = pca.explained_variance_ratio_[:10]

plt.figure(figsize=(10, 5))
plt.bar(range(1, 11), explained_variance)
plt.title("Erklärte Varianz der ersten 10 Hauptkomponenten")
plt.xlabel("Hauptkomponenten")
plt.ylabel("Erklärte Varianz")
plt.xticks(range(1, 11))
plt.tight_layout()
plt.show()

# In[] PC1 Zeitverlauf zeichnen

# Wandle den standardisierten Datensatz zurück in DataFrame mit Index
X_pca = pca.transform(X_std)
pc1_series = pd.Series(X_pca[:, 0], index=df.dropna().loc[:, "date_parsed"])

# Plotten der ersten Hauptkomponente
plt.figure(figsize=(10, 4))
plt.plot(pc1_series, label="PC1")
plt.title("Erste Hauptkomponente (PC1) über die Zeit")
plt.xlabel("Datum")
plt.ylabel("PC1-Wert")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# In[] Heat map
# Anzahl der Hauptkomponenten und Variablen anzeigen
num_pcs_to_plot = 4

# Absolute Werte der Ladungen
eigenvectors_abs = eigenvectors.iloc[:num_pcs_to_plot].copy()
eigenvectors_abs = eigenvectors_abs.apply(np.abs)

# Optional: Sortiere Variablen nach ihrer Gesamtbedeutung in PC1–PC4
total_influence = eigenvectors_abs.sum(axis=0)
sorted_vars = total_influence.sort_values(ascending=False).index
eigenvectors_sorted = eigenvectors_abs[sorted_vars]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
cax = ax.imshow(eigenvectors_sorted, cmap="plasma", aspect="auto")

# Achsenbeschriftungen
ax.set_xticks(range(len(eigenvectors_sorted.columns)))
ax.set_xticklabels(eigenvectors_sorted.columns, rotation=90, fontsize=8)
ax.set_yticks(range(num_pcs_to_plot))
ax.set_yticklabels(eigenvectors_sorted.index)

# Farbleiste
fig.colorbar(cax, ax=ax)

plt.title("Heatmap der absoluten Ladungen (PC1–PC4, sortiert nach Einfluss)")
plt.tight_layout()
plt.show()

# In[]

num_pcs_to_show = 4
top_n = 10  # nur die Top-N Variablen pro PC

for i in range(num_pcs_to_show):
    component = eigenvectors.iloc[i]

    # Top-N nach absolutem Wert
    top_loadings = component.abs().sort_values(ascending=False).head(top_n)
    top_vars = component.loc[top_loadings.index]  # originale Werte inkl. Vorzeichen

    # Plot
    plt.figure(figsize=(10, 5))
    top_vars.plot(kind='bar')
    plt.title(f"Top {top_n} Variablen in PC{i + 1}")
    plt.ylabel("Ladung (Loading)")
    plt.xlabel("Variable")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()