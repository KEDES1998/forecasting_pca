import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
# In[Function]

# In[Paths]

project_root = Path().resolve().parent
print(f"Projektroot: {project_root}")

processed_folder = project_root / "data" / "processed"
processed_file = processed_folder / "cleaned_macro_series2.pkl"

df = pd.read_pickle(processed_file)


# In[Column selection/exclusion]

exclude_cols = {"year", "month", "quarter", "date", "date_parsed", "ngdp",
                "gdp_prod", "ngdpos", "ngdp", "pgdp", "gdpoi", "gdpos"}

relevant_cols = [col for col in df.columns if col not in exclude_cols]


# In[Probe PCA]

is_relevant_col = lambda col: col in relevant_cols

X_cols = [col for col in df.columns if is_relevant_col(col)]

# Daten für PCA vorbereiten
X = df[X_cols]
X_std = StandardScaler().fit_transform(X)

# PCA durchführen
pca = PCA()
pca.fit(X_std)

# Eigenvektoren als DataFrame
eigenvectors = pd.DataFrame(pca.components_, columns=X.columns)
eigenvectors.index = [f"PC{i + 1}" for i in range(len(eigenvectors))]

print("Erklärte Varianz je Komponente:")
print(pca.explained_variance_ratio_)

print("\nEigenvektoren (Ladungen):")
print(eigenvectors)

all_eigenvectors = processed_folder / "eigenvectors_full.xlsx"
eigenvectors.to_excel(all_eigenvectors)

all_eigenvectors_pkl = processed_folder / "eigenvectors_full.pkl"
eigenvectors.to_pickle(all_eigenvectors_pkl)

# In[Screeplot]

explained_variance = pca.explained_variance_ratio_[:10]

plt.figure(figsize=(10, 5))
plt.bar(range(1, 11), explained_variance)
plt.title("Erklärte Varianz der ersten 10 Hauptkomponenten")
plt.xlabel("Hauptkomponenten")
plt.ylabel("Erklärte Varianz")
plt.xticks(range(1, 11))
plt.tight_layout()
plt.show()

# Plot CDF of explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)


# In[Eigenvalue CDF's]

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title("Kumulative erklärte Varianz der Hauptkomponenten")
plt.xlabel("Hauptkomponenten")
plt.ylabel("Kumulative erklärte Varianz")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), cumulative_variance[:20], marker='o')
plt.title("Kumulative erklärte Varianz der ersten 20 Hauptkomponenten")
plt.xlabel("Hauptkomponenten")
plt.ylabel("Kumulative erklärte Varianz")
plt.xticks(range(1, 21))
plt.grid(True)

plt.text(0.95, -0.2, f"Kumulative erklärte Varianz der ersten 20 Komponenten: {cumulative_variance[19]:.3f}",
         transform=plt.gca().transAxes, fontsize=10, ha='right')

plt.tight_layout()
plt.show()

# In[PC1 Zeitverlauf]

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

# In[Plot loadings]

num_pcs_to_show = 3
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

# In[Heat map]

num_pcs_to_plot = 3

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


# In[3D PCA plot]
eigenvectors = pd.DataFrame({
    "PC1": np.random.rand(100),
    "PC2": np.random.rand(100),
    "PC3": np.random.rand(100)
})

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Daten aus den Eigenvektoren abrufen
pc1 = eigenvectors.loc[:, "PC1"]
pc2 = eigenvectors.loc[:, "PC2"]
pc3 = eigenvectors.loc[:, "PC3"]

# Farbwert basierend auf einer Komponente (z. B. PC1)
colors = pc1

# Scatterplot mit Farbgradient
scatter = ax.scatter(
    pc1, pc2, pc3,
    c=colors, cmap="viridis", s=60, alpha=0.8, edgecolors="w", linewidth=0.5
)

# Farbskala hinzufügen
cbar = fig.colorbar(scatter, ax=ax, pad=0.2)
cbar.set_label("Farbskalierung: PC1", fontsize=12)

# Achsentitel mit verbessertem Textstil
ax.set_title("3D-Scatterplot der ersten 3 Hauptkomponenten", fontsize=16, weight='bold', pad=20)
ax.set_xlabel("PC1", fontsize=12, labelpad=10)
ax.set_ylabel("PC2", fontsize=12, labelpad=10)
ax.set_zlabel("PC3", fontsize=12, labelpad=10)

# Anpassung des Hintergrunddesigns
ax.xaxis.pane.fill = True
ax.yaxis.pane.fill = True
ax.zaxis.pane.fill = True
ax.xaxis.pane.set_facecolor('#f0f0f0')
ax.yaxis.pane.set_facecolor('#f0f0f0')
ax.zaxis.pane.set_facecolor('#f0f0f0')

# Gitter und Transparenz für angenehme Lesbarkeit
ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

plt.show()

# In[Schleife über alle Train-Excels]

