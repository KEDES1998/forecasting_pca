import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# In[Setup]
project_root = Path().resolve()
print(f"Projektroot: {project_root}")

processed_folder = project_root / "data" / "processed"
input_excel = processed_folder / "test_train" /"train_splits.xlsx"
output_folder = processed_folder / "pca_outputs"
output_folder.mkdir(exist_ok=True)

sheet_dict = pd.read_excel(input_excel, sheet_name=None)
pca_results = {}
# In[Loop über alle Sheets]
for sheet_name, df in sheet_dict.items():
    print(f"\n Verarbeite Sheet: {sheet_name}")

    exclude_cols = {"year", "month", "quarter", "date", "date_parsed", "ngdp",
                    "gdp_prod", "ngdpos", "pgdp", "gdpoi", "gdpos"}

    relevant_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[relevant_cols].dropna()
    X_std = StandardScaler().fit_transform(X)

    pca = PCA()
    pca.fit(X_std)

    eigenvectors = pd.DataFrame(pca.components_, columns=relevant_cols)
    eigenvectors.index = [f"PC{i + 1}" for i in range(len(eigenvectors))]

    eigenvectors.to_excel(output_folder / f"eigenvectors_{sheet_name}.xlsx")
    eigenvectors.to_pickle(output_folder / f"eigenvectors_{sheet_name}.pkl")

    # Screeplot
    explained_variance = pca.explained_variance_ratio_[:10]
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, 11), explained_variance)
    plt.title(f"Erklärte Varianz der ersten 10 PCs – {sheet_name}")
    plt.xlabel("Hauptkomponenten")
    plt.ylabel("Erklärte Varianz")
    plt.tight_layout()
    plt.show()

    # CDF der erklärten Varianz
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.title(f"Kumulative erklärte Varianz – {sheet_name}")
    plt.xlabel("Hauptkomponenten")
    plt.ylabel("Kumulative erklärte Varianz")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 21), cumulative_variance[:20], marker='o')
    plt.title(f"Kumulative erklärte Varianz (1–20) – {sheet_name}")
    plt.xlabel("Hauptkomponenten")
    plt.ylabel("Kumulative erklärte Varianz")
    plt.xticks(range(1, 21))
    plt.grid(True)
    plt.text(0.95, -0.2,
             f"Kumulative erklärte Varianz der ersten 20 Komponenten: {cumulative_variance[19]:.3f}",
             transform=plt.gca().transAxes, fontsize=10, ha='right')
    plt.tight_layout()
    plt.show()

    # PC1-Zeitreihe (falls Datum dabei ist)
    if "date_parsed" in df.columns:
        X_pca = pca.transform(X_std)
        pc1_series = pd.Series(X_pca[:, 0], index=df.dropna().loc[:, "date_parsed"])
        plt.figure(figsize=(10, 4))
        plt.plot(pc1_series, label="PC1")
        plt.title(f"PC1 über Zeit – {sheet_name}")
        plt.xlabel("Datum")
        plt.ylabel("PC1-Wert")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Top Loadings
    num_pcs_to_show = 3
    top_n = 10
    for i in range(num_pcs_to_show):
        component = eigenvectors.iloc[i]
        top_loadings = component.abs().sort_values(ascending=False).head(top_n)
        top_vars = component.loc[top_loadings.index]
        plt.figure(figsize=(10, 5))
        top_vars.plot(kind='bar')
        plt.title(f"Top {top_n} Variablen in PC{i + 1} – {sheet_name}")
        plt.ylabel("Ladung (Loading)")
        plt.xlabel("Variable")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Heatmap
    num_pcs_to_plot = 3
    eigenvectors_abs = eigenvectors.iloc[:num_pcs_to_plot].copy().apply(np.abs)
    total_influence = eigenvectors_abs.sum(axis=0)
    sorted_vars = total_influence.sort_values(ascending=False).index
    eigenvectors_sorted = eigenvectors_abs[sorted_vars]

    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(eigenvectors_sorted, cmap="plasma", aspect="auto")
    ax.set_xticks(range(len(eigenvectors_sorted.columns)))
    ax.set_xticklabels(eigenvectors_sorted.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(num_pcs_to_plot))
    ax.set_yticklabels(eigenvectors_sorted.index)
    fig.colorbar(cax, ax=ax)
    plt.title(f"Heatmap PC1–PC3 (sorted) – {sheet_name}")
    plt.tight_layout()
    plt.show()

    # Dummy-3D-Plot
    eigenvectors_3d = pd.DataFrame({
        "PC1": np.random.rand(100),
        "PC2": np.random.rand(100),
        "PC3": np.random.rand(100)
    })

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        eigenvectors_3d["PC1"], eigenvectors_3d["PC2"], eigenvectors_3d["PC3"],
        c=eigenvectors_3d["PC1"], cmap="viridis", s=60, alpha=0.8, edgecolors="w", linewidth=0.5
    )
    fig.colorbar(scatter, ax=ax, pad=0.2).set_label("Farbskalierung: PC1", fontsize=12)
    ax.set_title(f"3D-Plot der ersten 3 PCs – {sheet_name}", fontsize=16, weight='bold', pad=20)
    ax.set_xlabel("PC1", fontsize=12, labelpad=10)
    ax.set_ylabel("PC2", fontsize=12, labelpad=10)
    ax.set_zlabel("PC3", fontsize=12, labelpad=10)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = True
        axis.pane.set_facecolor('#f0f0f0')
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.show()

    pca_results[sheet_name] = {
        "eigenvectors": eigenvectors,
        "explained_variance": pca.explained_variance_ratio_,
        "pc1_series": pc1_series if "date_parsed" in df.columns else None,
        "relevant_columns": relevant_cols,
        "df_raw": df,
    }

print("\nAlle Sheets erfolgreich analysiert.")

