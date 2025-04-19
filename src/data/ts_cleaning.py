# In[1]
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




# In[2] Path

project_root = Path().resolve()
print(f"Projektroot: {project_root}")
raw_folder = project_root / "data" / "raw"
raw_data = raw_folder / "Macro_series_FS25.xlsx"
processed_folder = project_root / "data" / "processed"
print(f"Data: {raw_data}")

df = pd.read_excel(raw_data)
df.rename(columns={df.columns[0]: "date"}, inplace=True)

# In[3]

"""
# Creating dictionary from the first two rows (column name and description)
first_two_rows_dict = {col: list(df[col][:2]) for col in df.columns}
output_file = raw_folder / "c_name_c_description.txt"

# Saving the dictionary as a text file
with open(output_file, "w") as f:
    f.write(str(first_two_rows_dict))
"""

# In[4] Beschreibung entfernt
df.drop(index=0, inplace=True)

# In[5]
print(df["date"].unique())
print(f"Type von (Einzel)Einträge von Datum {type(df['date'].iloc[0])}")


# In[6]

df["year"] = df["date"].str.split(" ").str[0].astype(int)  # Extract year as integer
df["quarter"] = df["date"].str.split(" ").str[1]

quarter_to_month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}  # Define the quarter-to-month mapping
df["month"] = df["quarter"].map(quarter_to_month)

df["date_parsed"] = pd.to_datetime(
    df[["year", "month"]].assign(day=1)  # Assign the first day of the month
)

# In[7]

df["date_parsed"] = pd.to_datetime(df["date_parsed"], format="%Y Q%q")
#df['date_parsed'] = pd.PeriodIndex(df['date_parsed'], freq='Q')

# In[8] GDP-Plot

fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs[0].plot(df["date_parsed"], df["gdp"])
axs[0].set_title("GDP")

axs[1].plot(df["date_parsed"], df["cpi"])
axs[1].set_title("CPI")

axs[2].plot(df["date_parsed"], df["srate"])
axs[2].set_title("Short-Term Interest Rate")

axs[3].plot(df["date_parsed"], df["lrate"])
axs[3].set_title("Long-Term Interest Rate")

plt.tight_layout()
plt.show()

# In[9] Statiobarity testing (ADFuller)

def adf_test(series, name=''):
    result = adfuller(series.dropna())
    print(f'{name}: ADF={result[0]:.3f}, p-value={result[1]:.3f}')

for col in ["gdp", "cpi", "srate", "lrate"]:
    adf_test(df[col], name=col)

# In[10]

df_diff = df.copy()

for col in ["gdp", "cpi", "lrate"]:
    df_diff[col + "_diff"] = df[col].diff()


# In[11]

for col in ["gdp_diff", "cpi_diff", "srate", "lrate_diff"]:
    adf_test(df_diff[col], name=col)

# In[12]

for col in ["gdp_diff", "cpi_diff", "srate", "lrate_diff"]:
    series = df_diff[col].dropna()
    stl = STL(series, period=4)  # quartalsweise Daten
    result = stl.fit()

    result.plot()
    plt.show()


# gdp_diff: Only "stronger" seasonal oscillation towards the end -> no seasonal correction needed
# cpi_diff: so clear seasonal pattern visible -> no seasonal correction needed
# srate:    no clear seasonal oscillation visible, only varying oscillation in the timeframe
# lrate_diff: shows pattern of a seasonal pattern -> needs seasonal adjustment


# In[13]

series = df_diff['lrate_diff'].dropna()
stl = STL(series, period=4)
result = stl.fit()
df_diff['lrate_adj'] = series - result.seasonal

# In[14]
df_diff.rename(columns={"gdp_diff": "gdp_adj", "srate": "srate_adj", "cpi_diff": "cpi_adj"}, inplace=True)

"""SUMMARY

CPI : Differantiated

GDP : Differantiated

LRate : Differantiated + Seasonaly adjusted

SRate : -

"""

# In[15]

processed_file = processed_folder / "cleaned_macro_series.xlsx"
processed_file_pkl = processed_folder / "cleaned_macro_series.pkl"


#df_diff.to_excel(processed_file, index=False)
#df_diff.to_pickle(processed_file_pkl)

print(f"Processed DataFrame saved to {processed_folder}")

# In[] Teil Marc, Identifikation Non_stationary

# Exclude Columns
exclude_cols = {"year", "month", "quarter", "date", "date_parsed"}

# Konvertiere alle relevanten Spalten zu numerisch
for col in df.columns:
    if col not in exclude_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Wähle numerische Spalten aus
candidate_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_cols]

# ADF-Test
non_stationary = []

for col in candidate_columns:
    try:
        result = adfuller(df[col].dropna())
        adf = result[0]
        pval = result[1]
        if pval > 0.05:
            non_stationary.append(col)
    except Exception as e:
        print(f"{col}: Fehler beim ADF-Test – {str(e)}")

# Ausgabe
print("\nNicht-stationäre Variablen (p > 0.05):")
print(non_stationary)
# In[] Diff falls non_stationary

diff_cols = {col + "_diff": df[col].diff() for col in non_stationary}
df_diff = pd.concat([df, pd.DataFrame(diff_cols)], axis=1)
# In[] Kontrolle

# ADF-Test Funktion
def adf_test(series, name=''):
    result = adfuller(series.dropna())
    print(f'{name}: ADF={result[0]:.3f}, p-value={result[1]:.3f}')

# ADF-Test auf alle numerischen Spalten in df_diff
for col in df_diff.columns:
    if pd.api.types.is_numeric_dtype(df_diff[col]):
        try:
            adf_test(df_diff[col], name=col)
        except:
            pass

# In[] Delete all that didn't pass the df test
stationary_cols = []

for col in df_diff.columns:
    if pd.api.types.is_numeric_dtype(df_diff[col]):
        try:
            result = adfuller(df_diff[col].dropna())
            pval = result[1]
            if pval < 0.05:
                stationary_cols.append(col)
        except:
            pass

# Füge 'date_parsed' hinzu, falls vorhanden
if 'date_parsed' in df_diff.columns:
    stationary_cols.append('date_parsed')

# Neuer DataFrame mit nur stationären Spalten + date_parsed
df_diff_proper = df_diff[stationary_cols]

# Neue Spaltenreihenfolge definieren
cols_order = ["date_parsed", "cpi_diff", "srate", "lrate_diff", "gdp_diff"]
remaining_cols = [col for col in df_diff_proper.columns if col not in cols_order]
df_diff_proper = df_diff_proper[cols_order + remaining_cols]

# Normalisieren mit Maximum, aber ohne 'month'
for col in df_diff_proper.columns:
    if col != "month" and pd.api.types.is_numeric_dtype(df_diff_proper[col]):
        max_val = df_diff_proper[col].max()
        if max_val != 0 and pd.notna(max_val):
            df_diff_proper[col] = df_diff_proper[col] / max_val

# In[] Neues Excel
df_diff_proper.to_excel("processed_macro_diff_M.xlsx", index=False)
df_diff_proper.to_pickle("processed_macro_diff_M.pkl")

print("Processed DataFrame saved to 'processed_macro_diff_M.xlsx' and 'processed_macro_diff_M.pkl'")

# In[] Graphen, saisonale decomp
for col in df_diff_proper.columns:
    if col != "date_parsed":
        try:
            series = df_diff_proper[[col]].copy()
            series["date_parsed"] = df_diff_proper["date_parsed"]
            series = series.set_index("date_parsed")
            stl = STL(series[col].dropna(), period=4)
            result = stl.fit()
            result.plot()
            plt.suptitle(f'STL Decomposition – {col}', fontsize=14)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Fehler bei Spalte {col}: {e}")


# In[] Probe PCA

# Abhängige Variablen definieren
dependent_vars = ["gdp_diff", "lrate_diff", "srate", "cpi_diff"]

# Unabhängige Spalten (alle numerischen, außer den abhängigen und 'date_parsed')
X_cols = [col for col in df_diff_proper.columns if col not in dependent_vars + ["date_parsed"]]

# Daten für PCA vorbereiten
X = df_diff_proper[X_cols].dropna()
X_std = StandardScaler().fit_transform(X)

# PCA durchführen
pca = PCA()
pca.fit(X_std)

# Eigenvektoren als DataFrame
eigenvectors = pd.DataFrame(pca.components_, columns=X.columns)
eigenvectors.index = [f"PC{i+1}" for i in range(len(eigenvectors))]

# Ausgabe
import matplotlib.pyplot as plt

print("Erklärte Varianz je Komponente:")
print(pca.explained_variance_ratio_)

print("\nEigenvektoren (Ladungen):")
print(eigenvectors)

eigenvectors.to_excel("eigenvectors_full.xlsx")
print("Eigenvektoren gespeichert in 'eigenvectors_full.xlsx'")
eigenvectors.to_pickle("eigenvectors_full.pkl")
print("Eigenvektoren gespeichert in 'eigenvectors_full.pkl'")

#In[] Scree-Plot
# Scree-Plot: Erklärte Varianz pro Komponente
explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(10, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.title("Erklärte Varianz pro Hauptkomponente")
plt.xlabel("Hauptkomponenten")
plt.ylabel("Erklärte Varianz")
plt.xticks(range(1, len(explained_variance) + 1), rotation=90)
plt.tight_layout()
plt.show()

# In[] PC1 Zeitverlauf zeichnen

# Wandle den standardisierten Datensatz zurück in DataFrame mit Index
X_pca = pca.transform(X_std)
pc1_series = pd.Series(X_pca[:, 0], index=df_diff_proper.dropna().loc[:, "date_parsed"])

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

#In[] Heat map
# Beispiel: Zeige die ersten 10 Hauptkomponenten
num_pcs_to_plot = 10
fig, ax = plt.subplots(figsize=(12, 8))

cax = ax.imshow(eigenvectors.iloc[:num_pcs_to_plot], cmap="viridis", aspect="auto")

# Achsenbeschriftungen
ax.set_xticks(range(len(eigenvectors.columns)))
ax.set_xticklabels(eigenvectors.columns, rotation=90, fontsize=8)
ax.set_yticks(range(num_pcs_to_plot))
ax.set_yticklabels(eigenvectors.index[:num_pcs_to_plot])

# Farbleiste
fig.colorbar(cax, ax=ax)

plt.title("Heatmap der PCA-Komponenten (Eigenvektoren)")
plt.tight_layout()
plt.show()