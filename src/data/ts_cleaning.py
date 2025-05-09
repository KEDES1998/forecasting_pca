# In[1]
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings(
    "ignore",
    category=pd.errors.PerformanceWarning
)


# In[Path and Parameters]

#project_root = Path().resolve()
project_root = Path().resolve().parent.parent
print(f"Projektroot: {project_root}")
raw_folder = project_root / "data" / "raw"
raw_data = raw_folder / "Macro_series_FS25.xlsx"
processed_folder = project_root / "data" / "processed"
print(f"Data: {raw_data}")

dep_vars = ["g_gdpos", "inflation", "srate", "lrate"]
log_vars = ["gdpos", "cpi", "pop"]
period = 4

# In[Loading Data]

df = pd.read_excel(raw_data)
df.rename(columns={df.columns[0]: "date"}, inplace=True)


# In[Column Renaming and dropping multicollinear gdp's]
df.drop(index=0, inplace=True)
df.drop(columns=["ngdp","gdp_prod", "gdp", "ngdpos", "pgdp", "gdpoi"], inplace=True)


# In[Date Parsing]
print(df["date"].unique())
print(f"Type von (Einzel)Einträge von Datum {type(df['date'].iloc[0])}")


df["year"]    = df["date"].str.split(" ").str[0].astype(int)
df["quarter"] = df["date"].str.split(" ").str[1]
quarter_to_month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
df["month"]   = df["quarter"].map(quarter_to_month)


df["date_parsed"] = pd.to_datetime(
    df[["year", "month"]].assign(day=1)
)

# 7) Datums-Index setzen und Hilfsspalten entfernen
df = df.set_index("date_parsed").drop(columns=["date", "year", "quarter", "month"])

# In[Seasonal adjusment]

df_sa = pd.DataFrame(index=df.index)

for col in df.columns:
    series = pd.to_numeric(df[col], errors='coerce')
    # STL auf non-NA-Teilserie anwenden
    stl = STL(series.dropna(), period=period, robust=True)
    res = stl.fit()
    # Saisonale Komponente abziehen → saisonbereinigte Serie
    df_sa[col] = series - res.seasonal
    print(f"{col}: saisonal bereinigt")

# Ersetze dein Original-DF durch das bereinigte

df = df_sa

# In[ Growthrates: gdpos, cpi, pop]

df["gdpos"] = pd.to_numeric(df["gdpos"], errors="raise")
df["pop"]   = pd.to_numeric(df["pop"],   errors="raise")
df["cpi"]   = pd.to_numeric(df["cpi"],   errors="raise")

df["g_gdpos"]   = np.log(df["gdpos"]).diff()
df["n_pop"]     = np.log(df["pop"]) # So we won't difference twice
df["inflation"] = np.log(df["cpi"]).diff()

df.drop(columns=log_vars, inplace=True)

# In[Plotter Function]

def plot_economic_indicators(df, date_column, indicators):
    """
    Plots each indicator in its own figure, reading the x-axis from either
    a column or (if missing) from the DataFrame’s index.

    """
    # 1) Determine dates for the x-axis
    if date_column in df.columns:
        dates = pd.to_datetime(df[date_column])
    else:
        # Fallback: assume the index holds the dates
        dates = pd.to_datetime(df.index)

    # 2) Plot each indicator separately
    for col in indicators:
        if col not in df.columns:
            raise KeyError(f"Indicator column '{col}' not found in DataFrame.")
        plt.figure(figsize=(10, 3))
        plt.plot(dates, df[col], linewidth=1)
        plt.title(col)
        plt.xlabel(date_column if date_column in df.columns else "Date (index)")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

# In[Stationarity testing function (ADFuller)]

def adf_test(series, name='', alpha=0.05):

    result = adfuller(series.dropna())
    adf_stat, p_value = result[0], result[1]
    print(f'{name}: ADF={adf_stat:.3f}, p-value={p_value:.3f}')
    # True = nicht stationär (p > alpha) → hier wollen wir differenzieren
    return p_value > alpha


for col in dep_vars:
    clean_series = df[col].dropna()
    adf_test(df[col], name=col)

# In[Differentiating Function]

def differentiate_df(df: pd.DataFrame,
                     dep_vars: list[str],
                     alpha: float = 0.05) -> pd.DataFrame:
    """
    For each column in df:
      – If ADF says non-stationary (p > alpha), replace with series.diff()
      – Otherwise leave it in levels
    """
    df_diff = pd.DataFrame(index=df.index)

    for col in df.columns:
        series = df[col].astype(float)
        needs_diff = adf_test(series, name=col, alpha=alpha)

        if needs_diff:
            transformed = series.diff()
            df_diff[col] = transformed
            if col in dep_vars:
                print(f"{col}: Differenziert (1st difference)")

        else:
            df_diff[col] = series
            if col in dep_vars:
                print(f"{col}: Bereits stationär, keine Differenzierung")

    return df_diff

# In[Differntiating]

df_diff = differentiate_df(df, dep_vars, alpha=0.05)

df["inflation"] = 100 * df["inflation"]

# In[Plotting]

plot_economic_indicators(df, "date_parsed", dep_vars)
plot_economic_indicators(df_diff, "date_parsed", dep_vars)

# In[Scaling]

scaler = StandardScaler()
scaled_arr = scaler.fit_transform(df_diff)
df_scaled = pd.DataFrame(scaled_arr, index=df_diff.index, columns=df_diff.columns)


# In[Saving]

df_diff.to_csv(processed_folder / 'cleaned_data.csv')
df_scaled.to_csv(processed_folder / 'scaled_data.csv')


# In[14]

"""SUMMARY

CPI : Seasonaly adjusted + Log-Differantiated --> Inflation

GDPOS : Seasonaly adjusted + Log-Differantiated --> n_gdpos

LRate : Seasonaly adjusted + Differentiated

SRate : Seasonaly adjusted + Differentiated

-----
POP : Seasonaly adjusted + Log- Differentiated --> n_pop

"""
