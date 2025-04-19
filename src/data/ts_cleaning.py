# In[1]
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL


# In[2] Path

project_root = Path().resolve().parent.parent
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

processed_file = processed_folder / "cleaned_macro_series1.xlsx"
processed_file_pkl = processed_folder / "cleaned_macro_series1.pkl"


df_diff.to_excel(processed_file, index=False)
df_diff.to_pickle(processed_file_pkl)

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



# In[] Graphen, saisonale decomp
"""for col in df_diff_proper.columns:
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

"""
# In[]

df_fully_adjusted = pd.DataFrame()
df_fully_adjusted["date_parsed"] = df_diff_proper["date_parsed"]

# Für jede Spalte (außer Datum): saisonale Komponente entfernen
for col in df_diff_proper.columns:
    if col != "date_parsed":
        try:
            series = df_diff_proper[[col]].copy()
            series["date_parsed"] = df_diff_proper["date_parsed"]
            series = series.set_index("date_parsed")

            # STL mit period=4 (z.B. Quartalsdaten)
            stl = STL(series[col].dropna(), period=4)
            result = stl.fit()

            # Saisonbereinigte Serie = Trend + Residuum (also ohne saisonale Komponente)
            adjusted = result.trend + result.resid
            adjusted.name = col

            # Wieder zum gemeinsamen DF hinzufügen
            df_fully_adjusted = df_fully_adjusted.merge(
                adjusted.reset_index(), on="date_parsed", how="left"
            )

        except Exception as e:
            print(f"Fehler bei Spalte {col}: {e}")


processed_file2 = processed_folder / "cleaned_macro_series2.xlsx"
processed_file2_pkl = processed_folder / "cleaned_macro_series2.pkl"

df_fully_adjusted.columns = [col.replace("_diff", "") for col in df_fully_adjusted.columns]

df_fully_adjusted.to_excel(processed_file2, index=False)
df_fully_adjusted.to_pickle(processed_file2_pkl)

print("Processed DataFrame saved")

########################################
##########       DONE       ############
########################################
