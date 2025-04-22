# In[1]
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

# In[Train-Test]



# In[Path]

project_root = Path().resolve().parent.parent
print(f"Projektroot: {project_root}")
raw_folder = project_root / "data" / "raw"
raw_data = raw_folder / "Macro_series_FS25.xlsx"
processed_folder = project_root / "data" / "processed"
print(f"Data: {raw_data}")

df = pd.read_excel(raw_data)
df.rename(columns={df.columns[0]: "date"}, inplace=True)

# In[Column Renaming]
df.drop(index=0, inplace=True)

# In[5]
print(df["date"].unique())
print(f"Type von (Einzel)Einträge von Datum {type(df['date'].iloc[0])}")


# In[Date Parsing]

df["year"] = df["date"].str.split(" ").str[0].astype(int)  # Extract year as integer
df["quarter"] = df["date"].str.split(" ").str[1]

quarter_to_month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}  # Define the quarter-to-month mapping
df["month"] = df["quarter"].map(quarter_to_month)

df["date_parsed"] = pd.to_datetime(
    df[["year", "month"]].assign(day=1)  # Assign the first day of the month
)

# In[Formatting numerics]

exclude_cols = {"year", "month", "quarter", "date", "date_parsed"}

relevant_cols = [col for col in df.columns if col not in exclude_cols]

# Convert all relevant columns to numeric
for col in relevant_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"Columns with NaNs: {[col for col in df.columns if df[col].isna().sum() > 0]}")
print("NaN counts for each column:")
for col in relevant_cols:
    if df[col].isna().sum() > 0:
        print(f"{col}: {df[col].isna().sum()} NaN values")

# In[NaN Handling]

def fill_missing_with_first_observation(df, column):
    """Fill NaN values in a column with the first valid (non-NaN) observation."""
    first_valid = df[column].dropna().iloc[0]
    df[column] = df[column].fillna(first_valid)

for col in relevant_cols:
    if col not in exclude_cols and df[col].isna().sum() > 0:
        fill_missing_with_first_observation(df, col)


# In[NORMALIZATION]
for col in relevant_cols:
    mean_col = df[col].mean
    std_col = df[col].std
    if std_col != 0:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

# In[Plotting function]

def plot_economic_indicators(df, date_column, indicators):
    """
    PLOTTING FUNCTION THAT TAKES DF AND INDICATORS AS INPUT
    """
    fig, axs = plt.subplots(len(indicators), 1, figsize=(10, 12), sharex=True)

    for ax, (column, title) in zip(axs, indicators):
        ax.plot(df[date_column], df[column])
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

indicators1 = [
    ("gdp", "GDP"),
    ("cpi", "CPI"),
    ("srate", "Short-Term Interest Rate"),
    ("lrate", "Long-Term Interest Rate")
]

plot_economic_indicators(df, "date_parsed", indicators1)


# In[9] Stationarity testing function (ADFuller)

def adf_test(series, name=''):
    result = adfuller(series.dropna())
    print(f'{name}: ADF={result[0]:.3f}, p-value={result[1]:.3f}')
    

for col in ["gdp", "cpi", "srate", "lrate"]:
    adf_test(df[col], name=col)

# In[10]

df_temp = df.copy()

for col in ["gdp", "cpi", "lrate"]:
    df_temp[col] = df[col].diff()


# In[Check if stationarity is now given]

for col in ["gdp", "cpi", "srate", "lrate"]:
    adf_test(df_temp[col], name=col)


plot_economic_indicators(df_temp, "date_parsed", indicators1)

# In[Plot for trends and seasonality]

for col in ["gdp", "cpi", "srate", "lrate"]:
    series = df_temp[col].dropna()
    stl = STL(series, period=4)  # quartalsweise Daten
    result = stl.fit()

    result.plot()
    plt.show()


# gdp_diff: Only "stronger" seasonal oscillation towards the end -> no seasonal correction needed
# cpi_diff: so clear seasonal pattern visible -> no seasonal correction needed
# srate:    no clear seasonal oscillation visible, only varying oscillation in the timeframe
# lrate_diff: shows pattern of a seasonal pattern -> needs seasonal adjustment

# NOTE: Adjustment for seanoality will be done quantitavely in the later section on the full data set...

# In[Adjusting one variable for seasonality]

series = df_temp['lrate'].dropna()
stl = STL(series, period=4)
result = stl.fit()
df_temp['lrate'] = series - result.seasonal

# In[14]

"""SUMMARY

CPI : Differantiated

GDP : Differantiated

LRate : Differantiated + Seasonaly adjusted

SRate : -

"""

# In[15]

processed_file = processed_folder / "cleaned_macro_series1.xlsx"
processed_file_pkl = processed_folder / "cleaned_macro_series1.pkl"


df_temp.to_excel(processed_file, index=False)
df_temp.to_pickle(processed_file_pkl)

print(f"Processed DataFrame saved to {processed_folder}")

########################################
########      STATIONARITY     #########
########################################

# In[Non-Stationarity identifying]

def identify_non_stationary_columns(dataframe, columns, significance_level=0.05):

    non_stationary_columns = []

    for col in columns:
        try:
            # ADF-Test durchführen
            result = adfuller(dataframe[col].dropna())
            pval = result[1]

            # Spalte als nicht-stationär hinzufügen, falls p-Wert > Signifikanzniveau
            if pval > significance_level:
                non_stationary_columns.append(col)
        except Exception as e:
            print(f"{col}: Fehler beim ADF-Test – {str(e)}")

    return non_stationary_columns

non_stationary1 = identify_non_stationary_columns(df, relevant_cols)

print("\nNicht-stationäre Variablen (p > 0.05):")
print(non_stationary1)
print("Anzahl Nicht-stationäre Variabeln: " + str(len(non_stationary1)) + " von " + str(len(relevant_cols)))


# In[Diff falls non_stationary]

df_2 = df.copy()

if non_stationary1:
    # Differenzierte Werte berechnen und die Originalspalten ersetzen
    for col in non_stationary1:
        df_2[col] = df[col].diff()


# In[Check 1 for stationarity]

non_stationary2 = identify_non_stationary_columns(df_2, relevant_cols)
print("\nNicht-stationäre Variablen (p > 0.05):")
print(non_stationary2)
print("Anzahl Nicht-stationäre Variabeln: " + str(len(non_stationary2)) + " von " + str(len(relevant_cols)))
for col in ['van19_21', 'van1921', 'ndomdem', 'niinv1', 'fwork', 'pop', 'lfpot']:
    adf_test(df_2[col], name=col)

indicators2 = [
    ("van19_21", "Wertschöpfung, Verarbeitendes Gewerbe/Herstellung  von Waren (Pharma)"),
    ("van1921", "Wertschöpfung,"),
    ("ndomdem", "Inländische Nachfrage"),
    ("fwork", "Grenzgänger"),
    ("pop", "Ständige Wohnbevölkerung"),
    ("lfpot", "Arbeitskräftepotential"),
    ("niinv1", "Lagervänderungen inkl. Stat Differenz")
]


plot_economic_indicators(df_2, "date_parsed", indicators2)

# In[Second diff-run]

df_3 = df_2.copy()

if non_stationary2:
    # Differenzierte Werte berechnen und die Originalspalten ersetzen
    for col in non_stationary2:
        df_3[col] = df_2[col].diff()


# In[Check 2 for stationarity]

non_stationary3 = identify_non_stationary_columns(df_3, relevant_cols)
print("\nNicht-stationäre Variablen (p > 0.05):")
print(non_stationary3)
print("Anzahl Nicht-stationäre Variabeln: " + str(len(non_stationary3)) + " von " + str(len(relevant_cols)))

for col in ['van19_21', 'van1921', 'ndomdem', 'niinv1', 'fwork', 'pop', 'lfpot']:
    adf_test(df_3[col], name=col)

plot_economic_indicators(df_3, "date_parsed", indicators2)





# In[Seasonaility Adjustment]

########################################
########      SEASONALITY      #########
########################################


# Für jede relevante Spalte : saisonale Komponente entfernen
for col in df_3.columns:
    if col not in {"date", "year", "month", "quarter", "date_parsed"}:  # Skip non-relevant columns
        series = df_3[col].dropna()  # Drop NaN values for STL processing
        stl = STL(series, period=4)  # STL decomposition with quarterly data
        result = stl.fit()

        # Saisonbereinigte Serie = Trend + Residuum (also ohne saisonale Komponente)
        df_3[col] = series - result.seasonal



processed_file2 = processed_folder / "cleaned_macro_series2.xlsx"
processed_file2_pkl = processed_folder / "cleaned_macro_series2.pkl"

df_3 = df_3.iloc[2:] # Because we differentiated twice -> won't have NaN's
df_3.to_excel(processed_file2, index=False)
df_3.to_pickle(processed_file2_pkl)

print("Processed DataFrame saved")

########################################
##########       DONE       ############
########################################
