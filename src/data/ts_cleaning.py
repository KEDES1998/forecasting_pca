# In[1]
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt

from requests import delete

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

plt.plot(df['date_parsed'], df['gdp'])
plt.show()

# In[9] CPI-Plot

plt.plot(df['date_parsed'], df['cpi'])
plt.show()

# In[10] Short-term interest rate Plot

plt.plot(df['date_parsed'], df['srate'])
plt.show()

# In[11] Long-term interest rate Plot

plt.plot(df['date_parsed'], df['lrate'])
plt.show()

# In[12]

processed_file = processed_folder / "cleaned_macro_series.xlsx"
processed_file_pkl = processed_folder / "cleaned_macro_series.pkl"


df.to_excel(processed_file, index=False)
df.to_pickle(processed_file_pkl)

print(f"Processed DataFrame saved to {processed_folder}")

# In[]
