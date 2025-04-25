import pandas as pd
from pathlib import Path


from pathlib import Path
import pandas as pd
import numpy as np
import pickle

project_root = Path().resolve().parent.parent
print(f"Projektroot: {project_root}")

processed_folder = project_root / "data" / "processed"
processed_data = processed_folder / "cleaned_macro_series2.xlsx"
save_path = processed_folder / "test_train"
print(f"Processed Data Path: {processed_data}")

df = pd.read_excel(processed_data)

# In[Train-Test] Funktion

def split_time_series(df: pd.DataFrame, date_col: str = "date_parsed", train_ratio: float = 0.5):

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    return train_df, test_df

# In[Train-Test] Schlaufe über Funktion mit 0.025er Schritten

ratios = np.arange(0.2, 0.8, 0.025)
ratios = [round(r, 3) for r in ratios]

train_dfs = {}
test_dfs = {}

for ratio in ratios:
    train_df, test_df = split_time_series(df, train_ratio=ratio)


    train_dfs[ratio] = train_df
    test_dfs[ratio] = test_df


    print(f"\nTrain ratio = {ratio:.3f}")
    print(f"  Train: {train_df['date_parsed'].min().date()} → {train_df['date_parsed'].max().date()}  ({len(train_df)} Zeilen)")
    print(f"  Test : {test_df['date_parsed'].min().date()} → {test_df['date_parsed'].max().date()}  ({len(test_df)} Zeilen)")

# In[Train-Test] Schlaufe Sets einsehen

train_dfs[0.8]

# In[Train-Test] Excel mit mehreren sheets


# Excel mit mehreren Sheets (je Trainings-Ratio)
excel_path = save_path / "train_splits.xlsx"
with pd.ExcelWriter(excel_path) as writer:
    for ratio, df in train_dfs.items():
        sheet_name = f"train_{str(ratio).replace('.', '_')}"
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Train-Excel-Datei gespeichert unter: {excel_path}")


pickle_path = save_path / "train_dfs.pkl"
with open(pickle_path, "wb") as f:
    pickle.dump(train_dfs, f)

print(f"Train-Pickle-Datei gespeichert unter: {pickle_path}")

# Excel-Datei mit mehreren Sheets (eine pro Test-Ratio)
test_excel_path = save_path / "test_splits.xlsx"
with pd.ExcelWriter(test_excel_path) as writer:
    for ratio, df in test_dfs.items():
        sheet_name = f"test_{str(ratio).replace('.', '_')}"
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Test-Excel-Datei gespeichert unter: {test_excel_path}")


test_pickle_path = save_path / "test_dfs.pkl"
with open(test_pickle_path, "wb") as f:
    pickle.dump(test_dfs, f)

print(f"Test-Pickle-Datei gespeichert unter: {test_pickle_path}")