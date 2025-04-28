# In[IMPORTS]
import pickle
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from pathlib import Path

# In[Setting Parameters]

SPLIT_PARAM = "0_3"

# In[Setting Path's]

project_root = Path().resolve()
print(f"Projektroot: {project_root}")

processed_folder = project_root / "data" / "processed"
data_train_path = processed_folder / "test_train" / "train_splits.xlsx"
data_test_path = processed_folder / "test_train" / "test_splits.xlsx"
pca_path = processed_folder / "pca_outputs" / f"eigenvectors_train_{SPLIT_PARAM}.pkl"
print(f"Data Path: & pca_path: {pca_path}")

results_path = processed_folder / "results"
models_path = results_path/ "forecasts"

# In[Loading Data]

eigen_full = pickle.load(open(pca_path, "rb"))
eigenvectors = eigen_full.iloc[:20, :] # Only the first 20 PCA's

exclude_cols = {"year", "month", "quarter", "date", "date_parsed", "ngdp",
                "gdp_prod", "ngdpos", "pgdp", "gdpoi", "gdpos"}

X = pd.read_excel(data_train_path, sheet_name=f"train_{SPLIT_PARAM}")

X = X[[col for col in X.columns if col not in exclude_cols]]

X_test = pd.read_excel(data_test_path, sheet_name=f"test_{SPLIT_PARAM}")

# In[De-Bug: Matrix-Multiplication consitency check (Dimensions)]

print(f"X shape: {X.shape}")
print(f"eigenvectors shape: {eigenvectors.T.shape}")
print(f"eigenvectors FULL shape: {eigen_full.shape}")

# In[Projection: factor  -> \]

factors = X @ eigenvectors.T
factors_df = pd.DataFrame(factors, index=X.index)

# In[VAR(1) estimate]

model = VAR(factors_df)
results = model.fit(maxlags=1)

# In[Forecast for next quarter]

factors_forecast = results.forecast(factors_df.values[-results.k_ar:], steps=1)

# In[Transformation into original "space"]

predicted_X = factors_forecast @ eigenvectors
predicted_X = pd.DataFrame(predicted_X, columns=X.columns)

# In[Predictions]

predicted_CPI = predicted_X['cpi'].values[0]
predicted_GDP = predicted_X['gdp'].values[0]

actual_CPI = X_test['cpi'].values[0]
actual_GDP = X_test['gdp'].values[0]

print(f"Predicted CPI next quarter:, {predicted_CPI} | Actual CPI next quarter:, {actual_CPI}")
print(f"Predicted GDP next quarter:, {predicted_GDP} | Actual GDP next quarter:, {actual_GDP}")
# In[]



# In[]
