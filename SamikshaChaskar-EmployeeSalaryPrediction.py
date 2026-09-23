# -*- coding: utf-8 -*-
"""
train_model.py
--------------
Loads the raw employee dataset, cleans it, engineers features,
trains a Linear Regression model to predict SALARY, evaluates it,
and saves the trained pipeline (preprocessing + model) to /model.
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "employee_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------------------------------------
# 1. Load raw data
# -----------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Raw dataset shape: {df.shape}")

# -----------------------------------------------------------------
# 2. Clean data
#    - The raw file has inconsistent casing ("Sales" / "SALES"),
#      missing values, and occasional duplicate records.
# -----------------------------------------------------------------
for col in ["department", "city", "remote_work"]:
    df[col] = df[col].astype(str).str.strip().str.title()
    df[col] = df[col].replace({"Nan": np.nan})

# Salary is the prediction target — rows without it can't be used
df = df.dropna(subset=["salary"])

# Drop exact duplicate employee records
df = df.drop_duplicates(subset=["emp_name", "join_date"])

# Engineer tenure (years) from join_date instead of using the raw date
df["join_date"] = pd.to_datetime(df["join_date"], errors="coerce")
REFERENCE_DATE = pd.Timestamp.today().normalize()
df["tenure_years"] = (REFERENCE_DATE - df["join_date"]).dt.days / 365.25
df = df.dropna(subset=["tenure_years"])

print(f"Cleaned dataset shape: {df.shape}")

NUMERIC_FEATURES     = ["age", "performance_rating", "tenure_years"]
CATEGORICAL_FEATURES = ["department", "city", "remote_work"]
TARGET                = "salary"

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET].values

# -----------------------------------------------------------------
# 3. Train / test split
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------------
# 4. Preprocessing + model pipeline
#    (missing numeric -> median, missing categorical -> most frequent,
#     numeric -> scaled, categorical -> one-hot encoded)
# -----------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()),
])

model.fit(X_train, y_train)

# -----------------------------------------------------------------
# 5. Evaluate
# -----------------------------------------------------------------
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n-- Model Evaluation --")
print(f"  MAE  : ${mae:,.2f}")
print(f"  RMSE : ${rmse:,.2f}")
print(f"  R2   : {r2:.4f}")

# -----------------------------------------------------------------
# 6. Save the fitted pipeline (preprocessing + model in one object)
# -----------------------------------------------------------------
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
print("\nModel pipeline saved to /model/model.pkl")

# Save the cleaned dataset alongside the raw one for reference / the frontend
cleaned_path = os.path.join(BASE_DIR, "data", "employee_data_cleaned.csv")
df.to_csv(cleaned_path, index=False)
print(f"Cleaned dataset saved: {cleaned_path}")

# -----------------------------------------------------------------
# 7. Diagnostic plots (saved to file; no GUI window needed)
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Employee Salary Prediction - Model Diagnostics", fontsize=14)

# Actual vs Predicted
axes[0].scatter(y_test, y_pred, color="#3b82d4", edgecolors="white", s=80, alpha=0.85)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             "r--", lw=1.5, label="Perfect fit")
axes[0].set_xlabel("Actual Salary ($)")
axes[0].set_ylabel("Predicted Salary ($)")
axes[0].set_title("Actual vs Predicted")
axes[0].legend()

# Residuals
residuals = y_test - y_pred
axes[1].hist(residuals, bins=10, color="#7c5cd8", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
axes[1].set_xlabel("Residual ($)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, "diagnostics.png")
plt.savefig(plot_path, dpi=120)
print(f"Diagnostics plot saved: {plot_path}")
