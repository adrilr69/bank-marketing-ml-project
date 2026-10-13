# =============================================================================
# 
# Step 1: Exploratory Data Analysis (EDA), Leakage Audit & Train/Test Split
#
# This script performs the initial data loading and basic integrity checks 
# for the Bank Marketing dataset (Option 2).
# It prepares the environment by creating 'data' and 'plots' directories.
# It renames features to meaningful names and converts the target to binary.
# It calculates basic statistics like row counts, missing values, 
# and class imbalance to understand the dataset before modeling.
# Finally, it audits for data leakage (dropping 'duration') and splits the 
# data into a development and a held-out test set using stratification.
# =============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# =============================================================================
# INITIAL SETUP & DATA LOADING
# =============================================================================
# Create folders to save our clean data and plots for the LaTeX report
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Load dataset
print("Loading Bank Marketing dataset from OpenML...")
bank = fetch_openml(data_id=1461, as_frame=True)
X = bank.data.copy()
y = bank.target.copy()

# Rename columns to meaningful business names
column_names = {
    "V1": "age",
    "V2": "job",
    "V3": "marital",
    "V4": "education",
    "V5": "default",
    "V6": "balance",
    "V7": "housing",
    "V8": "loan",
    "V9": "contact",
    "V10": "day",
    "V11": "month",
    "V12": "duration",
    "V13": "campaign",
    "V14": "pdays",
    "V15": "previous",
    "V16": "poutcome"
}
X = X.rename(columns=column_names)

# Convert target to binary 0 and 1
y = (y.astype(str) == "2").astype(int)
y.name = "target"

# =============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n--- BASIC DATASET INTEGRITY ---")
print(f"Number of rows: {X.shape[0]}")
print(f"Number of columns (predictors): {X.shape[1]}")
print(f"Missing values (NaN): {X.isna().sum().sum()}")
print(f"Fully duplicated rows: {pd.concat([X, y], axis=1).duplicated().sum()}")

# Check target distribution (Class Imbalance)
pos_count = (y == 1).sum()
neg_count = (y == 0).sum()
pos_rate = y.mean() * 100
print("\n--- TARGET DISTRIBUTION ---")
print(f"Class 0 (No subscription): {neg_count}")
print(f"Class 1 (Subscription):    {pos_count}")
print(f"Positive class prevalence: {pos_rate:.2f}%")

# Check for hidden missing values (coded as 'unknown')
print("\n--- 'UNKNOWN' CATEGORIES IN FEATURES ---")
categorical_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
for col in categorical_cols:
    unknown_count = (X[col].astype(str) == "unknown").sum()
    if unknown_count > 0:
        pct = (unknown_count / len(X)) * 100
        print(f"  - {col}: {unknown_count} 'unknown' values ({pct:.2f}%)")

# =============================================================================
# DATA LEAKAGE AUDIT (Session 1)
# =============================================================================
print("\n--- DATA LEAKAGE AUDIT ---")
# Drop 'duration' to prevent temporal data leakage 
# (call duration is unknown before placing the call)
X = X.drop(columns=["duration"])
print("Dropped 'duration' column to prevent data leakage.")

# =============================================================================
# SPLITTING STRATEGY (Session 1 - 80% Development / 20% Held-out Test)
# =============================================================================
print("\n--- SPLITTING DATA (80% DEV / 20% TEST, STRATIFIED) ---")
# Stratified split to maintain the positive class prevalence across sets
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(f"Development set: {X_dev.shape[0]} rows (Positive rate: {y_dev.mean()*100:.2f}%)")
print(f"Held-out test set: {X_test.shape[0]} rows (Positive rate: {y_test.mean()*100:.2f}%)")

# Save datasets so the next scripts use the exact same split
X_dev.to_csv("data/X_dev.csv", index=False)
y_dev.to_csv("data/y_dev.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
print("\nSaved X_dev, y_dev, X_test, y_test to 'data/' folder. Step 1 complete!")