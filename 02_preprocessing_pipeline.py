# =============================================================================
#
# This script loads the development data and builds a scikit-learn Pipeline.
# It includes a custom transformer to engineer 3 new features, followed by a 
# ColumnTransformer to handle numerical (scaling) and categorical (encoding) data.
# =============================================================================

import pandas as pd
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# Create a folder to store our saved pipeline/models
os.makedirs("models", exist_ok=True)

# 1. Load the development data saved from Step 1
print("Loading development data...")
X_dev = pd.read_csv("data/X_dev.csv")
y_dev = pd.read_csv("data/y_dev.csv").squeeze() # squeeze converts single-column DataFrame to Series

# =============================================================================
# FEATURE ENGINEERING (Requirement 2)
# =============================================================================
class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to engineer new features.
    Integrating this into the pipeline prevents data leakage.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Feature 1: Has the client been contacted before this campaign? (Binary)
        X_new['previously_contacted'] = (X_new['previous'] > 0).astype(int)
        
        # Feature 2: Is the account balance negative? (Binary)
        X_new['is_negative_balance'] = (X_new['balance'] < 0).astype(int)
        
        # Feature 3: Total interactions with the client
        X_new['total_interactions'] = X_new['campaign'] + X_new['previous']
        
        return X_new

# =============================================================================
# COLUMN TRANSFORMER & PIPELINE (Requirements 2 & 5)
# =============================================================================
# Identify column types, including the 3 new engineered features
numeric_features = [
    'age', 'balance', 'day', 'campaign', 'pdays', 'previous', 
    'previously_contacted', 'is_negative_balance', 'total_interactions'
]
categorical_features = [
    'job', 'marital', 'education', 'default', 'housing', 
    'loan', 'contact', 'month', 'poutcome'
]

# Define preprocessing for numerical columns (Impute missing -> Scale)
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Define preprocessing for categorical columns (Impute missing -> Encode)
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
])

# Combine both branches into a ColumnTransformer
print("Building the preprocessing pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# Wrap the custom feature engineering and the preprocessor into one main Pipeline
full_preprocessing_pipeline = Pipeline(steps=[
    ("engineering", BankFeatureEngineer()),
    ("preprocessing", preprocessor)
])

# =============================================================================
# FIT AND SAVE
# =============================================================================
# Fit the pipeline EXCLUSIVELY on the development data to avoid target leakage
print("Fitting the pipeline on the development data...")
X_dev_processed = full_preprocessing_pipeline.fit_transform(X_dev)

# Save the fitted pipeline so we can load it directly in the modeling scripts
joblib.dump(full_preprocessing_pipeline, "models/preprocessing_pipeline.pkl")

print(f"\nSuccess! Shape of processed data: {X_dev_processed.shape}")
print("Saved 'preprocessing_pipeline.pkl' to 'models/' folder. Step 2 & 5 complete!")