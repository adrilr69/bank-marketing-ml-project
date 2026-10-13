# =============================================================================
# 
# This script uses Optuna to find the best hyperparameters for LightGBM.
# It uses 5-fold cross-validation, optimizing for PR-AUC (Average Precision),
# which is the most appropriate metric for our imbalanced fraud detection task.
# The entire preprocessing pipeline is fitted inside each CV fold to prevent leakage.
# =============================================================================

import pandas as pd
import numpy as np
import optuna
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold
from lightgbm import LGBMClassifier

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("Loading development data...")
X_dev = pd.read_csv("data/X_dev.csv")
y_dev = pd.read_csv("data/y_dev.csv").squeeze()

# =============================================================================
# 2. REBUILD PIPELINE ARCHITECTURE (For strict Cross-Validation)
# =============================================================================
class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_new = X.copy()
        X_new['previously_contacted'] = (X_new['previous'] > 0).astype(int)
        X_new['is_negative_balance'] = (X_new['balance'] < 0).astype(int)
        X_new['total_interactions'] = X_new['campaign'] + X_new['previous']
        return X_new

numeric_features = [
    'age', 'balance', 'day', 'campaign', 'pdays', 'previous', 
    'previously_contacted', 'is_negative_balance', 'total_interactions'
]
categorical_features = [
    'job', 'marital', 'education', 'default', 'housing', 
    'loan', 'contact', 'month', 'poutcome'
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))]), categorical_features)
    ]
)

# =============================================================================
# 3. DEFINE OPTUNA OBJECTIVE FUNCTION (Requirement 6)
# =============================================================================
def objective(trial):
    """
    Optuna objective function: defines the hyperparameter search space 
    and evaluates it using cross-validation.
    """
    # 1. Define the search space for LightGBM
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'class_weight': 'balanced', # Keep our imbalance handling strategy
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # 2. Instantiate the model with the suggested parameters
    model = LGBMClassifier(**params)
    
    # 3. Build the full pipeline
    pipeline = Pipeline(steps=[
        ("engineering", BankFeatureEngineer()),
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])
    
    # 4. Evaluate using 5-Fold Stratified CV
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # We optimize for Average Precision (PR-AUC) due to the class imbalance
    scores = cross_val_score(
        pipeline, X_dev, y_dev, 
        cv=cv_strategy, 
        scoring='average_precision',
        n_jobs=-1
    )
    
    # Return the mean CV score
    return np.mean(scores)

# =============================================================================
# 4. RUN OPTUNA STUDY
# =============================================================================
print("\n--- STARTING OPTUNA HYPERPARAMETER SEARCH ---")
print("Model: LightGBM")
print("Metric: PR-AUC (Average Precision)")
print("CV Strategy: 5-Fold Stratified")

# We want to MAXIMIZE the PR-AUC score
study = optuna.create_study(direction="maximize", study_name="LightGBM_Tuning")

# Run 20 trials (keep it small for a POC, increase in production)
study.optimize(objective, n_trials=20)

print("\n--- OPTUNA SEARCH FINISHED ---")
print(f"Best PR-AUC Score: {study.best_value:.4f}")
print("Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# =============================================================================
# 5. REFIT AND SAVE THE BEST MODEL
# =============================================================================
print("\n--- REFITTING BEST MODEL ON ALL DEVELOPMENT DATA ---")

# Create a new pipeline using the best parameters found
best_model = LGBMClassifier(
    **study.best_params, 
    class_weight="balanced", 
    random_state=42, 
    n_jobs=-1, 
    verbose=-1
)

best_pipeline = Pipeline(steps=[
    ("engineering", BankFeatureEngineer()),
    ("preprocessing", preprocessor),
    ("classifier", best_model)
])

# Fit on the entire X_dev / y_dev
best_pipeline.fit(X_dev, y_dev)

# Save the final tuned model
joblib.dump(best_pipeline, "models/best_lightgbm_pipeline.pkl")
print("Saved 'best_lightgbm_pipeline.pkl' to 'models/' folder. Step 6 complete!")