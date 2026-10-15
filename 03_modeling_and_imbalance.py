# =============================================================================
#
# This script evaluates 3 baseline models using 5-fold cross-validation.
# To handle the imbalanced target (11.7% positive rate), we apply cost-sensitive 
# learning using class_weight='balanced' for all models.
# We report ROC-AUC and Average Precision (PR-AUC) as they are more informative 
# than accuracy for rare events.
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

def build_full_pipeline(model):
    """Wraps the feature engineering, preprocessor, and the chosen model."""
    return Pipeline(steps=[
        ("engineering", BankFeatureEngineer()),
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

# =============================================================================
# 3. DEFINE MODELS & CLASS IMBALANCE STRATEGY (Requirements 3 & 4)
# =============================================================================
# We use class_weight="balanced" to heavily penalize missing the minority class.
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    "LightGBM": LGBMClassifier(class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
}

# =============================================================================
# 4. CROSS-VALIDATION EVALUATION
# =============================================================================
# Use StratifiedKFold to maintain the 11.7% imbalance in every fold
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ['roc_auc', 'average_precision'] # average_precision is PR-AUC

print("\n--- EVALUATING MODELS (5-Fold CV) ---")
print("Target Strategy: class_weight='balanced' to handle imbalance.\n")

results = []
for name, model in models.items():
    print(f"Training {name}...")
    pipeline = build_full_pipeline(model)
    
    cv_results = cross_validate(
        pipeline, X_dev, y_dev, 
        cv=cv_strategy, 
        scoring=scoring_metrics,
        n_jobs=-1
    )
    
    mean_roc_auc = np.mean(cv_results['test_roc_auc'])
    mean_pr_auc = np.mean(cv_results['test_average_precision'])
    
    results.append({
        "Model": name,
        "ROC-AUC": f"{mean_roc_auc:.4f}",
        "PR-AUC (Avg Precision)": f"{mean_pr_auc:.4f}"
    })

# =============================================================================
# 5. DISPLAY RESULTS
# =============================================================================
results_df = pd.DataFrame(results)
print("\n--- FINAL CROSS-VALIDATION RESULTS ---")
print(results_df.to_string(index=False))
print("\nStep 3 & 4 complete! Next step: Hyperparameter tuning with Optuna.")