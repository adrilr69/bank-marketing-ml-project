# Shared code used by all the scripts from step 3 to step 10
# (same feature engineering + preprocessing as 02_preprocessing_pipeline.py)
# We put it in one file so that every script builds the exact same pipeline
# and so the saved .pkl models can be loaded again from another script.

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        # 1. was the client contacted in a previous campaign ?
        X_new["previously_contacted"] = (X_new["previous"] > 0).astype(int)
        # 2. is the balance negative ?
        X_new["is_negative_balance"] = (X_new["balance"] < 0).astype(int)
        # 3. total number of contacts (this campaign + before)
        X_new["total_interactions"] = X_new["campaign"] + X_new["previous"]
        return X_new


numeric_features = [
    "age", "balance", "day", "campaign", "pdays", "previous",
    "previously_contacted", "is_negative_balance", "total_interactions"
]
categorical_features = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome"
]


def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])
    return preprocessor


def build_full_pipeline(model):
    # feature engineering -> preprocessing -> model, all in one Pipeline
    pipeline = Pipeline([
        ("engineering", BankFeatureEngineer()),
        ("preprocessing", make_preprocessor()),
        ("classifier", model)
    ])
    return pipeline


def load_data():
    # same split as step 1 (saved in the data folder)
    X_dev = pd.read_csv("data/X_dev.csv")
    y_dev = pd.read_csv("data/y_dev.csv").squeeze()
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()
    return X_dev, y_dev, X_test, y_test
