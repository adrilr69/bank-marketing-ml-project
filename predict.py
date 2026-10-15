# predict.py - use the final model to score new clients
#
# The saved model is the whole pipeline (feature engineering + preprocessing + LightGBM + Platt calibration),
# so it takes the raw columns directly (same columns as data/X_test.csv, without "duration").
#
# Usage:
#   python predict.py                      -> scores the example client written below
#   python predict.py data/X_test.csv      -> scores every row of a csv file

import sys
import json
import joblib
import pandas as pd

model = joblib.load("models/final_calibrated_model.pkl")
info = json.load(open("results/final_model_info.json"))
threshold = info["threshold"]

if len(sys.argv) > 1:
    clients = pd.read_csv(sys.argv[1])
else:
    clients = pd.DataFrame([{
        "age": 34,
        "job": "management",
        "marital": "single",
        "education": "tertiary",
        "default": "no",
        "balance": 2500,
        "housing": "no",
        "loan": "no",
        "contact": "cellular",
        "day": 14,
        "month": "apr",
        "campaign": 1,
        "pdays": 120,
        "previous": 2,
        "poutcome": "success",
    }])

proba = model.predict_proba(clients)[:, 1]
clients["proba_subscribe"] = proba.round(3)
clients["call_client"] = proba >= threshold

print("Model:", info["model"], "+", info["calibration"], "| threshold =", round(threshold, 3))
print(clients[["age", "job", "contact", "poutcome", "proba_subscribe", "call_client"]].head(20).to_string())
if len(clients) > 1:
    print("\nNumber of clients:", len(clients), "| clients to call:", int(clients["call_client"].sum()))
