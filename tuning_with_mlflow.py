# Step 6 & 7 - Optuna tuning of the 3 models + experiment tracking with MLflow
#
# In hyperparameter_tuning.py only LightGBM was tuned. Here we tune the 3 models
# (Logistic Regression, Random Forest, LightGBM) with Optuna and we log EVERY trial in MLflow
# (parameters + cross-validation metrics). For the best model of each family we also log
# artifacts (fitted pipeline, optimization plot, table of all trials).
#
# Score optimized = mean PR-AUC with 5-fold stratified CV on the dev set.
# The test set is NOT used here, so the CV score is only used to choose, and the test set
# stays untouched for the final evaluation (final_comparison.py).
#
# MLflow is not in the course slides, we use it because it is the standard tool for experiment tracking.
# To look at the runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db   then open http://127.0.0.1:5000

import os
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import mlflow
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from features import build_full_pipeline, load_data

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bank_marketing_tuning")

X_dev, y_dev, X_test, y_test = load_data()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# number of Optuna trials for each model
n_trials = {"logistic_regression": 20, "random_forest": 25, "lightgbm": 30}


def suggest_params(model_name, trial):
    # search space of each model
    if model_name == "logistic_regression":
        params = {
            "C": trial.suggest_float("C", 0.001, 10, log=True),
            # l1_ratio: 0 = Ridge (L2), 1 = Lasso (L1), in between = Elastic Net (session 2)
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    elif model_name == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "max_features": trial.suggest_float("max_features", 0.1, 0.7),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    else:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10, log=True),
            # weight of the positive class (session 6: N_neg / N_pos = 7.5 is only a starting point)
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 8.0),
        }
    return params


def make_model(model_name, params):
    if model_name == "logistic_regression":
        model = LogisticRegression(**params, solver="saga", max_iter=2000, random_state=42)
    elif model_name == "random_forest":
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    else:
        # subsample only works in LightGBM if subsample_freq > 0 (it was missing in hyperparameter_tuning.py)
        model = LGBMClassifier(**params, subsample_freq=1, random_state=42, n_jobs=-1, verbose=-1)
    return model


def objective(trial, model_name):
    params = suggest_params(model_name, trial)
    pipeline = build_full_pipeline(make_model(model_name, params))

    start = time.time()
    scores = cross_validate(pipeline, X_dev, y_dev, cv=cv,
                            scoring=["average_precision", "roc_auc"], n_jobs=-1)
    duration = time.time() - start

    pr_auc = scores["test_average_precision"].mean()

    # one MLflow run for each trial (inside the run of the model)
    with mlflow.start_run(run_name=model_name + "_trial_" + str(trial.number), nested=True):
        mlflow.log_param("model", model_name)
        mlflow.log_param("trial_number", trial.number)
        mlflow.log_params(params)
        mlflow.log_metric("cv_pr_auc", pr_auc)
        mlflow.log_metric("cv_pr_auc_std", scores["test_average_precision"].std())
        mlflow.log_metric("cv_roc_auc", scores["test_roc_auc"].mean())
        mlflow.log_metric("cv_time_seconds", duration)

    return pr_auc


best_params = {}
summary = []

for model_name in n_trials:
    print("\n========== TUNING", model_name, "==========")

    with mlflow.start_run(run_name=model_name + "_optuna_study"):
        mlflow.log_param("model", model_name)
        mlflow.log_param("n_trials", n_trials[model_name])
        mlflow.log_param("cv", "5-fold stratified")
        mlflow.log_param("metric", "PR-AUC (average precision)")

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda trial: objective(trial, model_name), n_trials=n_trials[model_name])

        print("Best CV PR-AUC:", round(study.best_value, 4))
        print("Best params:", study.best_params)

        for key in study.best_params:
            mlflow.log_param("best_" + key, study.best_params[key])
        mlflow.log_metric("best_cv_pr_auc", study.best_value)

        # all the trials in a csv
        trials_df = study.trials_dataframe()
        trials_df.to_csv("results/optuna_trials_" + model_name + ".csv", index=False)
        mlflow.log_artifact("results/optuna_trials_" + model_name + ".csv")

        # plot: score of each trial + best score so far
        values = trials_df["value"].values
        plt.figure(figsize=(7, 4))
        plt.plot(values, "o", label="trial score")
        plt.plot(np.maximum.accumulate(values), "-", label="best so far")
        plt.xlabel("Trial")
        plt.ylabel("CV PR-AUC")
        plt.title("Optuna history - " + model_name)
        plt.legend()
        plt.savefig("plots/optuna_history_" + model_name + ".png", dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact("plots/optuna_history_" + model_name + ".png")

        # refit the best pipeline on the whole dev set and save it
        best_pipeline = build_full_pipeline(make_model(model_name, study.best_params))
        best_pipeline.fit(X_dev, y_dev)
        joblib.dump(best_pipeline, "models/best_" + model_name + ".pkl")
        mlflow.log_artifact("models/best_" + model_name + ".pkl")

        best_params[model_name] = study.best_params
        summary.append({"model": model_name, "n_trials": n_trials[model_name],
                        "best_cv_pr_auc": study.best_value})

# save the best params for the next scripts
with open("models/best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

summary_df = pd.DataFrame(summary)
print("\n--- SUMMARY OF THE TUNING ---")
print(summary_df.round(4).to_string(index=False))
summary_df.to_csv("results/tuning_summary.csv", index=False)

# export of all the MLflow runs (so we have them in the repo / report)
runs = mlflow.search_runs(experiment_names=["bank_marketing_tuning"])
runs.to_csv("results/mlflow_runs.csv", index=False)
print("Number of MLflow runs logged:", len(runs))
