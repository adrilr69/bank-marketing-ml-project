# Step 9 - Calibration of the final model (session 7)
#
# Part A - diagnostic of the final model (trained on the whole dev set), on the test set:
#          reliability diagram + histogram of the probabilities, Brier score, ECE
#          -> is the model over-confident or under-confident ?
# Part B - recalibration. The calibrator must learn on predictions of data the model never saw.
#          So we refit the same model on 80% of the dev set (train) and we fit Platt scaling and
#          isotonic regression on the other 20% (calibration set). The test set is only used to compare.

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, average_precision_score, precision_recall_curve
)

from features import load_data

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

X_dev, y_dev, X_test, y_test = load_data()

final_choice = json.load(open("results/final_choice.json"))
model_name = final_choice["model"]
pipeline = joblib.load("models/best_" + model_name + ".pkl")
print("Final model:", model_name)


def reliability_curve(y_true, y_prob, n_bins=10):
    # equal-width bins between 0 and 1, we skip the empty bins
    y_true = np.array(y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    conf = []
    acc = []
    counts = []
    for m in range(n_bins):
        if m == n_bins - 1:
            in_bin = (y_prob >= bins[m]) & (y_prob <= bins[m + 1])
        else:
            in_bin = (y_prob >= bins[m]) & (y_prob < bins[m + 1])
        if in_bin.sum() > 0:
            conf.append(y_prob[in_bin].mean())   # mean predicted probability
            acc.append(y_true[in_bin].mean())    # observed frequency of y = 1
            counts.append(in_bin.sum())
    return np.array(conf), np.array(acc), np.array(counts)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    conf, acc, counts = reliability_curve(y_true, y_prob, n_bins)
    return np.sum(counts / len(y_true) * np.abs(acc - conf))


def brier_score(y_true, y_prob):
    return np.mean((y_prob - np.array(y_true)) ** 2)


# ---------------- PART A : diagnostic ----------------
proba_test = pipeline.predict_proba(X_test)[:, 1]

conf, acc, counts = reliability_curve(y_test, proba_test)
bins_table = pd.DataFrame({"mean_predicted": conf, "observed_freq": acc, "n_points": counts,
                           "gap (pred - obs)": conf - acc})
print("\nReliability table (test set, 10 equal-width bins):")
print(bins_table.round(3).to_string(index=False))

# check with sklearn
sk_acc, sk_conf = calibration_curve(y_test, proba_test, n_bins=10, strategy="uniform")
print("Same as sklearn calibration_curve:", np.allclose(conf, sk_conf) and np.allclose(acc, sk_acc))

print("\nBrier score (our function):", round(brier_score(y_test, proba_test), 5))
print("Brier score (sklearn)     :", round(brier_score_loss(y_test, proba_test), 5))
print("Brier score of a model that always predicts the base rate:",
      round(brier_score(y_test, np.full(len(y_test), y_dev.mean())), 5))
print("ECE (10 bins):", round(expected_calibration_error(y_test, proba_test), 5))
print("Mean predicted probability:", round(proba_test.mean(), 4), "| real positive rate:", round(y_test.mean(), 4))
if proba_test.mean() > y_test.mean():
    print("-> on average the model predicts too high probabilities (over-estimation)")
else:
    print("-> on average the model predicts too low probabilities (under-estimation)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
axes[0].plot(conf, acc, "-", color="tab:blue")
axes[0].scatter(conf, acc, s=counts / counts.max() * 300 + 10, color="tab:blue", label="final model (size = n points)")
axes[0].set_xlabel("Mean predicted probability")
axes[0].set_ylabel("Observed frequency")
axes[0].set_title("Reliability diagram - " + model_name + " (test set)")
axes[0].legend()
axes[1].hist(proba_test, bins=50)
axes[1].set_yscale("log")
axes[1].set_xlabel("Predicted probability")
axes[1].set_ylabel("Number of clients (log scale)")
axes[1].set_title("Where the model puts its probabilities")
plt.savefig("plots/calibration_final_model.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------- PART B : recalibration ----------------
X_train, X_calib, y_train, y_calib = train_test_split(
    X_dev, y_dev, test_size=0.20, stratify=y_dev, random_state=42
)
print("\nTrain:", X_train.shape, "| Calibration:", X_calib.shape)

base_model = clone(pipeline)
base_model.fit(X_train, y_train)

# FrozenEstimator = the base model is not retrained, only the calibration map is learned
platt = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method="sigmoid")
platt.fit(X_calib, y_calib)
isotonic = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method="isotonic")
isotonic.fit(X_calib, y_calib)

probas = {
    "raw (trained on 80% dev)": base_model.predict_proba(X_test)[:, 1],
    "Platt (sigmoid)": platt.predict_proba(X_test)[:, 1],
    "isotonic": isotonic.predict_proba(X_test)[:, 1],
}

rows = []
for name in probas:
    rows.append({
        "model": name,
        "brier": brier_score(y_test, probas[name]),
        "ece": expected_calibration_error(y_test, probas[name]),
        "roc_auc": roc_auc_score(y_test, probas[name]),
        "pr_auc": average_precision_score(y_test, probas[name]),
        "mean_proba": probas[name].mean(),
    })
calib_df = pd.DataFrame(rows)
print("\n--- BEFORE / AFTER CALIBRATION (test set) --- real positive rate =", round(y_test.mean(), 4))
print(calib_df.round(4).to_string(index=False))
calib_df.to_csv("results/calibration_comparison.csv", index=False)

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
for name in probas:
    c, a, n = reliability_curve(y_test, probas[name])
    plt.plot(c, a, "o-", label=name)
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed frequency")
plt.title("Reliability diagrams before / after calibration (test set)")
plt.legend()
plt.savefig("plots/calibration_before_after.png", dpi=150, bbox_inches="tight")
plt.close()

# we keep Platt scaling (session 7: simple and stable default, only 2 parameters)
# new threshold chosen on the calibration set (best F1), not on the test set
proba_calib = platt.predict_proba(X_calib)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_calib, proba_calib)
f1 = 2 * precision * recall / (precision + recall + 1e-10)
new_threshold = thresholds[np.argmax(f1[:-1])]
print("\nThreshold for the calibrated model (best F1 on the calibration set):", round(new_threshold, 3))

joblib.dump(platt, "models/final_calibrated_model.pkl")
with open("results/final_model_info.json", "w") as f:
    json.dump({"model": model_name, "calibration": "sigmoid (Platt)", "threshold": float(new_threshold)}, f, indent=4)
print("Saved models/final_calibrated_model.pkl and results/final_model_info.json")
