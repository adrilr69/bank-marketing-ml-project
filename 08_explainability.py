# Step 8 - Explainability (session 8)
#
# Global importance, 3 different methods (they answer different questions):
#   - native gain importance of LightGBM (computed during training)
#   - permutation importance on the test set with PR-AUC (how much the score drops if a column is shuffled)
#   - mean |SHAP value|
# Local explanations: SHAP waterfall plots for 4 clients of the test set (TP, FN, FP, TN)
# Careful: for LightGBM, SHAP explains the raw score = log-odds, not the probability.

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance

from features import load_data, numeric_features, categorical_features

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

X_dev, y_dev, X_test, y_test = load_data()

final_choice = json.load(open("results/final_choice.json"))
model_name = final_choice["model"]
threshold = final_choice["threshold"]
print("Final model:", model_name, "- threshold:", round(threshold, 3))

pipeline = joblib.load("models/best_" + model_name + ".pkl")
model = pipeline.named_steps["classifier"]

# the model only sees the data after feature engineering + preprocessing
X_test_transformed = pipeline[:-1].transform(X_test)
feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
print("Transformed test set:", X_test_transformed.shape)

# 1. native importance (gain)
if model_name == "lightgbm":
    gain = model.booster_.feature_importance(importance_type="gain")
else:
    gain = model.feature_importances_
gain_importance = pd.Series(gain, index=feature_names).sort_values(ascending=False)
gain_importance = gain_importance / gain_importance.sum()
print("\nTop 10 gain importance (normalized):")
print(gain_importance.head(10).round(3))

# 2. permutation importance on the raw columns, with the whole pipeline
perm = permutation_importance(pipeline, X_test, y_test, scoring="average_precision",
                              n_repeats=5, random_state=42, n_jobs=-1)
perm_importance = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
print("\nPermutation importance (drop of test PR-AUC when the column is shuffled):")
print(perm_importance.round(4))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
gain_importance.head(15).sort_values().plot(kind="barh", ax=axes[0])
axes[0].set_title("LightGBM gain importance (top 15)")
perm_importance.sort_values().plot(kind="barh", ax=axes[1])
axes[1].set_title("Permutation importance on test (drop in PR-AUC)")
plt.tight_layout()
plt.savefig("plots/gain_vs_permutation_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. SHAP values on the test set
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test_transformed)
if len(shap_values.shape) == 3:
    # random forest gives one value per class -> we keep class 1
    shap_values = shap_values[:, :, 1]
print("\nSHAP values shape:", shap_values.shape)

# the plots would show the scaled values (ex: age = -1.97), so we put back the real values
# for the numeric columns (they are the first columns of the transformed data)
X_test_engineered = pipeline.named_steps["engineering"].transform(X_test)
real_values = shap_values.data.copy()  # (the original array is read-only)
for j in range(len(numeric_features)):
    real_values[:, j] = X_test_engineered[numeric_features[j]].values
shap_values.data = real_values
print("Base value (average log-odds):", round(float(np.mean(shap_values.base_values)), 3))

mean_abs_shap = pd.Series(np.abs(shap_values.values).mean(axis=0), index=feature_names).sort_values(ascending=False)
print("\nTop 10 mean |SHAP| :")
print(mean_abs_shap.head(10).round(4))

# one-hot cuts a variable like "month" in 12 columns, so its importance is split.
# SHAP values are additive -> SHAP of the original variable = sum of the SHAP of its columns
shap_by_variable = {}
for var in numeric_features + categorical_features:
    cols = []
    for j in range(len(feature_names)):
        if feature_names[j] == var or feature_names[j].startswith(var + "_"):
            cols.append(j)
    shap_by_variable[var] = np.abs(shap_values.values[:, cols].sum(axis=1)).mean()
shap_by_variable = pd.Series(shap_by_variable).sort_values(ascending=False)
print("\nMean |SHAP| by original variable (one-hot columns added together):")
print(shap_by_variable.round(4))
shap_by_variable.to_csv("results/shap_by_variable.csv", header=["mean_abs_shap"])

plt.figure(figsize=(7, 5))
shap_by_variable.sort_values().plot(kind="barh")
plt.xlabel("mean |SHAP value| (log-odds)")
plt.title("SHAP importance by original variable (test set)")
plt.savefig("plots/shap_by_variable.png", dpi=150, bbox_inches="tight")
plt.close()

importance_table = pd.DataFrame({
    "gain_importance": gain_importance,
    "mean_abs_shap": mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=False)
importance_table.to_csv("results/shap_and_gain_importance.csv")
perm_importance.to_csv("results/permutation_importance.csv", header=["drop_in_pr_auc"])

# global plots
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title("SHAP beeswarm (test set) - log-odds scale")
plt.savefig("plots/shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

shap.plots.bar(shap_values, max_display=15, show=False)
plt.title("Mean |SHAP value| (test set)")
plt.savefig("plots/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# 4. local explanations for 4 clients
proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (proba >= threshold).astype(int)
y_true = y_test.values

tp = np.where((y_pred == 1) & (y_true == 1))[0]
fn = np.where((y_pred == 0) & (y_true == 1))[0]
fp = np.where((y_pred == 1) & (y_true == 0))[0]
tn = np.where((y_pred == 0) & (y_true == 0))[0]

clients = {
    "true_positive": tp[np.argmax(proba[tp])],   # subscriber with the highest score
    "false_negative": fn[np.argmin(proba[fn])],  # subscriber the model missed the most
    "false_positive": fp[np.argmax(proba[fp])],  # non-subscriber with the highest score
    "true_negative": tn[np.argmin(proba[tn])],   # non-subscriber with the lowest score
}

for case in clients:
    i = clients[case]
    print("\n=====", case, "- test row", i, "=====")
    print("predicted probability:", round(proba[i], 3), "| real label:", y_true[i])
    print(X_test.iloc[i].to_dict())
    shap.plots.waterfall(shap_values[i], max_display=10, show=False)
    plt.title(case + " - proba = " + str(round(proba[i], 3)) + ", real label = " + str(y_true[i]))
    plt.savefig("plots/shap_waterfall_" + case + ".png", dpi=150, bbox_inches="tight")
    plt.close()

print("\nSaved SHAP plots in plots/")
