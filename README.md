# Supervised Learning & Model Engineering — Project
**Course:** MSA-DATB09-04 (Ilana Sebag)  
**Dataset:** Option 2 — Bank Marketing (`OpenML data_id=1461`)

## Project Overview
End-to-end supervised learning pipeline for binary classification on the Bank Marketing dataset, predicting whether a client will subscribe to a term deposit (`y = 1`, only 11.7% of the clients).

## Repository Structure
The scripts are the main implementation. Each script has a notebook version (`*_notebook.ipynb`) with the outputs and plots, only for demonstration.

| Step | Script | What it does |
|---|---|---|
| 1 | `01_overall_analysis.py` | EDA, leakage audit (drop `duration`), stratified 80/20 dev/test split |
| 2 & 5 | `02_preprocessing_pipeline.py` | 3 engineered features + preprocessing `Pipeline` (impute, scale, one-hot) |
| 3 & 4 | `03_modeling_and_imbalance.py` | first comparison of the 3 models (LogReg, Random Forest, LightGBM) with 5-fold CV |
| 6 | `04_hyperparameter_tuning.py` | first Optuna tuning (LightGBM only) |
| 3 | `05_imbalance_remedies.py` | comparison of the imbalance remedies: none / threshold tuning / class weights / SMOTE |
| 6 & 7 | `06_tuning_with_mlflow.py` | Optuna tuning of the 3 models, every trial logged in MLflow |
| 4 | `07_final_comparison.py` | default vs tuned models, threshold chosen on CV, evaluation on the untouched test set |
| 8 | `08_explainability.py` | gain vs permutation importance, SHAP global + local explanations |
| 9 | `09_calibration.py` | reliability diagram, Brier score, ECE, Platt / isotonic recalibration |
| 10 | `10_fairness.py` | fairness audit by age group (Fairlearn) + ThresholdOptimizer mitigation |
| — | `predict.py` | scores new clients with the final model |
| — | `features.py` | shared code (feature engineering + preprocessing) used by the scripts of steps 3 to 10 |

Folders:
- `data/`: dev and test sets created by step 1 (the test set is only used for the final evaluations)
- `models/`: saved pipelines and best hyperparameters
- `results/`: csv/json tables with all the results
- `plots/`: all the figures used in the report

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (on Mac, LightGBM needs OpenMP: `brew install libomp`)
2. Run the scripts in this order:
   ```bash
   python 01_overall_analysis.py
   python 02_preprocessing_pipeline.py
   python 03_modeling_and_imbalance.py
   python 05_imbalance_remedies.py
   python 06_tuning_with_mlflow.py
   python 07_final_comparison.py
   python 08_explainability.py
   python 09_calibration.py
   python 10_fairness.py
   python predict.py
   ```
3. Look at the experiment tracking (all Optuna trials of the 3 models):
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
   then open http://127.0.0.1:5000 (a csv export of all the runs is in `results/mlflow_runs.csv`).

## Main Results
Main metric: **PR-AUC** (only 11.7% positives, a random model gets 0.117). The test set was only used once, at the end.

| Model | CV PR-AUC default | CV PR-AUC tuned | Test PR-AUC | Test ROC-AUC | Test F1 (threshold from CV) |
|---|---|---|---|---|---|
| Logistic Regression | 0.401 | 0.402 | 0.412 | 0.773 | 0.452 |
| Random Forest | 0.412 | 0.446 | 0.461 | 0.804 | 0.489 |
| **LightGBM** (final model) | 0.444 | **0.454** | **0.472** | **0.809** | **0.499** |

- **Imbalance**: class weights and SMOTE did not improve PR-AUC; we use a decision threshold chosen on out-of-fold predictions, and the class weight is tuned as a hyperparameter.
- **Explainability**: the most important variables are `contact`, `month`, `housing`, `balance`, `day` and `poutcome` (SHAP and permutation importance).
- **Calibration**: the tuned LightGBM over-estimates the probabilities (mean 0.21 vs real rate 0.117, ECE = 0.090). After Platt scaling: ECE = 0.009, Brier 0.0905 → 0.0804, same PR-AUC.
- **Fairness (age groups)**: with one threshold, TPR is 0.43 for 25-59 vs 0.80 for 60+. ThresholdOptimizer (equalized odds) reduces the equalized odds difference from 0.37 to 0.03, but F1 goes from 0.50 to 0.33.
