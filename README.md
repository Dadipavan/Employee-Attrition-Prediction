# Employee Attrition Prediction — Final Year Project

## What this project is about

This project aims to build and evaluate machine learning models that predict employee attrition (whether an employee will leave the company) using the provided HR dataset `WA_Fn-UseC_-HR-Employee-Attrition.csv`.

Why it matters: predicting attrition helps HR teams proactively identify at-risk employees and take retention actions, reducing hiring costs and preserving institutional knowledge.

## Problem statement

Given historical employee records (demographics, job role, compensation, performance, and tenure), predict the binary target `Attrition` (Yes/No). The model should be accurate, interpretable, and robust to class imbalance. The output will be used to prioritize retention interventions.

Inputs: employee-level features (numeric and categorical) from the CSV.
Output: probability or label for `Attrition` (Yes/No).

Success criteria (examples):
- Baseline: logistic regression with cross-validated F1-score > 0.50 (adjust as project baseline).
- Production-quality candidate: tree-based model (Random Forest / XGBoost) with ROC AUC > 0.85 and recall for the `Yes` class high enough to be actionable (customize threshold with business constraints).

## Dataset summary

- File: `WA_Fn-UseC_-HR-Employee-Attrition.csv`
- Rows: 1,471 employee records (plus header)
- Columns: 35 columns. Header/column names:

```
Age, Attrition, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, EmployeeCount, EmployeeNumber, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, Over18, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
```

- Target variable: `Attrition` — categorical, values `Yes`/`No`.
- Notes: some columns appear constant (e.g., `EmployeeCount`, `Over18`, `StandardHours`) and `EmployeeNumber` is a unique identifier — these are typically dropped from modeling. Check for missing values and data quality during EDA.

## Data types (approximate / from header)

- Numerical (examples): `Age`, `DailyRate`, `DistanceFromHome`, `HourlyRate`, `MonthlyIncome`, `MonthlyRate`, `NumCompaniesWorked`, `PercentSalaryHike`, `StockOptionLevel`, `TotalWorkingYears`, `TrainingTimesLastYear`, `YearsAtCompany`, `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`, and several satisfaction/scale fields.
- Categorical: `Attrition`, `BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`, etc.

## Objectives

- Primary: build a model that predicts employee attrition (Yes/No) with strong generalization.
- Secondary: identify the most important features associated with attrition to provide interpretable insights for HR.

## Proposed approach

1. Exploratory Data Analysis (EDA)
   - Inspect distributions, missing values, outliers, and class balance for `Attrition`.
   - Visualize relationships between features and attrition (boxplots, histograms, correlation heatmap).

2. Data cleaning & feature engineering
   - Drop identifiers and constant columns (`EmployeeNumber`, `EmployeeCount`, `Over18`, `StandardHours`).
   - Convert categorical variables to appropriate encodings (one-hot for nominal, ordinal encoding where applicable).
   - Create engineered features if helpful (e.g., tenure buckets, average compensation ratios).
   - Handle missing values (impute medians for numeric, add 'Missing' category for categorical, or use model-based imputation).

3. Handle class imbalance
   - Check `Attrition` distribution. If imbalanced, consider class weights, resampling (SMOTE, ADASYN) or threshold tuning.

4. Modeling
   - Baseline: Logistic Regression (with scaling and class-weighted loss) to get a performance baseline and interpretability.
   - Tree-based models: Random Forest, Gradient Boosting (XGBoost / LightGBM) for stronger predictive performance.
   - Optional: simple feed-forward neural network if needed.
   - Use cross-validation (stratified K-fold) for robust evaluation and hyperparameter tuning (GridSearchCV or randomized search).

5. Evaluation & explainability
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC AUC. For business use, pay special attention to recall (catching those who will leave) and precision (avoiding wasted interventions).
   - Calibration: check predicted probabilities and calibrate if necessary.
   - Explainability: feature importance (tree SHAP values), partial dependence plots, and coefficients for linear models.

6. Final deliverables
   - Trained model(s) + saved artifacts (pickle or joblib), evaluation report, and a short dashboard or notebook to reproduce results.

## Suggested evaluation plan

- Train/validation/test split: stratified split (e.g., 70% train, 15% val, 15% test) or nested CV for model selection.
- Use stratified K-fold CV for hyperparameter search (k=5).
- Report final metrics on a held-out test set and provide confusion matrix, ROC curve, and precision-recall curve.

## Project artifacts (recommended)

- `notebooks/01_EDA.ipynb` — data exploration and cleaning steps.
- `notebooks/02_modeling.ipynb` — baseline modeling, feature selection, and cross-validation.
- `notebooks/03_final_model.ipynb` — final training and evaluation on hold-out test set.
- `src/` — scripts to preprocess data, train models, and produce predictions.
- `requirements.txt` — Python dependencies (pandas, numpy, scikit-learn, xgboost, imbalanced-learn, matplotlib, seaborn, shap, jupyter).
- `README.md` — (this file) overview and how to run.

## How to get started (commands for Windows PowerShell)

```
# create and activate a venv
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Or install minimal packages
pip install pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn jupyterlab

# start JupyterLab and open the EDA notebook
jupyter lab
```

If you don't have a `requirements.txt` yet, create one with the main libraries listed above.

## Next steps (short-term)

1. Create `notebooks/01_EDA.ipynb` and run initial EDA: target balance, missingness, and feature distributions.
2. Implement preprocessing pipeline in `src/preprocessing.py` (drop ID columns, impute, encode, scale). Create unit tests for the pipeline if possible.
3. Train a baseline logistic regression model and record cross-validated metrics.
4. Iterate with tree-based models and tune hyperparameters.

## Notes & assumptions

- Assumed dataset has ~1.5k rows as read from the CSV; final modeling decisions should respect this limited sample size (avoid overly complex models without regularization).
- Do a careful check for leakage (features that directly reveal `Attrition`) before training.

## Contacts / Attribution

This README was auto-generated as a starting point for the final year project. Expand the EDA and modeling sections as you progress.
