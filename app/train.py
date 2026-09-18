"""
Train UPI fraud detection models.

Usage:
    python train.py

Outputs:
    ../models/fraud_pipeline.joblib   -> full sklearn Pipeline (preprocess + model)
    ../outputs/eda_*.png              -> EDA charts
    ../outputs/model_comparison.txt   -> metrics for all models tried
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, RocCurveDisplay,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

sys.path.append(".")
from features import build_features, get_feature_lists, inject_realistic_noise

DATA_PATH = "../data/upi_transactions.csv"
MODEL_PATH = "../models/fraud_pipeline.joblib"
OUT_DIR = "../outputs"

# ---------------------------------------------------------------------
# 1. Load + build features
# ---------------------------------------------------------------------
print("Loading data...")
raw = pd.read_csv(DATA_PATH)
print(f"Raw shape: {raw.shape}, fraud rate: {raw['is_fraud'].mean():.3%}")

feat_df = build_features(raw)
feat_df = inject_realistic_noise(feat_df, noise_rate=0.08)
print("Injected realistic noise into deterministic risk flags (8% flip rate) "
      "so the model must learn from imperfect signals, like real fraud systems do.")
y = feat_df["is_fraud"]
X = feat_df.drop(columns=["is_fraud"])
numeric_cols, categorical_cols = get_feature_lists(feat_df)
print(f"Feature matrix: {X.shape[1]} columns "
      f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")

# ---------------------------------------------------------------------
# 2. EDA charts (saved as PNGs)
# ---------------------------------------------------------------------
print("Generating EDA charts...")
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
raw["is_fraud"].value_counts().rename({0: "Legit", 1: "Fraud"}).plot(
    kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
axes[0].set_title("Class balance")
axes[0].set_ylabel("Count")

sns.kdeplot(data=raw, x="amount", hue="is_fraud", common_norm=False, fill=True, ax=axes[1])
axes[1].set_title("Transaction amount distribution by fraud label")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_overview.png", dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4.5))
raw.groupby("transaction_time_of_day")["is_fraud"].mean().plot(kind="bar", ax=ax, color="#C44E52")
ax.set_title("Fraud rate by hour of day")
ax.set_xlabel("Hour")
ax.set_ylabel("Fraud rate")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_hourly_fraud.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

# ---------------------------------------------------------------------
# 4. Train & compare models (with SMOTE to handle imbalance)
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    ),
}

results = []
best_name, best_pipeline, best_auc = None, None, -1

with open(f"{OUT_DIR}/model_comparison.txt", "w") as report_file:
    for name, model in models.items():
        pipe = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
        cm = confusion_matrix(y_test, y_pred)

        block = (
            f"\n{'='*60}\n{name}\n{'='*60}\n"
            f"ROC-AUC: {auc:.4f}\n\n{report}\nConfusion matrix:\n{cm}\n"
        )
        print(block)
        report_file.write(block)

        results.append((name, auc, pipe))
        if auc > best_auc:
            best_auc = auc

# Among models within 0.001 AUC of the best, prefer Random Forest: it ties on
# accuracy here but gives us feature_importances_ for explainability, which
# matters more for a fraud analyst than a marginal metric difference.
tied = [(n, a, p) for n, a, p in results if best_auc - a < 0.001]
preference_order = ["Random Forest", "XGBoost", "Logistic Regression"]
tied.sort(key=lambda t: preference_order.index(t[0]))
best_name, best_auc, best_pipeline = tied[0]

print(f"\nBest model: {best_name} (ROC-AUC = {best_auc:.4f})")

# ---------------------------------------------------------------------
# 5. Save best pipeline (preprocessing + SMOTE + model, all in one)
# ---------------------------------------------------------------------
# Sensible defaults for any feature the app form doesn't ask the user for directly
defaults = {}
for col in numeric_cols:
    defaults[col] = float(X[col].median())
for col in categorical_cols:
    defaults[col] = X[col].mode().iloc[0]

joblib.dump({
    "pipeline": best_pipeline,
    "model_name": best_name,
    "feature_columns": list(X.columns),
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "defaults": defaults,
}, MODEL_PATH)
print(f"Saved best pipeline to {MODEL_PATH}")

# ROC comparison chart
plt.figure(figsize=(6, 5))
for name, auc, pipe in results:
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=plt.gca())
plt.title("ROC curves — model comparison")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_comparison.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------
# 6. Feature importance (explainability) for the deployed model
# ---------------------------------------------------------------------
fitted_model = best_pipeline.named_steps["model"]
if hasattr(fitted_model, "feature_importances_"):
    ohe = best_pipeline.named_steps["preprocess"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
    all_feature_names = numeric_cols + cat_feature_names

    importances = pd.Series(fitted_model.feature_importances_, index=all_feature_names)
    top15 = importances.sort_values(ascending=False).head(15)

    plt.figure(figsize=(7, 6))
    top15.sort_values().plot(kind="barh", color="#55A868")
    plt.title(f"Top 15 fraud signals — {best_name}")
    plt.xlabel("Feature importance")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=130)
    plt.close()
    print("\nTop 10 fraud signals:")
    print(top15.sort_values(ascending=False).head(10))

print("Done.")
