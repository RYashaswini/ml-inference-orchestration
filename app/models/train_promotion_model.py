"""
Trains the Promotion Readiness model on the synthetic dataset
and saves the fitted pipeline (preprocessing + model) as one artifact.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("app/models/promotion_data.csv")

FEATURES = [
    "department", "current_level", "tenure_in_role_months",
    "avg_performance_rating_last_4q", "goals_completed_pct",
    "peer_review_score", "training_hours_last_year",
    "cross_team_projects", "manager_recommendation",
]
TARGET = "PromotionReady"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical = ["department", "current_level"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ],
    remainder="passthrough",
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )),
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

joblib.dump(pipeline, "app/models/artifacts/promotion_model.joblib")
print("Saved model to app/models/artifacts/promotion_model.joblib")