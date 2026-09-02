"""
Generates synthetic employee data + a PromotionReady label.
Purely synthetic - no dependency on any real HR system.
Independent dataset/schema from the attrition model on purpose,
since in a real platform these would come from different model owners.
"""
import numpy as np
import pandas as pd

np.random.seed(7)
N = 3000

departments = ["Engineering", "Sales", "Support", "HR", "Finance", "Marketing"]
levels = ["Junior", "Mid", "Senior"]

df = pd.DataFrame({
    "employee_id": range(1, N + 1),
    "department": np.random.choice(departments, N),
    "current_level": np.random.choice(levels, N),
    "tenure_in_role_months": np.random.randint(1, 60, N),
    "avg_performance_rating_last_4q": np.round(np.random.uniform(1, 5, N), 2),
    "goals_completed_pct": np.round(np.random.uniform(0, 100, N), 1),
    "peer_review_score": np.round(np.random.uniform(1, 10, N), 2),
    "training_hours_last_year": np.random.randint(0, 80, N),
    "cross_team_projects": np.random.poisson(1, N),
    "manager_recommendation": np.random.choice([0, 1], N, p=[0.6, 0.4]),
})

score = (
    -3.0
    + 0.9 * (df["avg_performance_rating_last_4q"] - 3)
    + 0.02 * (df["goals_completed_pct"] - 50)
    + 0.35 * (df["peer_review_score"] - 5)
    + 0.02 * df["training_hours_last_year"]
    + 0.4 * df["cross_team_projects"]
    + 1.2 * df["manager_recommendation"]
    + 0.015 * (df["tenure_in_role_months"] - 24)
    + np.random.normal(0, 0.6, N)
)
prob = 1 / (1 + np.exp(-score))
df["PromotionReady"] = (np.random.rand(N) < prob).astype(int)

df.to_csv("app/models/promotion_data.csv", index=False)
print(df["PromotionReady"].value_counts(normalize=True))
print(f"Saved {len(df)} rows to app/models/promotion_data.csv")