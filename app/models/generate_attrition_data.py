"""
Generates synthetic employee data + an Attrition label.
Purely synthetic - no dependency on any real HR system.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000

departments = ["Engineering", "Sales", "Support", "HR", "Finance", "Marketing"]
work_modes = ["Onsite", "Hybrid", "Remote"]

df = pd.DataFrame({
    "employee_id": range(1, N + 1),
    "age": np.random.randint(21, 60, N),
    "department": np.random.choice(departments, N),
    "work_mode": np.random.choice(work_modes, N),
    "tenure_months": np.random.randint(1, 180, N),
    "monthly_salary": np.random.randint(30000, 200000, N),
    "months_since_last_hike": np.random.randint(0, 36, N),
    "productivity_score": np.round(np.random.uniform(1, 10, N), 2),
    "leaves_last_90_days": np.random.poisson(2, N),
    "manager_change_count": np.random.poisson(0.5, N),
    "performance_rating": np.random.randint(1, 6, N),  # 1-5
})

# Build attrition probability from a mix of realistic-ish signals
risk = (
    0.03 * (df["months_since_last_hike"] > 18).astype(int)
    + 0.03 * (df["performance_rating"] <= 2).astype(int)
    + 0.02 * (df["leaves_last_90_days"] > 4).astype(int)
    + 0.02 * (df["manager_change_count"] >= 2).astype(int)
    + 0.02 * (df["tenure_months"] < 6).astype(int)
    + 0.015 * (df["productivity_score"] < 4).astype(int)
    + np.random.normal(0, 0.05, N)  # noise
)
risk = risk.clip(0, 1)
df["Attrition"] = (np.random.rand(N) < (0.15 + risk)).astype(int)

df.to_csv("app/models/attrition_data.csv", index=False)
print(df["Attrition"].value_counts(normalize=True))
print(f"Saved {len(df)} rows to app/models/attrition_data.csv")