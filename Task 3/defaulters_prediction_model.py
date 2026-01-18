import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ======================================================
# 1. LOAD DATA
# ======================================================

CSV_PATH = r"c:\Users\USER01\Desktop\Projects\JPMorgan\Task 3\Task 3 and 4_Loan_Data.csv"

df = pd.read_csv(CSV_PATH)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# ======================================================
# 2. DEFINE FEATURES & TARGET
# ======================================================

FEATURES = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score"
]

TARGET = "default"

X = df[FEATURES]
y = df[TARGET]

# ======================================================
# 3. TRAIN / TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ======================================================
# 4. GRADIENT BOOSTING MODEL
# ======================================================

model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("gb", GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# ======================================================
# 5. MODEL EVALUATION
# ======================================================

y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nGradient Boosting AUC: {auc:.4f}")

# ======================================================
# 6. EXPECTED LOSS FUNCTION
# ======================================================

RECOVERY_RATE = 0.10
LGD = 1 - RECOVERY_RATE

def expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score
):
    """
    Returns (PD, Expected Loss)
    """

    features = pd.DataFrame([{
        "credit_lines_outstanding": credit_lines_outstanding,
        "loan_amt_outstanding": loan_amt_outstanding,
        "total_debt_outstanding": total_debt_outstanding,
        "income": income,
        "years_employed": years_employed,
        "fico_score": fico_score
    }])

    pd_estimate = model.predict_proba(features)[0, 1]
    el = pd_estimate * loan_amt_outstanding * LGD

    return pd_estimate, el

# ======================================================
# 7. SAMPLE TEST
# ======================================================

pd_est, el_est = expected_loss(
    credit_lines_outstanding=3,
    loan_amt_outstanding=50000,
    total_debt_outstanding=80000,
    income=75000,
    years_employed=6,
    fico_score=680
)

print("\nSample Borrower:")
print(f"Probability of Default (PD): {pd_est:.2%}")
print(f"Expected Loss (EL): ${el_est:,.2f}")
