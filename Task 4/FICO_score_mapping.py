import pandas as pd
import numpy as np


CSV_PATH = r"C:\Users\USER01\Desktop\Projects\JPMorgan\Task 4\Task 3 and 4_Loan_Data.csv"
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()


# Bin FICO into 5-point bins
df["fico_bin"] = (df["fico_score"] // 5) * 5

grouped = df.groupby("fico_bin")["default"].agg(["sum", "count"]).reset_index()
grouped.rename(columns={"sum": "defaults", "count": "total"}, inplace=True)

B = len(grouped)

defaults = grouped["defaults"].values
totals = grouped["total"].values
fico_vals = grouped["fico_bin"].values

# Cumulative sums
cum_defaults = np.cumsum(defaults)
cum_totals = np.cumsum(totals)


def interval_loglik(i, j):
    k = cum_defaults[j - 1] - (cum_defaults[i - 1] if i > 0 else 0)
    n = cum_totals[j - 1] - (cum_totals[i - 1] if i > 0 else 0)

    if n == 0 or k == 0 or k == n:
        return 0.0

    p = k / n
    return k * np.log(p) + (n - k) * np.log(1 - p)

LL = np.zeros((B, B))
for i in range(B):
    for j in range(i, B):
        LL[i, j] = interval_loglik(i, j + 1)


def optimal_buckets(K):
    dp = np.full((K, B), -np.inf)
    prev = np.zeros((K, B), dtype=int)

    dp[0] = LL[0]

    for k in range(1, K):
        for j in range(B):
            for i in range(k, j + 1):
                val = dp[k - 1, i - 1] + LL[i, j]
                if val > dp[k, j]:
                    dp[k, j] = val
                    prev[k, j] = i

    # Backtrack
    boundaries = []
    j = B - 1
    for k in range(K - 1, -1, -1):
        i = prev[k, j]
        boundaries.append((i, j))
        j = i - 1

    boundaries.reverse()
    return boundaries


def build_rating_map(K):
    buckets = optimal_buckets(K)
    rating_map = []

    for idx, (i, j) in enumerate(buckets):
        n = totals[i:j + 1].sum()
        k = defaults[i:j + 1].sum()
        pd_rate = k / n if n > 0 else 0  # renamed variable

        rating_map.append({
            "rating": idx + 1,          # lower = better credit
            "fico_min": fico_vals[i],
            "fico_max": fico_vals[j] + 4,
            "pd": pd_rate,
            "count": n
        })

    return pd.DataFrame(rating_map)



rating_table = build_rating_map(K=10)
print("\nOptimal FICO Rating Map:\n")
print(rating_table)
