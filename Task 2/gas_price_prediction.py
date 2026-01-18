import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# LOAD AND PREPARE NATURAL GAS PRICE DATA

CSV_PATH = r"C:\Users\USER01\Desktop\Projects\JPMorgan\Task 2\Nat_Gas.csv"

df = pd.read_csv(CSV_PATH)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Rename columns if needed (robust to vendor formats)
if "date" not in df.columns:
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
if "price" not in df.columns:
    df.rename(columns={df.columns[1]: "price"}, inplace=True)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Feature engineering
df["month"] = df["date"].dt.month
df["timeindex"] = np.arange(len(df))
df["logprice"] = np.log(df["price"])

# SEASONALITY EXTRACTION

monthly_avg = df.groupby("month")["logprice"].mean()
overall_avg = df["logprice"].mean()
seasonality = monthly_avg - overall_avg

# TREND MODEL (LINEAR REGRESSION)

df["deseasonalized"] = df.apply(
    lambda x: x["logprice"] - seasonality.loc[x["month"]],
    axis=1
)

trend_slope, trend_intercept = np.polyfit(
    df["timeindex"], df["deseasonalized"], 1
)

# PRICE ESTIMATION FUNCTION

def estimate_price(date):
    """
    Estimate natural gas price for any past or future date.
    """
    date = pd.to_datetime(date)
    month = date.month

    time_index = (date - df["date"].iloc[0]).days / 30.44

    log_price = (
        trend_intercept
        + trend_slope * time_index
        + seasonality.loc[month]
    )

    return float(np.exp(log_price))

# STORAGE CONTRACT PRICING FUNCTION

def price_storage_contract(
    injection_schedule,
    withdrawal_schedule,
    max_storage,
    injection_rate,
    withdrawal_rate,
    storage_cost_per_day,
    price_function
):
    """
    Deterministic intrinsic valuation of a natural gas storage contract.
    """

    events = []

    for d, v in injection_schedule:
        events.append((pd.to_datetime(d), "inject", v))

    for d, v in withdrawal_schedule:
        events.append((pd.to_datetime(d), "withdraw", v))

    events.sort(key=lambda x: x[0])

    stored_volume = 0.0
    contract_value = 0.0
    last_date = events[0][0]

    for date, action, volume in events:

        # Storage cost accrual
        days = (date - last_date).days
        contract_value -= stored_volume * storage_cost_per_day * days

        price = price_function(date)

        if action == "inject":
            if volume > injection_rate:
                raise ValueError("Injection rate exceeded")
            if stored_volume + volume > max_storage:
                raise ValueError("Storage capacity exceeded")

            contract_value -= volume * price
            stored_volume += volume

        elif action == "withdraw":
            if volume > withdrawal_rate:
                raise ValueError("Withdrawal rate exceeded")
            if volume > stored_volume:
                raise ValueError("Insufficient gas in storage")

            contract_value += volume * price
            stored_volume -= volume

        last_date = date

    return contract_value

# TEST CASE (REQUIRED BY TASK)

injection_schedule = [
    ("2024-06-01", 500_000),
    ("2024-07-01", 500_000)
]

withdrawal_schedule = [
    ("2024-12-01", 500_000),
    ("2025-01-01", 500_000)
]

max_storage = 1_000_000
injection_rate = 500_000
withdrawal_rate = 500_000
storage_cost_per_day = 100_000 / 30  # $100k per month

contract_value = price_storage_contract(
    injection_schedule=injection_schedule,
    withdrawal_schedule=withdrawal_schedule,
    max_storage=max_storage,
    injection_rate=injection_rate,
    withdrawal_rate=withdrawal_rate,
    storage_cost_per_day=storage_cost_per_day,
    price_function=estimate_price
)

print(f"\nEstimated storage contract value: ${contract_value:,.2f}")

# VISUALIZATION

future_dates = pd.date_range(
    start=df["date"].iloc[-1],
    periods=24,
    freq="M"
)

future_prices = [estimate_price(d) for d in future_dates]

plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["price"], label="Historical", marker="o")
plt.plot(future_dates, future_prices, label="Model Estimate", linestyle="--")
plt.title("Natural Gas Price Model (Historical + Extrapolation)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
