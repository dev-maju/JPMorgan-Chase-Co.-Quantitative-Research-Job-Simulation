import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# LOAD DATA

CSV_PATH = r"C:\Users\USER01\Desktop\Projects\JPMorgan\Task 1\Nat_Gas.csv"  # <-- change if needed

df = pd.read_csv(CSV_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

df["Month"] = df["Date"].dt.month
df["TimeIndex"] = np.arange(len(df))

# VISUALIZE HISTORICAL DATA

plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Price"], marker="o")
plt.title("Monthly Natural Gas Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# EXTRACT SEASONALITY

df["LogPrice"] = np.log(df["Price"])

monthly_avg = df.groupby("Month")["LogPrice"].mean()
overall_avg = df["LogPrice"].mean()
seasonality = monthly_avg - overall_avg

# DESEASONALIZE & FIT TREND

df["Deseasonalized"] = df.apply(
    lambda x: x["LogPrice"] - seasonality.loc[x["Month"]],
    axis=1
)

trend_coeffs = np.polyfit(df["TimeIndex"], df["Deseasonalized"], 1)
trend_slope, trend_intercept = trend_coeffs

# PRICE ESTIMATION FUNCTION

def estimate_price(input_date):
    """
    Estimate natural gas price for any past or future date
    """

    input_date = pd.to_datetime(input_date)
    month = input_date.month

    time_index = (input_date - df["Date"].iloc[0]).days / 30.44

    log_price_trend = trend_intercept + trend_slope * time_index

    seasonal_component = seasonality.loc[month]

    log_price = log_price_trend + seasonal_component
    return float(np.exp(log_price))

# EXTRAPOLATE 1 YEAR AHEAD

future_dates = pd.date_range(
    start=df["Date"].iloc[-1] + pd.offsets.MonthEnd(1),
    periods=12,
    freq="M"
)

future_prices = [estimate_price(d) for d in future_dates]

future_df = pd.DataFrame({
    "Date": future_dates,
    "EstimatedPrice": future_prices
})

# PLOT HISTORICAL + FORECAST

plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Price"], label="Historical", marker="o")
plt.plot(future_df["Date"], future_df["EstimatedPrice"],
         label="1Y Extrapolation", marker="o", linestyle="--")

plt.title("Natural Gas Price: History + 1Y Extrapolation")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# SAMPLE USAGE

test_date = "2023-06-15"
print(f"Estimated price on {test_date}: {estimate_price(test_date):.2f}")
