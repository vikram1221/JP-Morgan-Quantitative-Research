import pandas as pd
import numpy as np
import statsmodels.api as sm


# Load data
df = pd.read_csv("Nat_Gas.csv")
df["Dates"] = pd.to_datetime(df["Dates"])
df["Prices"] = pd.to_numeric(df["Prices"])

df = df.sort_values("Dates").set_index("Dates")

# Monthly time series (month-end prices)
ts = df["Prices"].asfreq("M")

# Seasonal regression
def fit_seasonal_regression(series):
    d = series.dropna().to_frame(name="price")
    d["t"] = np.arange(len(d))
    d["month"] = d.index.month

    month_dummies = pd.get_dummies(d["month"], prefix="m", drop_first=True)

    X = pd.concat(
        [
            pd.Series(1.0, index=d.index, name="const"),
            d["t"],
            month_dummies
        ],
        axis=1
    ).astype(float)

    y = d["price"].astype(float)

    model = sm.OLS(y, X).fit()
    return model, X.columns

model, exog_cols = fit_seasonal_regression(ts)

# Forecast next 12 months
def forecast_next_12_months(series, model, exog_cols):
    last_date = series.dropna().index[-1]
    future_idx = pd.date_range(
        last_date + pd.offsets.MonthEnd(1),
        periods=12,
        freq="M"
    )

    t_start = len(series.dropna())
    future_t = np.arange(t_start, t_start + 12)

    future_df = pd.DataFrame(index=future_idx)
    future_df["t"] = future_t
    future_df["month"] = future_idx.month

    month_dummies = pd.get_dummies(
        future_df["month"], prefix="m", drop_first=True
    )

    Xf = pd.concat(
        [
            pd.Series(1.0, index=future_idx, name="const"),
            future_df["t"],
            month_dummies
        ],
        axis=1
    )

    Xf = Xf.reindex(columns=exog_cols, fill_value=0).astype(float)

    return pd.Series(model.predict(Xf), index=future_idx)

future_prices = forecast_next_12_months(ts, model, exog_cols)

# Price estimation (interpolation)
def estimate_price(date_str):
    target = pd.to_datetime(date_str)

    hist = ts.dropna()
    extended = pd.concat([hist, future_prices]).sort_index()

    if target < extended.index.min() or target > extended.index.max():
        raise ValueError("Date out of supported range")

    extended.loc[target] = np.nan
    extended = extended.sort_index().interpolate(method="time")

    return float(extended.loc[target])

# Storage contract pricing
def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    injection_rate,
    withdrawal_rate,
    max_volume,
    storage_cost_per_day
):
    if not injection_dates and not withdrawal_dates:
        return 0.0

    storage = 0.0
    value = 0.0
    events = []

    for d in injection_dates:
        events.append((pd.to_datetime(d), "inject"))
    for d in withdrawal_dates:
        events.append((pd.to_datetime(d), "withdraw"))

    events.sort(key=lambda x: x[0])
    last_date = events[0][0]

    for date, action in events:
        days_elapsed = (date - last_date).days
        value -= storage * storage_cost_per_day * days_elapsed

        price = estimate_price(date.strftime("%Y-%m-%d"))

        if action == "inject":
            volume = min(injection_rate, max_volume - storage)
            storage += volume
            value -= volume * price

        elif action == "withdraw":
            volume = min(withdrawal_rate, storage)
            storage -= volume
            value += volume * price

        last_date = date

    return value

# Test
contract_value = price_storage_contract(
    injection_dates=["2024-04-01", "2024-05-01"],
    withdrawal_dates=["2024-11-01", "2024-12-01"],
    injection_rate=1000,
    withdrawal_rate=1000,
    max_volume=2000,
    storage_cost_per_day=0.02
)

print("Contract value:", round(contract_value, 2))
