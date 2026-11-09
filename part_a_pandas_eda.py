from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parent
clean_dir = base_dir / "data" / "clean"
orders = pd.read_csv(clean_dir / "orders.csv")
customers = pd.read_csv(clean_dir / "customers.csv")
products = pd.read_csv(clean_dir / "products.csv")

completed = orders.loc[orders["status"].eq("completed")].drop(
    columns="category",
    errors="ignore",
).copy()

# Rebuild Q2's three-table join
joined = (
    completed
    .merge(customers[["customer_id", "city"]], on="customer_id", how="left")
    .merge(products[["product_id", "category"]], on="product_id", how="left")
)

# Q2 pivot table, including row, column, and grand totals
pivot = pd.pivot_table(
    joined,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    margins=True,
    margins_name="All",
    fill_value=0,
)

sql_total = completed["order_amount"].sum()
pivot_total = pivot.loc["All", "All"]

print("Pivot grand total:", round(pivot_total, 2))
print("SQL completed revenue:", round(sql_total, 2))
print("Totals match:", abs(pivot_total - sql_total) < 0.005)

# Per-city named aggregation
city_summary = (
    joined.groupby("city")
    .agg(
        sum_order_amount=("order_amount", "sum"),
        mean_order_amount=("order_amount", "mean"),
        order_count=("order_amount", "count"),
    )
    .round(2)
)
print(city_summary)

# Distribution statistics and 1.5 × IQR outlier fences
amount = completed["order_amount"].dropna()
q1 = amount.quantile(0.25)
median = amount.median()
q3 = amount.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

print({
    "mean": amount.mean(),
    "median": median,
    "skewness": amount.skew(),
    "Q1": q1,
    "Q3": q3,
    "lower_fence": lower_fence,
    "upper_fence": upper_fence,
    "outliers_below": (amount < lower_fence).sum(),
    "outliers_above": (amount > upper_fence).sum(),
})

# Pearson correlation matrix among completed orders
correlation_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes",
]
corr = completed[correlation_columns].corr()
print(corr.round(3))

# Find the variable pair with the largest absolute correlation
upper_triangle = corr.where(
    pd.DataFrame(
        [[i < j for j in range(len(corr.columns))] for i in range(len(corr.columns))],
        index=corr.index,
        columns=corr.columns,
    )
)
strongest_pair = upper_triangle.abs().stack().idxmax()
print(
    "Largest absolute correlation:",
    strongest_pair,
    round(corr.loc[strongest_pair[0], strongest_pair[1]], 3),
)