from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parent
orders = pd.read_csv(base_dir / "data" / "raw" / "orders.csv")
products = pd.read_csv(base_dir / "data" / "raw" / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    errors="coerce"
)
orders = orders.drop_duplicates()
print("Rows remaining after exact duplicate removal:", len(orders))

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")

orders = (
    orders.sort_values("order_ts")
          .drop_duplicates(subset="order_id", keep="last")
          .reset_index(drop=True)
)
print("Rows remaining after cleaning duplicate orders:", len(orders))


orders = orders.merge(
    products[["product_id", "price_inr"]],
    on="product_id",
    how="left"
)
# Convert existing amounts to numbers; unparseable values become missing
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)
# Calculate a replacement amount
calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

# Fill missing amounts with the calculated amount
orders["order_amount"] = orders["order_amount"].fillna(calculated_amount) 

#convert delivery_minutes to missing
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA

# Add category from products if it isn't already in orders
if "category" not in orders.columns:
    orders = orders.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left"
    )

completed = orders["status"].eq("completed")

# For each completed order, get its category's median delivery time
category_median = (
    orders.loc[completed]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

# Fill missing delivery times for completed orders only
orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed, "delivery_minutes"].fillna(category_median)
)

cancelled = orders["status"].eq("cancelled")
orders.loc[cancelled, "delivery_minutes"] = pd.NA

completed = orders["status"].eq("completed")
cancelled = orders["status"].eq("cancelled")
pending = orders["status"].eq("pending")

print("Completed orders:", completed.sum())
print("Cancelled orders:", cancelled.sum())
print("Pending orders:", pending.sum())

print("Missing order_amount:", orders["order_amount"].isna().sum())
print("Missing delivery_minutes among completed:", orders.loc[completed, "delivery_minutes"].isna().sum())
print("Missing delivery_minutes among cancelled:", orders.loc[cancelled, "delivery_minutes"].isna().sum())
print("Missing delivery_minutes among pending:", orders.loc[pending, "delivery_minutes"].isna().sum())

print("Total order_amount for completed orders:", orders.loc[completed, "order_amount"].sum())

clean_dir = base_dir / "data" / "clean"
clean_dir.mkdir(parents=True, exist_ok=True)
orders.to_csv(clean_dir / "orders.csv", index=False)