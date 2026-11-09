from pathlib import Path
import sqlite3
import subprocess
import sys

import pandas as pd

base_dir = Path(__file__).resolve().parent

products = pd.read_csv(base_dir / "data" / "raw" / "products.csv")

# Run each cleaning script and stop if either one fails
subprocess.run(
    [sys.executable, str(base_dir / "part_a_clean_customers.py")],
    check=True,
    cwd=base_dir,
)
subprocess.run(
    [sys.executable, str(base_dir / "part_a_clean_orders.py")],
    check=True,
    cwd=base_dir,
)

clean_dir = base_dir / "data" / "clean"
clean_dir.mkdir(parents=True, exist_ok=True)
customers = pd.read_csv(clean_dir / "customers.csv")
orders = pd.read_csv(clean_dir / "orders.csv")

# Write the clean CSV files
products.to_csv(clean_dir / "products.csv", index=False)

# Format order_ts as YYYY-MM-DD HH:MM:SS for SQLite
orders_for_db = orders.copy()
orders_for_db["order_ts"] = (
    pd.to_datetime(orders_for_db["order_ts"], errors="coerce")
    .dt.strftime("%Y-%m-%d %H:%M:%S")
)

# Write the clean tables to SQLite
db_path = base_dir / "data" / "instamart.db"
with sqlite3.connect(db_path) as connection:
    customers.to_sql("customers", connection, if_exists="replace", index=False)
    products.to_sql("products", connection, if_exists="replace", index=False)
    orders_for_db.to_sql("orders", connection, if_exists="replace", index=False)

print("Cleaning complete and data loaded into SQLite.")

# Validate the clean tables


# Treat blank text cells as missing for validation
age = pd.to_numeric(customers["age"].replace("", pd.NA), errors="coerce")
delivery = pd.to_numeric(orders["delivery_minutes"].replace("", pd.NA), errors="coerce")
amount = pd.to_numeric(orders["order_amount"].replace("", pd.NA), errors="coerce")

def check(condition, description):
    assert condition, f"FAILED: {description}"
    print(f"PASSED: {description}")

# Customer checkpoints
check(customers["city"].nunique() == 6, "Customer data contains exactly 6 unique cities")
check(set(customers["gender"].dropna().unique()) <= {"male", "female", "unknown"}, \
    "All non-missing gender values are male, female, or unknown")
check(not age.dropna().lt(10).any(), "No customer age is below 10")
check(not age.dropna().gt(100).any(), "No customer age is above 100")

# Order checkpoints
check(orders["order_id"].is_unique, "Each order has a unique order_id")
check(orders.duplicated().sum() == 0, "There are no exact duplicate order rows")
check(orders["order_ts"].astype(str).str.match(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
).all(), "All order_ts values use YYYY-MM-DD HH:MM:SS format")

cancelled = orders["status"].eq("cancelled")
check(delivery[cancelled].isna().all(), \
    "Cancelled orders have no delivery_minutes")

check(not delivery.dropna().lt(0).any(), "All non-missing delivery_minutes are non-negative")
check(not amount.dropna().lt(0).any(), "All non-missing order_amount values are non-negative")

print("All validation checks passed.")
print("ALL CHECKPOINTS PASSED") 