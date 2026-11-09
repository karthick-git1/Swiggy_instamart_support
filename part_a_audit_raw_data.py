import pandas as pd

#customer table

customers = pd.read_csv("data/raw/customers.csv")

print("\n--- Customers ---")
print("Shape:", customers.shape)
print("\nData types:")
print(customers.dtypes)

missing = pd.DataFrame({
    "Missing count": customers.isna().sum(),
    "Missing percentage": (customers.isna().mean() * 100).round(2),
})
print("\nMissing values by column:")
print(missing)
age_text = customers["age"].astype("string").str.strip()
age_numeric = pd.to_numeric(age_text, errors="coerce")
age_blank = age_text.isna() | age_text.eq("")

city_text = customers["city"].astype("string")
gender_text = customers["gender"].astype("string").str.strip()
gender_blank = gender_text.isna() | gender_text.eq("")

print("Blank age:", age_blank.sum())
print("Age outside 10–100:", ((age_numeric < 10) | (age_numeric > 100) | (age_numeric.isna() & ~age_blank)).sum())
print("Distinct raw city spellings:", customers["city"].nunique(dropna=True))
print("Distinct non-blank raw gender spellings:", gender_text[~gender_blank].nunique())
print("Blank gender:", gender_blank.sum())

# orders table 

orders = pd.read_csv("data/raw/orders.csv")

print("\n--- Orders ---")
print("Shape:", orders.shape)
print("\nData types:")
print(orders.dtypes)

missing = pd.DataFrame({
    "Missing count": orders.isna().sum(),
    "Missing percentage": (orders.isna().mean() * 100).round(2),
})
print("\nMissing values by column:")
print(missing)
print("Total rows:", len(orders))
print("Exact duplicate rows:", orders.duplicated().sum())
print("Rows with pending status:", orders["status"].eq("pending").sum())
print("Distinct order_id values:", orders["order_id"].nunique())
    # Values that pandas cannot parse as numbers, excluding blanks
amount_text = orders["order_amount"].astype("string").str.strip()
amount_numeric = pd.to_numeric(amount_text, errors="coerce")
#invalid_amount = amount_text.notna() & amount_text.ne("") & amount_numeric.isna()
blank_amount = amount_text.isna() | amount_text.eq("")
non_numeric_amount = ~blank_amount & amount_numeric.isna()

    # Delivery times
delivery_text = orders["delivery_minutes"].astype("string").str.strip()
delivery_numeric = pd.to_numeric(delivery_text, errors="coerce")
blank_delivery = delivery_text.isna() | delivery_text.eq("")

#print("Non-numeric order_amount entries:", invalid_amount.sum())
print("Non-numeric, non-blank order_amount entries:", non_numeric_amount.sum())
print("Negative delivery_minutes:", delivery_numeric.lt(0).sum())
print("Blank delivery times:", blank_delivery.sum())