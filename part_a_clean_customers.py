from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parent

#customer table cleaning

#cities cleaning
customers = pd.read_csv(base_dir / "data" / "raw" / "customers.csv")
customers["city"] = customers["city"].str.strip().str.title()
print("List of cities before sorting/cleaning:")
print(customers["city"].values)
print(customers["city"].value_counts())
print("List of distinct cities after sorting/cleaning:")
print("Distinct cities:", customers["city"].nunique())
print(sorted(customers["city"].dropna().unique()))

#gender cleaning
gender_text = customers["gender"].astype("string").str.strip().str.lower()
gender_map = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female",
    "": "unknown",
}
customers["gender"] = gender_text.map(gender_map).fillna("unknown")
print("List of distinct genders after cleaning:")
print(sorted(customers["gender"].dropna().unique()))
print(customers["gender"].value_counts())

#age cleaning
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[~customers["age"].between(10, 100), "age"] = pd.NA
city_medians = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_medians)
print("List of distinct ages after cleaning:")
print(sorted(customers["age"].dropna().unique()))
print(customers["age"].value_counts())
print("Missing ages:", customers["age"].isna().sum())
min_age = customers["age"].min()
max_age = customers["age"].max()
print(f"Every non-missing age is between {min_age} and {max_age}.")


customers["signup_date"] = pd.to_datetime(customers["signup_date"])
print("List of distinct signup dates after cleaning:")
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)
print(sorted(customers["signup_date"].dropna().unique()))

clean_dir = base_dir / "data" / "clean"
clean_dir.mkdir(parents=True, exist_ok=True)
customers.to_csv(clean_dir / "customers.csv", index=False)