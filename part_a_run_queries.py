import re
import sqlite3
from pathlib import Path

import pandas as pd

sql_text = Path("sql/queries.sql").read_text(encoding="utf-8")
queries = re.split(r"(?m)^-- Q[1-7]\s*$", sql_text)
queries = [query.strip() for query in queries if query.strip()]

if len(queries) != 7:
    raise ValueError(f"Expected 7 queries, found {len(queries)}")

with sqlite3.connect("data/instamart.db") as connection:
    results = {}
    for number, query in enumerate(queries, start=1):
        results[number] = pd.read_sql_query(query, connection)
        print(f"\nQ{number} ({len(results[number])} rows)")
        print(results[number].to_string(index=False))