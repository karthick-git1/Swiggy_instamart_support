Part A turns the messy exports into a trusted analytical database. Part B uses that database to segment customers and to predict who is likely to lapse. Part C reads support tickets, protects customer privacy, and routes each ticket using the segments from Part B. Part D builds a policy question-answering assistant and a refund agent that queries the database from Part A and reuses the policy retrieval built in the same Part. Every Part consumes what the previous Parts produced 



Reproduce:

1.	D:\Karthick\Masai\Projects\Swiggy_instamart_support> C:\Users\skjee\pythonProject2\Scripts\python.exe -m venv .venv

2.	PS D:\Karthick\Masai\Projects\Swiggy_instamart_support> .\.venv\Scripts\Activate.ps1

3.	python -m pip freeze > requirements.txt

4.	(.venv) PS D:\Karthick\Masai\Projects\Swiggy_instamart_support> python .\generate_data.py
generated: customers=1200 products=40 orders_raw=13084 tickets=164

5.	 Audit the raw data
(.venv) PS D:\Karthick\Masai\Projects\Swiggy_instamart_support> python .\part_a_audit_raw_data.py 

6.	Clean customers

(.venv) PS D:\Karthick\Masai\Projects\Swiggy_instamart_support> python .\part_a_clean_customers.py

7.	Clean orders
(.venv) PS D:\Karthick\Masai\Projects\Swiggy_instamart_support> python .\part_a_clean_orders.py


8.	Store and validate:
(.venv) PS D:\Karthick\Masai\Projects\Swiggy_instamart_support> python .\part_a_store_validate.py

9.	SQL analysis
(.venv) PS D:\Karthick\Masai\Projects\Swiggy_instamart_support> python .\part_a_run_queries.py

10.	pandas cross-check and exploratory analysis
(.venv) PS D:\Karthick\Masai\Projects\Swiggy_instamart_support> python .\part_a_pandas_eda.py

Part B — Customer Segments and Lapse Prediction

(.venv) PS D:\Karthick\Masai\Projects\Swiggy_instamart_support> python .\part_b.py

