Part A turns the messy exports into a trusted analytical database. Part B uses that database to segment customers and to predict who is likely to lapse. Part C reads support tickets, protects customer privacy, and routes each ticket using the segments from Part B. Part D builds a policy question-answering assistant and a refund agent that queries the database from Part A and reuses the policy retrieval built in the same Part. Every Part consumes what the previous Parts produced 



Reproduce:

1.	D:\Karthick\Masai\Projects\Swiggy_instamart_support> C:\Users\skjee\pythonProject2\Scripts\python.exe -m venv .venv

2.	PS D:\Karthick\Masai\Projects\Swiggy_instamart_support> .\.venv\Scripts\Activate.ps1

3.	python -m pip freeze > requirements.txt

4.	(.venv) PS D:\Karthick\Masai\Projects\Swiggy_instamart_support> python .\generate_data.py
generated: customers=1200 products=40 orders_raw=13084 tickets=164


