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

Part C — Support Ticket Triage with Guardrails 

(.venv) PS D:\Karthick\Masai\Projects\Swiggy_instamart_support> python .\part_c.py

Part D — Policy RAG Assistant and Refund Agent

(.venv) PS D:\Karthick\Masai\Projects\Swiggy_instamart_support> python .\part_d.py

## Part D — Policy retrieval and refund agent

Policies are split into 40-word chunks with 10-word overlap and ranked with scikit-learn TF-IDF cosine similarity. The assistant sends only the top three retrieved chunks as context to the local `mock_llm`; the refund agent uses the same retrieval function and database-backed order tools. Refund and escalation tools append simulated actions to `outputs/refund_ledger.jsonl`.
Term frequency (TF) measures how often a query term appears in a chunk, while inverse document frequency (IDF) downweights terms common across chunks and upweights rarer terms. TF-IDF can miss semantically equivalent wording when the query and relevant policy passage use different words.

### Retrieval and grounding evaluation

| Overlap | Chunks | Answerable doc hits | Expected fact found | Unanswerable fallback |
|---:|---:|---:|---:|---:|
| 10 words | 15 | 12/12 | 12/12 | 2/2 |
| 0 words | 13 | 12/12 | 10/12 | 2/2 |

A question that breaks with zero overlap: **What delivery time makes a completed order eligible for a full refund?**. The answer fact falls across a 40-word boundary, so neither isolated chunk contains the complete wording.

### Refund scenarios

| Scenario | Status | Refund INR | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 1087 |
| S2 | REFUND | 135.00 | 5 | 1087 |
| S3 | ESCALATE | 1999.00 | 5 | 1074 |
| S4 | DENIED | 0.00 | 4 | 757 |
| S5 | DENIED | 0.00 | 3 | 451 |
| S6 | NOT_FOUND | 0.00 | 2 | 148 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3228 |

Safety demos: S1 with `token_limit=200` returned **FALLBACK_TOKEN_LIMIT** (468 tokens); `time_limit_seconds=0` returned **FALLBACK_TIMEOUT** (0 model calls).

Crash/resume demo: tools across the crash and resume were **get_order, search_policy, calculate_refund, issue_refund** (no repeated tool execution).

### ReAct versus plan-and-execute

This ReAct run used 5 calls/1087 tokens for S1, 3 calls/451 tokens for S5, and 2 calls/148 tokens for S6. A plan-and-execute design would usually ask the model for a sequence once and call tools deterministically, reducing model calls and often tokens on routine cases. ReAct can use an unexpected observation such as NOT_FOUND or a cancelled status to change its next action immediately; a fixed plan may need a replan call or careful branching code. For refunds I prefer ReAct with strict tool-side eligibility checks and iteration limits, because it adapts to database observations while keeping financial decisions enforced by deterministic tools.
