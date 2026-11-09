import pandas as pd
from mock_llm import est_tokens

# Input guardrails

MAX_INPUT_TOKENS = 400

tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

tickets["status"] = "ready"
tickets["estimated_tokens"] = 0

for index, row in tickets.iterrows():
    text = row["text"]

    if not text.strip():
        tickets.at[index, "status"] = "skipped_empty"
        continue

    token_count = est_tokens(text)
    tickets.at[index, "estimated_tokens"] = token_count

    if token_count > MAX_INPUT_TOKENS:
        tickets.at[index, "status"] = "flagged_long"

print(tickets["status"].value_counts())
print("Long-ticket token estimates:")
print(tickets.loc[tickets["status"] == "flagged_long", "estimated_tokens"].tolist())

#Redact personal information

import re

patterns = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

substitution_counts = {name: 0 for name, _, _ in [
    (name, pattern, placeholder) for name, pattern, placeholder in patterns
]}

def redact(text):
    for name, pattern, placeholder in patterns:
        text, count = pattern.subn(placeholder, text)
        substitution_counts[name] += count
    return text

tickets["redacted_text"] = tickets["text"].map(redact)

total_substitutions = sum(substitution_counts.values())
print("Substitutions by type:", substitution_counts)
print("Total substitutions:", total_substitutions)

# Verify re-applying all three patterns finds no matches.
remaining = {
    name: sum(
        len(pattern.findall(text))
        for text in tickets["redacted_text"]
    )
    for name, pattern, _ in patterns
}
print("Matches remaining after redaction:", remaining)
assert all(count == 0 for count in remaining.values())

#Two prompts

#CLASSIFY the ticket into exactly one allowed category: billing, delivery, technical, or fraud. Return only valid JSON in this form: {"category": "...", "reason": "..."}. Use one of the four allowed category names exactly. Keep the reason brief and do not include any text outside the JSON object.

#CLASSIFY the ticket into exactly one allowed category: billing, delivery, technical, or fraud. Return only valid JSON in this form: {"category": "...", "reason": "..."}. Use one of the four allowed category names exactly. Keep the reason brief and do not include any text outside the JSON object.

#EXAMPLES:
#Ticket: I was charged twice for one order.
#Reply: {"category": "billing", "reason": "The ticket reports a duplicate charge."}

#Ticket: Someone used my account to place an order I did not make.
#Reply: {"category": "fraud", "reason": "The ticket reports an unauthorized order."}


import inspect
import mock_llm

for name, item in vars(mock_llm).items():
    if name.startswith("_") or inspect.isclass(item):
        continue
    if callable(item):
        try:
            print(name, inspect.signature(item))
        except (TypeError, ValueError):
            print(name, "(signature unavailable)")

#Resilient calls

import random
import time

import mock_llm
from mock_llm import TransientLLMError

BASE_DELAY = 0.05
MAX_RETRIES = 3  # retries after the initial attempt

zero_shot_system_prompt = """
CLASSIFY the ticket into exactly one allowed category: billing, delivery, technical, or fraud.
Return only valid JSON in this form: {"category": "...", "reason": "..."}.
Use one of the four allowed category names exactly. Keep the reason brief.
Do not include any text outside the JSON object.
"""

def call_with_retry(ticket_id, system_prompt, redacted_text):
    retries_used = 0

    for retry_number in range(MAX_RETRIES + 1):
        try:
            response = mock_llm.chat(messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": redacted_text},
            ])
            return response, retries_used

        except TransientLLMError:
            if retry_number == MAX_RETRIES:
                raise  # all 3 retries were used

            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )
            print(
                f"Retry ticket_id={ticket_id} "
                f"retry_number={retry_number} wait={wait:.4f}s"
            )
            time.sleep(wait)
            retries_used += 1

results = []
total_retries = 0

# Reset immediately before this full run.
mock_llm.reset()
for _, row in tickets.iterrows():
    if row["status"] != "ready":
        continue  # empty and long tickets never reach the model

    response, retries_used = call_with_retry(
        ticket_id=row["ticket_id"],
        system_prompt=zero_shot_system_prompt,
        redacted_text=row["redacted_text"],
    )
    total_retries += retries_used
    results.append({
        "ticket_id": row["ticket_id"],
        "response": response,
        "retries_used": retries_used,
    })

print("Tickets processed:", len(results))
print("Transient errors recovered by retry:", total_retries)