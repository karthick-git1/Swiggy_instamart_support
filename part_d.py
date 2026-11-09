"""Part D: retrieval-grounded policy assistant and tool-using refund agent.

Run from the project root:
    python part_d.py

The script uses the supplied local mock_llm module and the Part A SQLite DB.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mock_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "data").exists() else HERE.parent
POLICY_DIR = ROOT / "data" / "policies"
DB_PATH = ROOT / "data" / "instamart.db"
MAX_AGENT_STEPS = 6


@dataclass(frozen=True)
class PolicyChunk:
    source: str
    text: str

    @property
    def prompt_text(self) -> str:
        # mock_llm's policy retriever strips source tags in this form before
        # splitting context into candidate sentences.
        return f"[{self.source}] {self.text}"


class PolicyRetriever:
    """A small local TF-IDF sentence retriever over the five policy files."""

    def __init__(self, policy_dir: Path = POLICY_DIR) -> None:
        self.chunks: list[PolicyChunk] = []
        for path in sorted(policy_dir.glob("*.txt")):
            raw_text = path.read_text(encoding="utf-8")
            sentences = re.split(r"(?<=[.!?])\s+", raw_text.strip())
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    self.chunks.append(PolicyChunk(path.name, sentence))

        if not self.chunks:
            raise FileNotFoundError(f"No policy text files found in {policy_dir}")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform(
            [chunk.text for chunk in self.chunks]
        )

    def search(self, query: str, top_k: int = 3) -> list[PolicyChunk]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = scores.argsort()[::-1]
        return [self.chunks[i] for i in ranked[:top_k] if scores[i] > 0]


class PartD:
    def __init__(self, policy_dir: Path = POLICY_DIR, db_path: Path = DB_PATH):
        self.retriever = PolicyRetriever(policy_dir)
        self.db_path = db_path

    def answer_policy_question(self, question: str, top_k: int = 4) -> dict[str, Any]:
        chunks = self.retriever.search(question, top_k=top_k)
        if not chunks:
            return {
                "answer": "I could not find that information in the policy documents.",
                "sources": [],
            }

        context = "\n".join(chunk.prompt_text for chunk in chunks)
        messages = [
            {
                "role": "system",
                "content": (
                    "POLICY ASSISTANT. Answer using only the supplied policy context. "
                    "If the context does not support an answer, say so. Be concise."
                ),
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\nQUESTION: {question}",
            },
        ]
        response = mock_llm.chat(messages=messages)
        answer = response.choices[0].message.content or ""
        return {
            "answer": answer,
            "sources": list(dict.fromkeys(chunk.source for chunk in chunks)),
        }

    def _get_order(self, order_id: str) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT order_id, status, order_amount, delivery_minutes
                FROM orders
                WHERE order_id = ?
                """,
                (order_id,),
            ).fetchone()
        if row is None:
            return {"error": "order_not_found", "order_id": order_id}
        return dict(row)

    def run_refund_agent(self, request: str) -> dict[str, Any]:
        """Handle one request via mock tool calls; refund actions are simulated."""
        current_order: dict[str, Any] | None = None
        calculated_refund: float | None = None
        tool_events: list[dict[str, Any]] = []

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_order",
                    "description": "Look up an order in the Part A database.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_policy",
                    "description": "Retrieve relevant policy passages.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_refund",
                    "description": "Calculate an eligible late-delivery refund.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_amount": {"type": "number"},
                            "delivery_minutes": {"type": ["number", "null"]},
                        },
                        "required": ["order_amount", "delivery_minutes"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "issue_refund",
                    "description": "Simulate issuing an approved refund of INR 1000 or less.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "refund_inr": {"type": "number"},
                        },
                        "required": ["order_id", "refund_inr"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "escalate_to_human",
                    "description": "Escalate a refund above INR 1000; do not issue it.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "refund_inr": {"type": "number"},
                        },
                        "required": ["order_id", "refund_inr"],
                    },
                },
            },
        ]

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "AGENT. Process refund requests by calling tools. Use get_order, "
                    "retrieve policy with search_policy, calculate eligibility with "
                    "calculate_refund, then issue refunds only when the tool permits; "
                    "escalate amounts above INR 1000. Never invent order facts. "
                    "Finish within six tool steps or hand off to a human."
                ),
            },
            {"role": "user", "content": request},
        ]

        def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
            nonlocal current_order, calculated_refund

            if name == "get_order":
                current_order = self._get_order(str(args.get("order_id", "")))
                calculated_refund = None
                return current_order

            if name == "search_policy":
                chunks = self.retriever.search(str(args.get("query", "")), top_k=4)
                return {
                    "passages": [chunk.prompt_text for chunk in chunks],
                    "sources": list(dict.fromkeys(chunk.source for chunk in chunks)),
                }

            if name == "calculate_refund":
                if not current_order or current_order.get("error"):
                    return {"error": "order_must_be_looked_up_first", "refund_inr": 0}
                if current_order.get("status") != "completed":
                    calculated_refund = 0.0
                    return {"refund_inr": 0.0, "decision": "not_refundable_not_completed"}

                # Database values are authoritative; model-supplied arguments
                # cannot alter refund eligibility or amount.
                delivery = current_order.get("delivery_minutes")
                amount = current_order.get("order_amount")
                if delivery is None:
                    return {"error": "completed_order_missing_delivery_time"}

                calculated_refund = (
                    round(float(amount), 2) if float(delivery) > 45 else 0.0
                )
                decision = "late_delivery_full_refund" if calculated_refund else "within_45_minutes_no_refund"
                return {"refund_inr": calculated_refund, "decision": decision}

            if name in {"issue_refund", "escalate_to_human"}:
                order_id = str(args.get("order_id", ""))
                requested_amount = round(float(args.get("refund_inr", 0)), 2)
                if not current_order or current_order.get("error"):
                    return {"error": "order_not_verified"}
                if current_order.get("status") != "completed":
                    return {"error": "only_completed_orders_are_refundable"}
                if order_id != current_order.get("order_id"):
                    return {"error": "order_id_mismatch"}
                if calculated_refund is None or requested_amount != calculated_refund:
                    return {"error": "refund_amount_not_verified"}
                if requested_amount <= 0:
                    return {"error": "no_refund_due"}

                if name == "issue_refund":
                    if requested_amount > 1000:
                        return {"error": "refund_above_1000_requires_human_approval"}
                    return {
                        "decision": "refund_issued_simulated",
                        "order_id": order_id,
                        "refund_inr": requested_amount,
                    }

                if requested_amount <= 1000:
                    return {"error": "only_refunds_above_1000_are_escalated"}
                return {
                    "decision": "escalated_to_human_not_issued",
                    "order_id": order_id,
                    "refund_inr": requested_amount,
                }

            return {"error": f"unknown_tool:{name}"}

        for step in range(MAX_AGENT_STEPS):
            response = mock_llm.chat(messages=messages, tools=tools)
            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls or []
            if not tool_calls:
                return {
                    "status": "completed",
                    "reply": assistant_message.content or "",
                    "order": current_order,
                    "refund_inr": calculated_refund,
                    "steps": step + 1,
                    "tool_events": tool_events,
                }

            # Keep an assistant tool-call message in the conversation, then answer
            # each requested tool with a structured result.
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in tool_calls
                ],
            })

            for call in tool_calls:
                args = call.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                result = execute_tool(call.name, args)
                tool_events.append({"tool": call.name, "arguments": args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(result),
                })

        return {
            "status": "escalated_to_human",
            "reply": "Automated handling exceeded six steps; please review this request.",
            "order": current_order,
            "refund_inr": calculated_refund,
            "steps": MAX_AGENT_STEPS,
            "tool_events": tool_events,
        }


def main() -> None:
    assistant = PartD()
    print("Part D policy assistant and refund agent (local mock; no refund is actually issued).")
    print("Choose: policy <question> | refund <request containing an order ID> | quit")
    # One reset for this independent interactive run.
    mock_llm.reset()
    while True:
        user_input = input("\\n> ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if user_input.lower().startswith("policy "):
            result = assistant.answer_policy_question(user_input[7:])
            print(json.dumps(result, indent=2))
        elif user_input.lower().startswith("refund "):
            result = assistant.run_refund_agent(user_input[7:])
            print(json.dumps(result, indent=2))
        else:
            print("Use 'policy <question>' or 'refund <request>'.")


if __name__ == "__main__":
    main()
