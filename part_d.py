"""Part D: policy retrieval, grounded answers, and a resumable refund agent.

Copy this file to the project root (beside mock_llm.py), then run:
    python part_d.py

The mock is local/offline. Refund actions are recorded to a JSONL ledger and
are simulated; no payment system is connected.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(os.environ.get("INSTAMART_PROJECT_ROOT", Path(__file__).resolve().parent))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import mock_llm  # noqa: E402  (project root must be added to sys.path first)

POLICY_DIR = PROJECT_ROOT / "data" / "policies"
DB_PATH = PROJECT_ROOT / "data" / "instamart.db"
OUTPUT_DIR = Path(os.environ.get("INSTAMART_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
STATE_DIR = Path(os.environ.get("INSTAMART_STATE_DIR", PROJECT_ROOT / "state"))
README_PATH = Path(os.environ.get("INSTAMART_README", PROJECT_ROOT / "README.md"))
LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"
AGENT_RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"
MAX_AGENT_STEPS = 6


def chunk_documents(policy_dir: Path = POLICY_DIR, overlap: int = 10) -> list[dict[str, str]]:
    """Split each policy into 40-word chunks with the requested overlap."""
    if overlap < 0 or overlap >= 40:
        raise ValueError("overlap must be from 0 through 39")
    chunks: list[dict[str, str]] = []
    stride = 40 - overlap
    for path in sorted(policy_dir.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(start + 40, len(words))
            chunks.append({"source": path.name, "text": " ".join(words[start:end])})
            if end >= len(words):
                break
            start += stride
    return chunks


class Retriever:
    def __init__(self, chunks: list[dict[str, str]]):
        if not chunks:
            raise ValueError("Cannot fit TF-IDF on an empty chunk list")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([chunk["text"] for chunk in chunks])

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [
            {
                "source": self.chunks[index]["source"],
                "score": float(scores[index]),
                "text": self.chunks[index]["text"],
            }
            for index in order
        ]


CHUNKS = chunk_documents(overlap=10)
RETRIEVER = Retriever(CHUNKS)


def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    return RETRIEVER.retrieve(query, k)


POLICY_SYSTEM_PROMPT = (
    "POLICY ASSISTANT. Answer only from the supplied context. If the context does "
    "not contain the answer, say: I could not find the information in the policy documents."
)


def answer(question: str) -> dict[str, Any]:
    results = retrieve(question, k=3)
    context_lines = [f"[{item['source']}] {item['text']}" for item in results]
    user_content = "CONTEXT:\n" + "\n".join(context_lines) + "\nQUESTION:\n" + question
    response = mock_llm.chat(messages=[
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ])
    return {
        "answer": response.choices[0].message.content or "",
        "sources": list(dict.fromkeys(item["source"] for item in results)),
    }


def evaluate_retrieval(eval_path: Path, overlap: int) -> tuple[dict[str, int], list[str]]:
    import pandas as pd

    chunks = chunk_documents(POLICY_DIR, overlap=overlap)
    index = Retriever(chunks)
    frame = pd.read_csv(eval_path, keep_default_na=False)
    answerable = frame[frame["expected_doc"] != "NONE"]
    unanswerable = frame[frame["expected_doc"] == "NONE"]
    doc_hits = fact_hits = fallback_hits = 0
    broken: list[str] = []
    for row in frame.itertuples(index=False):
        hits = index.retrieve(row.question, 3)
        sources = [item["source"] for item in hits]
        result = _answer_with_retriever(row.question, index)
        if row.expected_doc != "NONE":
            has_doc = row.expected_doc in sources
            has_fact = row.expected_fact in result["answer"]
            doc_hits += int(has_doc)
            fact_hits += int(has_fact)
            if not has_doc or not has_fact:
                broken.append(row.question)
        else:
            fallback_hits += int(result["answer"].startswith("I could not find"))
    return {
        "answerable": len(answerable),
        "expected_doc_hits": doc_hits,
        "expected_fact_hits": fact_hits,
        "unanswerable": len(unanswerable),
        "fallback_hits": fallback_hits,
        "chunks": len(chunks),
    }, broken


def _answer_with_retriever(question: str, index: Retriever) -> dict[str, Any]:
    results = index.retrieve(question, 3)
    user_content = (
        "CONTEXT:\n"
        + "\n".join(f"[{item['source']}] {item['text']}" for item in results)
        + "\nQUESTION:\n"
        + question
    )
    response = mock_llm.chat(messages=[
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ])
    return {
        "answer": response.choices[0].message.content or "",
        "sources": list(dict.fromkeys(item["source"] for item in results)),
    }


_ACTIVE_ORDER: dict[str, Any] | None = None


def get_order(order_id: str) -> dict[str, Any]:
    global _ACTIVE_ORDER
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT order_id, status, order_amount, delivery_minutes "
            "FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    if row is None:
        _ACTIVE_ORDER = {"error": "order not found"}
        return _ACTIVE_ORDER
    _ACTIVE_ORDER = dict(row)
    return _ACTIVE_ORDER


def search_policy(query: str) -> dict[str, Any]:
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount: float, delivery_minutes: float) -> dict[str, float]:
    # The prior get_order observation is authoritative. This preserves the
    # required two-argument tool API while refusing calculations without a
    # verified, completed order.
    if not _ACTIVE_ORDER or _ACTIVE_ORDER.get("status") != "completed":
        return {"refund_inr": 0.0}
    verified_amount = _ACTIVE_ORDER.get("order_amount")
    verified_minutes = _ACTIVE_ORDER.get("delivery_minutes")
    if verified_amount is None or verified_minutes is None:
        return {"refund_inr": 0.0}
    amount = round(float(verified_amount), 2) if float(verified_minutes) > 45 else 0.0
    return {"refund_inr": amount}


def _append_ledger(tool_name: str, order_id: str, refund_inr: float) -> dict[str, float]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }
    with LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return {"refund_inr": record["refund_inr"]}


def issue_refund(order_id: str, refund_inr: float) -> dict[str, float]:
    order = get_order(order_id)
    if order.get("status") != "completed":
        return {"error": "only completed orders can be refunded"}
    expected = calculate_refund(order["order_amount"], order["delivery_minutes"])["refund_inr"]
    if expected <= 0 or round(float(refund_inr), 2) != expected:
        return {"error": "refund amount is not eligible"}
    if expected > 1000:
        return {"error": "refund above INR 1000 requires human approval"}
    return _append_ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id: str, refund_inr: float) -> dict[str, float]:
    order = get_order(order_id)
    if order.get("status") != "completed":
        return {"error": "only completed orders can be escalated for refund"}
    expected = calculate_refund(order["order_amount"], order["delivery_minutes"])["refund_inr"]
    if expected <= 1000 or round(float(refund_inr), 2) != expected:
        return {"error": "only eligible refunds above INR 1000 are escalated"}
    return _append_ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order and its refund-relevant fields in the SQLite database.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
    {
        "name": "search_policy",
        "description": "Retrieve the three most relevant policy chunks for a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the full order amount only when delivery exceeded 45 minutes.",
        "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, "required": ["order_amount", "delivery_minutes"]},
    },
    {
        "name": "issue_refund",
        "description": "Record an automatically approved refund of INR 1000 or less in the ledger.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]},
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund above INR 1000 for human approval without issuing it.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]},
    },
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


class SimulatedCrash(RuntimeError):
    pass


def _status_from_answer(text: str) -> str:
    return text.split(":", 1)[0].strip() if ":" in text else "UNKNOWN"


def _checkpoint_path(run_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    return STATE_DIR / f"{safe_id}.json"


def _save_checkpoint(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(state["run_id"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fallback(state: dict[str, Any], code: str, detail: str) -> dict[str, Any]:
    state["elapsed_seconds"] += max(0.0, time.monotonic() - state.pop("segment_started"))
    result = {
        "status": code,
        "final_answer": f"{code}: {detail} Please hand this request to a human support agent.",
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
    }
    state["last_result"] = result
    _save_checkpoint(state)
    return result


def _continue_agent(state: dict[str, Any], crash_after_tool: int | None = None) -> dict[str, Any]:
    state["segment_started"] = time.monotonic()
    while state["llm_calls"] < state["max_iterations"]:
        elapsed = state["elapsed_seconds"] + (time.monotonic() - state["segment_started"])
        if elapsed >= state["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT", "The automated time limit was reached.")

        response = mock_llm.chat(messages=state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += int(response.usage.total_tokens)
        assistant = response.choices[0].message
        calls = assistant.tool_calls or []

        if state["tokens_used"] > state["token_limit"]:
            return _fallback(
                state,
                "FALLBACK_TOKEN_LIMIT",
                f"The token budget was exceeded ({state['tokens_used']} tokens used).",
            )

        if not calls:
            final_answer = assistant.content or ""
            state["elapsed_seconds"] += max(0.0, time.monotonic() - state.pop("segment_started"))
            result = {
                "status": _status_from_answer(final_answer),
                "final_answer": final_answer,
                "steps": state["steps"],
                "llm_calls": state["llm_calls"],
                "tokens_used": state["tokens_used"],
            }
            state["last_result"] = result
            _save_checkpoint(state)
            return result

        assistant_calls = []
        for call in calls:
            arguments = call.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            assistant_calls.append({"id": call.id, "name": call.name, "arguments": arguments})
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": assistant_calls})

        for call_data in assistant_calls:
            name = call_data["name"]
            arguments = call_data["arguments"]
            tool = TOOL_MAP.get(name)
            observation = tool(**arguments) if tool else {"error": "unknown tool"}
            state["messages"].append({
                "role": "tool",
                "tool_call_id": call_data["id"],
                "name": name,
                "content": json.dumps(observation, ensure_ascii=False),
            })
            step = {"tool": name, "arguments": arguments, "observation": observation}
            state["steps"].append(step)
            state["tools_executed"].append(name)
            state["elapsed_seconds"] += max(0.0, time.monotonic() - state.pop("segment_started"))
            state["step_number"] += 1
            _save_checkpoint(state)

            # Start the elapsed-time segment again after checkpoint I/O.
            state["segment_started"] = time.monotonic()
            if crash_after_tool is not None and len(state["tools_executed"]) == crash_after_tool:
                raise SimulatedCrash(f"Simulated crash after tool {crash_after_tool}; checkpoint saved.")

    return _fallback(
        state,
        "FALLBACK_MAX_ITERATIONS",
        f"The agent reached its {state['max_iterations']}-call limit.",
    )


def run_agent(
    user_message: str,
    scenario_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10,
    *,
    run_id: str | None = None,
    crash_after_tool: int | None = None,
) -> dict[str, Any]:
    run_id = run_id or scenario_id
    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "AGENT. Resolve refund requests with the available tools. Check order "
                    "status, retrieve relevant policy, calculate refunds, issue amounts up "
                    "to INR 1000, and escalate larger amounts. Never refund cancelled or "
                    "missing orders. Use no more than six reasoning/tool iterations."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "step_number": 0,
        "tools_executed": [],
        "steps": [],
        "llm_calls": 0,
        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,
        "elapsed_seconds": 0.0,
    }
    _save_checkpoint(state)
    return _continue_agent(state, crash_after_tool=crash_after_tool)


def resume_agent(run_id: str) -> dict[str, Any]:
    global _ACTIVE_ORDER
    state = json.loads(_checkpoint_path(run_id).read_text(encoding="utf-8"))
    if state.get("last_result") is not None:
        return state["last_result"]
    # Reconstruct verified order context from saved tool observations so a
    # process restart between search_policy and calculate_refund is safe.
    _ACTIVE_ORDER = None
    for step in reversed(state["steps"]):
        if step["tool"] == "get_order":
            _ACTIVE_ORDER = step["observation"]
            break
    return _continue_agent(state)


def _refund_from_steps(result: dict[str, Any]) -> float:
    for step in reversed(result["steps"]):
        observation = step["observation"]
        if "refund_inr" in observation:
            return float(observation["refund_inr"])
    return 0.0


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_readme_section(
    overlap10: dict[str, int], overlap0: dict[str, int], broken: list[str],
    scenario_rows: list[dict[str, Any]], token_demo: dict[str, Any],
    timeout_demo: dict[str, Any], resume_tools: list[str],
) -> str:
    scenario_table = [
        "| Scenario | Status | Refund INR | Model calls | Tokens used |",
        "|---|---|---:|---:|---:|",
    ]
    for row in scenario_rows:
        scenario_table.append(
            f"| {row['scenario_id']} | {row['status']} | {row['refund_inr']:.2f} | "
            f"{row['llm_calls']} | {row['tokens_used']} |"
        )
    broken_question = broken[0] if broken else "(none in this run)"
    return "\n".join([
        "## Part D — Policy retrieval and refund agent",
        "",
        "Policies are split into 40-word chunks with 10-word overlap and ranked with "
        "scikit-learn TF-IDF cosine similarity. The assistant sends only the top three "
        "retrieved chunks as context to the local `mock_llm`; the refund agent uses the "
        "same retrieval function and database-backed order tools. Refund and escalation "
        "tools append simulated actions to `outputs/refund_ledger.jsonl`.",
        "Term frequency (TF) measures how often a query term appears in a chunk, while inverse document frequency (IDF) downweights terms common across chunks and upweights rarer terms. TF-IDF can miss semantically equivalent wording when the query and relevant policy passage use different words.",
        "",
        "### Retrieval and grounding evaluation",
        "",
        "| Overlap | Chunks | Answerable doc hits | Expected fact found | Unanswerable fallback |",
        "|---:|---:|---:|---:|---:|",
        f"| 10 words | {overlap10['chunks']} | {overlap10['expected_doc_hits']}/{overlap10['answerable']} | "
        f"{overlap10['expected_fact_hits']}/{overlap10['answerable']} | "
        f"{overlap10['fallback_hits']}/{overlap10['unanswerable']} |",
        f"| 0 words | {overlap0['chunks']} | {overlap0['expected_doc_hits']}/{overlap0['answerable']} | "
        f"{overlap0['expected_fact_hits']}/{overlap0['answerable']} | "
        f"{overlap0['fallback_hits']}/{overlap0['unanswerable']} |",
        "",
        f"A question that breaks with zero overlap: **{broken_question}**. The answer fact "
        "falls across a 40-word boundary, so neither isolated chunk contains the complete wording.",
        "",
        "### Refund scenarios",
        "",
        *scenario_table,
        "",
        f"Safety demos: S1 with `token_limit=200` returned **{token_demo['status']}** "
        f"({token_demo['tokens_used']} tokens); `time_limit_seconds=0` returned "
        f"**{timeout_demo['status']}** ({timeout_demo['llm_calls']} model calls).",
        "",
        f"Crash/resume demo: tools across the crash and resume were **{', '.join(resume_tools)}** "
        "(no repeated tool execution).",
        "",
        "### ReAct versus plan-and-execute",
        "",
        f"This ReAct run used {scenario_rows[0]['llm_calls']} calls/{scenario_rows[0]['tokens_used']} tokens for S1, "
        f"{scenario_rows[4]['llm_calls']} calls/{scenario_rows[4]['tokens_used']} tokens for S5, and "
        f"{scenario_rows[5]['llm_calls']} calls/{scenario_rows[5]['tokens_used']} tokens for S6. "
        "A plan-and-execute design would usually ask the model for a sequence once and call "
        "tools deterministically, reducing model calls and often tokens on routine cases. "
        "ReAct can use an unexpected observation such as NOT_FOUND or a cancelled status to "
        "change its next action immediately; a fixed plan may need a replan call or careful "
        "branching code. For refunds I prefer ReAct with strict tool-side eligibility checks "
        "and iteration limits, because it adapts to database observations while keeping "
        "financial decisions enforced by deterministic tools.",
        "",
    ])


def run_all() -> None:
    import pandas as pd

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    # Start a clean demonstration ledger; individual tools append one JSON line.
    LEDGER_PATH.write_text("", encoding="utf-8")

    eval_path = PROJECT_ROOT / "data" / "raw" / "policy_eval.csv"
    overlap10, _ = evaluate_retrieval(eval_path, overlap=10)
    overlap0, broken = evaluate_retrieval(eval_path, overlap=0)
    print("Chunk count (10-word overlap):", len(CHUNKS))
    print("Chunk sources:", {name: sum(c["source"] == name for c in CHUNKS) for name in sorted({c["source"] for c in CHUNKS})})
    print("Policy evaluation, overlap=10:", overlap10)
    print("Policy evaluation, overlap=0:", overlap0)
    print("Questions failing with no overlap:", broken)
    print("Top refund retrieval sources:", [r["source"] for r in retrieve("refund policy late delivery", 3)])

    scenario_path = PROJECT_ROOT / "data" / "raw" / "agent_scenarios.csv"
    scenarios = pd.read_csv(scenario_path, keep_default_na=False)
    scenario_rows: list[dict[str, Any]] = []
    for row in scenarios.itertuples(index=False):
        mock_llm.reset()
        result = run_agent(row.message, row.scenario_id, run_id=f"scenario_{row.scenario_id}")
        refund = _refund_from_steps(result)
        if abs(refund - float(row.expected_refund_inr)) > 0.01:
            raise AssertionError(
                f"{row.scenario_id}: refund {refund:.2f} != expected {row.expected_refund_inr:.2f}"
            )
        record = {
            "scenario_id": row.scenario_id,
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"],
            "refund_inr": refund,
        }
        scenario_rows.append(record)
        print(f"{row.scenario_id}: {record['status']} refund={refund:.2f} calls={record['llm_calls']} tokens={record['tokens_used']}")
    _write_jsonl(AGENT_RUNS_PATH, scenario_rows)

    s1 = scenarios.loc[scenarios["scenario_id"] == "S1"].iloc[0]
    mock_llm.reset()
    token_demo = run_agent(s1["message"], "S1_TOKEN_LIMIT", token_limit=200, run_id="demo_token_limit")
    mock_llm.reset()
    timeout_demo = run_agent(s1["message"], "S1_TIMEOUT", time_limit_seconds=0, run_id="demo_timeout")
    print("S1 token-limit demo:", token_demo["status"], token_demo["tokens_used"])
    print("Timeout demo:", timeout_demo["status"], timeout_demo["llm_calls"])

    mock_llm.reset()
    resume_id = "demo_S1_resume"
    try:
        run_agent(
            s1["message"], "S1_RESUME", run_id=resume_id,
            crash_after_tool=2,
        )
    except SimulatedCrash as exc:
        print(exc)
    resumed = resume_agent(resume_id)
    resume_tools = [step["tool"] for step in resumed["steps"]]
    if resume_tools != ["get_order", "search_policy", "calculate_refund", "issue_refund"]:
        raise AssertionError(f"Unexpected resume tool sequence: {resume_tools}")
    if resumed["status"] != "REFUND" or abs(_refund_from_steps(resumed) - 128.25) > 0.01:
        raise AssertionError("S1 resume demonstration did not finish with the expected refund")
    print("Resume result:", resumed["status"], resume_tools)

    section = _build_readme_section(
        overlap10, overlap0, broken, scenario_rows, token_demo, timeout_demo, resume_tools
    )
    old = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
    if "## Part D — Policy retrieval and refund agent" in old:
        old = old.split("## Part D — Policy retrieval and refund agent", 1)[0].rstrip()
    README_PATH.parent.mkdir(parents=True, exist_ok=True)
    README_PATH.write_text((old + "\n\n" + section).strip() + "\n", encoding="utf-8")
    print("Wrote:", AGENT_RUNS_PATH, LEDGER_PATH, README_PATH)


if __name__ == "__main__":
    run_all()
