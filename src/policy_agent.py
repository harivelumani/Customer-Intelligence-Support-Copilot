from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "data" / "policies"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
STATE = ROOT / "state"
DB = ROOT / "data" / "instamart.db"
FALLBACK = "I could not find that information in the policy documents."
POLICY_FILES = [
    "refund_policy.txt", "delivery_policy.txt", "payment_policy.txt",
    "privacy_policy.txt", "escalation_policy.txt",
]
TOOL_NAMES = {"get_order", "search_policy", "calculate_refund", "issue_refund", "escalate_to_human"}

TOOLS_SPEC = [
    {"name": "get_order", "description": "Fetch a clean order by its unique order ID.", "parameters": {"order_id": "string"}},
    {"name": "search_policy", "description": "Retrieve policy chunks relevant to a query.", "parameters": {"query": "string"}},
    {"name": "calculate_refund", "description": "Calculate a full refund only for delivery later than 45 minutes.", "parameters": {"order_amount": "number", "delivery_minutes": "integer"}},
    {"name": "issue_refund", "description": "Record an eligible automatic refund of INR 1000 or less.", "parameters": {"order_id": "string", "refund_inr": "number"}},
    {"name": "escalate_to_human", "description": "Record an eligible refund above INR 1000 for human approval.", "parameters": {"order_id": "string", "refund_inr": "number"}},
]


def chunk_documents(overlap: int = 10) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for filename in POLICY_FILES:
        words = (POLICIES / filename).read_text(encoding="utf-8").split()
        start = 0
        while start < len(words):
            end = min(start + 40, len(words))
            chunks.append({"source": filename, "text": " ".join(words[start:end])})
            if end == len(words):
                break
            start += 40 - overlap
    return chunks


class Retriever:
    def __init__(self, overlap: int = 10):
        self.chunks = chunk_documents(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(chunk["text"] for chunk in self.chunks)

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [
            {"source": self.chunks[index]["source"], "score": float(scores[index]), "text": self.chunks[index]["text"]}
            for index in order
        ]


def answer(question: str, retriever: Retriever) -> dict[str, Any]:
    matches = retriever.retrieve(question, 3)
    context = "\n".join(f"[{item['source']}] {item['text']}" for item in matches)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT: answer only from the supplied context. If it does not contain "
                f"the answer, respond exactly: {FALLBACK}"
            ),
        },
        {"role": "user", "content": user},
    ]
    response = mock_llm.chat(messages)
    result = response.choices[0].message.content or FALLBACK
    if not any(item["score"] > 0 for item in matches):
        result = FALLBACK
    return {"answer": result, "sources": [item["source"] for item in matches]}


def evaluate_retrieval(overlap: int) -> dict[str, Any]:
    retriever = Retriever(overlap)
    questions = pd.read_csv(RAW / "policy_eval.csv", keep_default_na=False)
    answerable_source_hits = 0
    answerable_fact_hits = 0
    unsupported_fallbacks = 0
    failed_questions = []
    rows = []
    for record in questions.to_dict(orient="records"):
        retrieved = retriever.retrieve(record["question"], 3)
        result = answer(record["question"], retriever)
        expected_doc = record["expected_doc"]
        expected_fact = record["expected_fact"]
        source_hit = expected_doc != "NONE" and expected_doc in [item["source"] for item in retrieved]
        fact_hit = expected_doc != "NONE" and expected_fact in result["answer"]
        fallback = result["answer"].startswith("I could not find")
        if expected_doc != "NONE":
            answerable_source_hits += int(source_hit)
            answerable_fact_hits += int(fact_hit)
            if not fact_hit:
                failed_questions.append(record["question"])
        else:
            unsupported_fallbacks += int(fallback)
        rows.append({
            "question": record["question"],
            "expected_doc": expected_doc,
            "retrieved_sources": [item["source"] for item in retrieved],
            "answer": result["answer"],
            "source_hit": source_hit,
            "fact_hit": fact_hit,
            "fallback": fallback,
        })
    return {
        "overlap": overlap,
        "chunk_counts": pd.Series([chunk["source"] for chunk in retriever.chunks]).value_counts().to_dict(),
        "total_chunks": len(retriever.chunks),
        "answerable_source_hits": answerable_source_hits,
        "answerable_fact_hits": answerable_fact_hits,
        "unsupported_fallbacks": unsupported_fallbacks,
        "failed_answerable_questions": failed_questions,
        "rows": rows,
        "retriever": retriever,
    }


class AgentCrash(RuntimeError):
    pass


def _load_order(order_id: str) -> dict[str, Any]:
    with sqlite3.connect(DB) as connection:
        row = connection.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": None if row[3] is None else int(row[3]),
    }


def _ledger_has_key(idempotency_key: str) -> bool:
    ledger = OUT / "refund_ledger.jsonl"
    if not ledger.exists():
        return False
    with ledger.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("idempotency_key") == idempotency_key:
                return True
    return False


class AgentRuntime:
    def __init__(self, retriever: Retriever, run_id: str):
        self.retriever = retriever
        self.run_id = run_id

    def get_order(self, order_id: str) -> dict[str, Any]:
        if not isinstance(order_id, str) or not re.fullmatch(r"ORD-\d{6}", order_id):
            raise ValueError("order_id must match ORD- followed by six digits.")
        return _load_order(order_id)

    def search_policy(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string.")
        return {"results": self.retriever.retrieve(query, 3)}

    def calculate_refund(self, order_amount: float, delivery_minutes: int) -> dict[str, Any]:
        if not isinstance(order_amount, (int, float)) or order_amount < 0:
            raise ValueError("order_amount must be a nonnegative number.")
        if not isinstance(delivery_minutes, int) or delivery_minutes < 0:
            raise ValueError("delivery_minutes must be a nonnegative integer.")
        amount = round(float(order_amount), 2) if delivery_minutes > 45 else 0.0
        return {"refund_inr": amount}

    def _validate_eligible_refund(self, order_id: str, refund_inr: float, must_escalate: bool) -> dict[str, Any]:
        if not isinstance(order_id, str) or not re.fullmatch(r"ORD-\d{6}", order_id):
            raise ValueError("order_id must match ORD- followed by six digits.")
        if not isinstance(refund_inr, (int, float)) or refund_inr <= 0:
            raise ValueError("refund_inr must be a positive number.")
        order = _load_order(order_id)
        if "error" in order or order["status"] != "completed" or order["delivery_minutes"] is None:
            raise ValueError("Refund action is not allowed for this order.")
        eligible = self.calculate_refund(order["order_amount"], order["delivery_minutes"])["refund_inr"]
        if abs(eligible - float(refund_inr)) > 0.01:
            raise ValueError("Refund amount does not match the policy-calculated amount.")
        if must_escalate != (eligible > 1000):
            raise ValueError("Refund action does not match the INR 1000 approval threshold.")
        return order

    def issue_refund(self, order_id: str, refund_inr: float, call_id: str) -> dict[str, Any]:
        self._validate_eligible_refund(order_id, refund_inr, must_escalate=False)
        key = f"{self.run_id}:{call_id}"
        if not _ledger_has_key(key):
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": "issue_refund",
                "run_id": self.run_id,
                "idempotency_key": key,
                "order_id": order_id,
                "refund_inr": round(float(refund_inr), 2),
            }
            with (OUT / "refund_ledger.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        return {"refund_inr": round(float(refund_inr), 2)}

    def escalate_to_human(self, order_id: str, refund_inr: float, call_id: str) -> dict[str, Any]:
        self._validate_eligible_refund(order_id, refund_inr, must_escalate=True)
        key = f"{self.run_id}:{call_id}"
        if not _ledger_has_key(key):
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": "escalate_to_human",
                "run_id": self.run_id,
                "idempotency_key": key,
                "order_id": order_id,
                "refund_inr": round(float(refund_inr), 2),
            }
            with (OUT / "refund_ledger.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        return {"refund_inr": round(float(refund_inr), 2)}

TOOL_MAP = {
    "get_order": AgentRuntime.get_order,
    "search_policy": AgentRuntime.search_policy,
    "calculate_refund": AgentRuntime.calculate_refund,
    "issue_refund": AgentRuntime.issue_refund,
    "escalate_to_human": AgentRuntime.escalate_to_human,
}


def _status_from_answer(final_answer: str) -> str:
    return final_answer.split(":", 1)[0].strip()


def _checkpoint_path(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id may contain only letters, digits, underscores, and hyphens.")
    STATE.mkdir(parents=True, exist_ok=True)
    return STATE / f"{run_id}.json"


def _save_checkpoint(run_id: str, state: dict[str, Any]) -> None:
    path = _checkpoint_path(run_id)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _continue_agent(
    run_id: str,
    state: dict[str, Any],
    retriever: Retriever,
    max_iterations: int,
    token_limit: int,
    time_limit_seconds: float,
    crash_after_tool_executions: int | None = None,
) -> dict[str, Any]:
    start_time = time.monotonic()
    runtime = AgentRuntime(retriever, run_id)
    while state["llm_calls"] < max_iterations:
        if time.monotonic() - start_time >= time_limit_seconds:
            summary = f"FALLBACK_TIMEOUT: hand off to a human; completed {len(state['steps'])} tool steps."
            return _agent_result(summary, state)
        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += int(response.usage.total_tokens)
        if state["tokens_used"] > token_limit:
            summary = f"FALLBACK_TOKEN_LIMIT: hand off to a human; completed {len(state['steps'])} tool steps."
            return _agent_result(summary, state)
        message = response.choices[0].message
        tool_calls = [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in (message.tool_calls or [])
        ]
        if not tool_calls:
            final_answer = message.content or "FALLBACK: hand off to a human because no answer was returned."
            return _agent_result(final_answer, state)
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        for call in tool_calls:
            name = call["name"]
            args = call["arguments"]
            if name not in TOOL_NAMES or not isinstance(args, dict):
                raise ValueError("Model requested an unknown tool or malformed arguments.")
            if name not in TOOL_MAP:
                raise ValueError(f"Tool {name} is not registered.")
            if name in {"issue_refund", "escalate_to_human"}:
                result = TOOL_MAP[name](runtime, **args, call_id=call["id"])
            else:
                result = TOOL_MAP[name](runtime, **args)
            observation = json.dumps(result, sort_keys=True)
            state["messages"].append({
                "role": "tool", "tool_call_id": call["id"], "name": name, "content": observation
            })
            step = {"tool_name": name, "arguments": args, "observation": result}
            state["steps"].append(step)
            state["tools_executed"].append(name)
            state["step_number"] += 1
            _save_checkpoint(run_id, state)
            if crash_after_tool_executions is not None and state["step_number"] == crash_after_tool_executions:
                raise AgentCrash(f"Simulated crash after checkpointed tool {name}.")
    final_answer = f"FALLBACK_MAX_ITERATIONS: hand off to a human; completed {len(state['steps'])} tool steps."
    return _agent_result(final_answer, state)


def _agent_result(final_answer: str, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": _status_from_answer(final_answer),
        "final_answer": final_answer,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "tools_executed": state["tools_executed"],
        "messages": state["messages"],
    }


def run_agent(
    user_message: str,
    scenario_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10,
    *,
    run_id: str | None = None,
    retriever: Retriever | None = None,
    crash_after_tool_executions: int | None = None,
) -> dict[str, Any]:
    selected_run_id = run_id or scenario_id
    active_retriever = retriever or Retriever()
    state = {
        "run_id": selected_run_id,
        "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": "AGENT: follow refund policy and use tools to verify every request."},
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "steps": [],
        "tools_executed": [],
    }
    _save_checkpoint(selected_run_id, state)
    return _continue_agent(
        selected_run_id, state, active_retriever, max_iterations,
        token_limit, time_limit_seconds, crash_after_tool_executions,
    )


def resume_agent(
    run_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10,
    retriever: Retriever | None = None,
) -> dict[str, Any]:
    state = json.loads(_checkpoint_path(run_id).read_text(encoding="utf-8"))
    return _continue_agent(
        run_id, state, retriever or Retriever(), max_iterations,
        token_limit, time_limit_seconds,
    )


def _refund_amount(result: dict[str, Any]) -> float:
    for step in reversed(result["steps"]):
        if step["tool_name"] in {"issue_refund", "escalate_to_human"}:
            return float(step["arguments"]["refund_inr"])
    match = re.search(r"INR\s+([\d,.]+)", result["final_answer"])
    return float(match.group(1).replace(",", "")) if match else 0.0


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    for name in ("agent_runs.jsonl", "refund_ledger.jsonl"):
        path = OUT / name
        if path.exists():
            path.unlink()

    overlap = evaluate_retrieval(10)
    no_overlap = evaluate_retrieval(0)
    retriever = overlap.pop("retriever")
    no_overlap.pop("retriever")
    if overlap["total_chunks"] != 15 or no_overlap["total_chunks"] != 13:
        raise AssertionError("Policy chunk counts differ from the assignment reference.")

    scenarios = pd.read_csv(RAW / "agent_scenarios.csv", keep_default_na=False)
    scenario_results = []
    s1 = scenarios.iloc[0].to_dict()
    try:
        run_agent(
            s1["message"], s1["scenario_id"], run_id=s1["scenario_id"],
            retriever=retriever, crash_after_tool_executions=2,
        )
    except AgentCrash:
        recovered = resume_agent(s1["scenario_id"], retriever=retriever)
    else:
        raise AssertionError("Crash-and-resume demonstration did not crash at the requested checkpoint.")
    for row in scenarios.to_dict(orient="records"):
        if row["scenario_id"] == s1["scenario_id"]:
            result = recovered
        else:
            result = run_agent(
                row["message"], row["scenario_id"], retriever=retriever, run_id=row["scenario_id"]
            )
        refund = _refund_amount(result)
        expected_refund = float(row["expected_refund_inr"])
        if abs(refund - expected_refund) > 0.01:
            raise AssertionError(
                f"{row['scenario_id']} refund mismatch: expected {expected_refund}, got {refund}."
            )
        scenario_results.append({
            "scenario_id": row["scenario_id"],
            "status": result["status"],
            "refund_inr": refund,
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"],
        })
    if scenario_results[0]["llm_calls"] != 5:
        raise AssertionError("S1 must take exactly five model calls.")
    if scenario_results[-1]["status"] != "FALLBACK_MAX_ITERATIONS" or scenario_results[-1]["llm_calls"] != 6:
        raise AssertionError("S7 must stop at the six-call iteration limit.")

    token_demo = run_agent(
        s1["message"], "S1-token-limit", run_id="S1-token-limit",
        retriever=retriever, token_limit=200,
    )
    timeout_demo = run_agent(
        s1["message"], "S1-timeout", run_id="S1-timeout",
        retriever=retriever, time_limit_seconds=0,
    )
    if token_demo["status"] != "FALLBACK_TOKEN_LIMIT" or timeout_demo["status"] != "FALLBACK_TIMEOUT":
        raise AssertionError("Stop-safely demonstrations returned unexpected statuses.")

    with (OUT / "agent_runs.jsonl").open("w", encoding="utf-8") as stream:
        for result in scenario_results:
            payload = {
                key: result[key] for key in
                ("scenario_id", "status", "refund_inr", "llm_calls", "tokens_used", "steps")
            }
            stream.write(json.dumps(payload) + "\n")
    report = {
        "overlap_retrieval": {key: value for key, value in overlap.items() if key != "rows"},
        "overlap_evaluation_rows": overlap["rows"],
        "zero_overlap_retrieval": {key: value for key, value in no_overlap.items() if key != "rows"},
        "zero_overlap_evaluation_rows": no_overlap["rows"],
        "scenario_results": scenario_results,
        "token_limit_demo": {
            "status": token_demo["status"],
            "llm_calls": token_demo["llm_calls"],
            "tokens_used": token_demo["tokens_used"],
        },
        "timeout_demo": {
            "status": timeout_demo["status"],
            "llm_calls": timeout_demo["llm_calls"],
            "tokens_used": timeout_demo["tokens_used"],
        },
        "crash_recovery": {
            "status": recovered["status"],
            "final_answer": recovered["final_answer"],
            "tools_executed": recovered["tools_executed"],
            "llm_calls": recovered["llm_calls"],
            "tokens_used": recovered["tokens_used"],
        },
        "tools_spec": TOOLS_SPEC,
    }
    (OUT / "part_d_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    run()
