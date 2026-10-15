from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import pandas as pd

import mock_llm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
ALLOWED = {"billing", "delivery", "technical", "fraud"}
PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("phone", re.compile(r"\b[6-9]\d{9}\b")),
    ("order_id", re.compile(r"ORD-\d{6}")),
]

ZERO_SHOT_PROMPT = (
    "CLASSIFY each support ticket into exactly one category: billing, delivery, technical, or fraud. "
    'Return JSON only with this shape: {"category": "...", "reason": "..."}'
)
FEW_SHOT_PROMPT = (
    "CLASSIFY each support ticket into exactly one category: billing, delivery, technical, or fraud. "
    'Return JSON only with this shape: {"category": "...", "reason": "..."}\n'
    "EXAMPLES:\n"
    'Ticket: "Unauthorized charge on my account."\n'
    'Response: {"category": "fraud", "reason": "unauthorized charge"}\n'
    'Ticket: "The delivery is late."\n'
    'Response: {"category": "delivery", "reason": "late delivery"}'
)


def redact(text: str) -> tuple[str, dict[str, int]]:
    redacted = text
    counts = {}
    for name, pattern in PII_PATTERNS:
        redacted, count = pattern.subn(
            {"email": "[EMAIL]", "phone": "[PHONE]", "order_id": "[ORDER_ID]"}[name],
            redacted,
        )
        counts[name] = count
    if any(pattern.search(redacted) for _, pattern in PII_PATTERNS):
        raise AssertionError("PII remains after redaction.")
    return redacted, counts


def _route(status: str, category: str | None, is_premium: bool) -> tuple[str, str]:
    if status in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "permanent_error"}:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


def _validate_response(content: str | None) -> tuple[str, str | None, str | None]:
    if content is None:
        return "flagged_invalid_output", None, None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        words = re.findall(r"[a-z]+", content.lower())
        matches = ALLOWED.intersection(words)
        if len(matches) == 1:
            category = next(iter(matches))
            return "classified_keyword", category, "keyword extracted from non-JSON response"
        return "flagged_invalid_output", None, None
    category = parsed.get("category") if isinstance(parsed, dict) else None
    reason = parsed.get("reason") if isinstance(parsed, dict) else None
    if category in ALLOWED and isinstance(reason, str):
        return "classified_json", category, reason
    return "flagged_invalid_output", None, None


def run_full(prompt: str, token_budget: int | None = None, retry_log: list[str] | None = None) -> dict:
    tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)
    segments = pd.read_csv(OUT / "customer_segments.csv")
    premium_by_customer = segments.set_index("customer_id")["is_premium"].to_dict()
    mock_llm.reset()
    rng = random.Random(42)
    total_tokens = 0
    results = []
    retry_lines = retry_log if retry_log is not None else []
    redaction_counts = {"email": 0, "phone": 0, "order_id": 0}
    response_status = {"classified_json": 0, "classified_keyword": 0, "flagged_invalid_output": 0}
    exceptions = 0
    for ticket in tickets.to_dict(orient="records"):
        text = str(ticket["text"])
        status: str
        category: str | None = None
        tokens_used = 0
        if not text.strip():
            status = "skipped_empty"
        elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        else:
            redacted, counts = redact(text)
            for key, count in counts.items():
                redaction_counts[key] += count
            if any(pattern.search(redacted) for _, pattern in PII_PATTERNS):
                raise AssertionError(f"Unredacted PII in ticket {ticket['ticket_id']}.")
            user_message = {"role": "user", "content": redacted}
            if user_message["content"] != redacted:
                raise AssertionError("The model user prompt must consist only of redacted ticket text.")
            system_message = {"role": "system", "content": prompt}
            if token_budget is not None and total_tokens >= token_budget:
                status = "budget_exhausted"
            else:
                response = None
                for attempt in range(4):
                    try:
                        response = mock_llm.chat([system_message, user_message])
                        break
                    except mock_llm.TransientLLMError:
                        exceptions += 1
                        if attempt >= 3:
                            status = "permanent_error"
                            break
                        delay = BASE_DELAY * (2 ** attempt) * rng.uniform(0.9, 1.15)
                        retry_lines.append(
                            f"{ticket['ticket_id']} retry={attempt} wait={delay:.6f}s"
                        )
                        time.sleep(delay)
                if response is not None:
                    tokens_used = int(response.usage.total_tokens)
                    total_tokens += tokens_used
                    status, category, _ = _validate_response(response.choices[0].message.content)
                    response_status[status] += 1
                elif "status" not in locals() or status != "permanent_error":
                    status = "permanent_error"
        is_premium = bool(premium_by_customer.get(ticket["customer_id"], False))
        route, priority = _route(status, category, is_premium)
        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "status": status,
            "category": category or "",
            "segment_name": _segment_name(segments, ticket["customer_id"]),
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used,
            "_true_category": ticket["true_category"],
        })

    classified = [row for row in results if row["status"] in {"classified_json", "classified_keyword"}]
    correct = sum(row["category"] == row["_true_category"] for row in classified)
    demo = pd.DataFrame([{key: value for key, value in row.items() if key != "_true_category"} for row in results])
    return {
        "results": demo,
        "raw_results": results,
        "tickets_classified": len(classified),
        "correct_classifications": int(correct),
        "accuracy": float(correct / len(classified)) if classified else 0.0,
        "total_tokens": total_tokens,
        "tokens_per_classified_ticket": float(total_tokens / len(classified)) if classified else 0.0,
        "status_counts": pd.Series([row["status"] for row in results]).value_counts().to_dict(),
        "response_status_counts": response_status,
        "transient_exceptions": exceptions,
        "retry_log": retry_lines,
        "redaction_counts": redaction_counts,
        "redaction_total": int(sum(redaction_counts.values())),
        "remaining_pii_matches": 0,
    }


def _segment_name(segments: pd.DataFrame, customer_id: str) -> str:
    found = segments.loc[segments["customer_id"].eq(customer_id), "segment_name"]
    return str(found.iloc[0]) if not found.empty else ""


def _write_results(report: dict, path: Path) -> None:
    frame = report["results"].drop(columns=["_true_category"], errors="ignore")
    frame.to_csv(path, index=False)


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    retry_lines: list[str] = []
    zero_shot = run_full(ZERO_SHOT_PROMPT, retry_log=retry_lines)
    zero_matrix = pd.crosstab(
        pd.Series([row["_true_category"] for row in zero_shot["raw_results"] if row["status"] in {"classified_json", "classified_keyword"}], name="actual"),
        pd.Series([row["category"] for row in zero_shot["raw_results"] if row["status"] in {"classified_json", "classified_keyword"}], name="predicted"),
        dropna=False,
    ).reindex(index=sorted(ALLOWED), columns=sorted(ALLOWED), fill_value=0)
    few_shot = run_full(FEW_SHOT_PROMPT, retry_log=retry_lines)
    budget_demo = run_full(ZERO_SHOT_PROMPT, token_budget=6000, retry_log=retry_lines)
    final = run_full(FEW_SHOT_PROMPT, token_budget=100000, retry_log=retry_lines)
    _write_results(budget_demo, OUT / "triage_budget_demo.csv")
    _write_results(final, OUT / "triage_results.csv")
    (OUT / "triage_retries.log").write_text("\n".join(retry_lines) + ("\n" if retry_lines else ""), encoding="utf-8")
    report = {
        "zero_shot": {
            key: zero_shot[key] for key in (
                "tickets_classified", "correct_classifications", "accuracy", "total_tokens",
                "tokens_per_classified_ticket", "status_counts", "response_status_counts",
                "transient_exceptions", "redaction_counts", "redaction_total",
            )
        },
        "zero_shot_confusion": zero_matrix.to_dict(),
        "few_shot": {
            key: few_shot[key] for key in (
                "tickets_classified", "correct_classifications", "accuracy", "total_tokens",
                "tokens_per_classified_ticket", "status_counts", "response_status_counts",
                "transient_exceptions", "redaction_counts", "redaction_total",
            )
        },
        "budget_demo": {
            "budget": 6000,
            "tokens_used": budget_demo["total_tokens"],
            "exhausted_tickets": int(budget_demo["status_counts"].get("budget_exhausted", 0)),
            "status_counts": budget_demo["status_counts"],
        },
        "final_few_shot": {
            "budget": 100000,
            "tokens_used": final["total_tokens"],
            "rows": int(len(final["results"])),
            "status_counts": final["status_counts"],
        },
        "retry_log_entries": len(retry_lines),
        "retry_log": retry_lines,
    }
    (OUT / "part_c_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    run()
