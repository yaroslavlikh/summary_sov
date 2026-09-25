"""Mechanical metrics of one /ask run, plus their delivery to Langfuse.

Nothing here calls a model: everything is computed from the final AskState, so
these numbers are cheap enough to record on every production answer and are the
same numbers the offline benchmark (tests/evals/compare_episodes.py) compares
between RAW and RAW+EPISODES.

They describe delivery and shape of the answer -- how much context was
assembled, how much of it came from episode expansion, how the answer cites it --
never whether the answer is correct. Quality is judged separately in
llm/answer_judges.py.
"""
from __future__ import annotations

import re
from typing import Any, Optional

_NO_ANSWER_MARKERS = ("не нашёл ответа", "не нашел ответа")
_MARKDOWN_RE = re.compile(r"(\*\*|\*[^*\n]+\*|^#{1,6}\s|```)", re.MULTILINE)
_SENTENCE_RE = re.compile(r"[.!?…]+(?:\s|$)")


def _tokens(text: str) -> int:
    """o200k token count when tiktoken is available (same counter the paper used),
    else a 4-chars-per-token estimate. Never fails an answer over a metric."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return round(len(text) / 4)


def cited_indices(answer_plain: Optional[str], legend_size: int) -> tuple[list[int], int]:
    """[N] markers in the model's answer, split into ones that point at a real
    context line and ones that do not (the generator inventing a number)."""
    valid, invalid = [], 0
    for match in re.finditer(r"\[(\d+)\]", answer_plain or ""):
        index = int(match.group(1))
        if 1 <= index <= legend_size:
            if index not in valid:
                valid.append(index)
        else:
            invalid += 1
    return valid, invalid


def collect(state: dict[str, Any], condition: str, latency_seconds: Optional[float] = None) -> dict[str, float]:
    """Metrics of a finished AskState. `condition` is "raw" or "episodes"."""
    window_rows = state.get("window_rows") or []
    episode_row_ids = set(state.get("episode_row_ids") or [])
    answer = state.get("answer_plain") or ""
    context_text = "\n".join(state.get("context_lines") or []) + (state.get("episode_context") or "")
    valid, invalid = cited_indices(answer, len(window_rows))
    cited_rows = [window_rows[index - 1][0] for index in valid]

    metrics = {
        "memory_condition": 1.0 if condition == "episodes" else 0.0,
        "candidates_raw": float(len(state.get("candidate_ids") or [])),
        "candidates_memory": float(len(state.get("candidate_event_ids") or [])),
        "memory_selected": float(len(state.get("match_event_ids") or [])),
        "context_messages": float(len(window_rows)),
        "context_tokens": float(_tokens(context_text)),
        "context_from_memory": float(len(episode_row_ids)),
        "citations": float(len(valid)),
        "citations_invented": float(invalid),
        "citations_from_memory": float(sum(1 for row_id in cited_rows if row_id in episode_row_ids)),
        "no_answer": float(not answer or any(marker in answer.lower() for marker in _NO_ANSWER_MARKERS)),
        "answer_tokens": float(_tokens(answer)),
        # The /ask prompt asks for 1-3 plain-text Russian sentences; these two say
        # whether the answer kept to that, without judging its content.
        "answer_sentences": float(len(_SENTENCE_RE.findall(answer.strip())) or bool(answer.strip())),
        "answer_has_markdown": float(bool(_MARKDOWN_RE.search(answer))),
    }
    if latency_seconds is not None:
        metrics["latency_seconds"] = round(float(latency_seconds), 3)
    return metrics


def push_scores_in_new_trace(name: str, metrics: dict[str, float], comment: Optional[str] = None,
                             payload: Optional[dict] = None) -> bool:
    """Scores for work that has no trace of its own (the memory worker): opens a
    span so the numbers have somewhere to hang, then records them on it."""
    if not _tracing_configured():
        return False
    try:
        from langfuse import get_client

        client = get_client()
        with client.start_as_current_observation(name=name, as_type="span", input=payload or {}):
            return push_scores(metrics, trace_id=client.get_current_trace_id(), comment=comment)
    except Exception as error:  # noqa: BLE001
        print(f"Не смог открыть трейс для метрик {name}: {error}")
        return False


def _tracing_configured() -> bool:
    import os

    import config  # noqa: F401 -- importing it loads .env, so this works outside main.py too

    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def push_scores(metrics: dict[str, float], trace_id: Optional[str] = None, comment: Optional[str] = None) -> bool:
    """Send metrics to Langfuse as scores on a trace. Returns False when tracing is
    not configured or the client rejects them -- an answer must never fail because
    a metric could not be recorded."""
    if not _tracing_configured():
        return False
    try:
        from langfuse import get_client

        client = get_client()
        target = trace_id or client.get_current_trace_id()
        if not target:
            return False
        for name, value in metrics.items():
            client.create_score(name=name, value=float(value), trace_id=target, data_type="NUMERIC", comment=comment)
        return True
    except Exception as error:  # noqa: BLE001 -- observability must not break /ask
        print(f"Не смог отправить метрики в Langfuse: {error}")
        return False
