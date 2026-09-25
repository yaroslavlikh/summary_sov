"""Run the unchanged simple semantic-key gate with the 120B primary model."""
from pathlib import Path

from research import semantic_key_gate as gate
from research import semantic_slot_linking_v2 as v2
from research.socialmembench_pilot import _parse_json_object


def _primary_only(prompt: str, _model: str = "primary") -> dict:
    from langchain_groq import ChatGroq
    from llm.groq_client import API_KEY, PRIMARY_MODEL

    message = ChatGroq(
        model=PRIMARY_MODEL, api_key=API_KEY, temperature=0,
        max_retries=2, max_tokens=8192,
    ).invoke(prompt)
    return _parse_json_object(message.content if isinstance(message.content, str) else "") or {}


def main() -> None:
    out = Path(".research_runs/semantic_key_gate_primary_v1")
    gate.OUT_DIR = out
    for attribute, filename in {
        "RANKINGS": "rankings.jsonl", "GATE_CACHE": "gate_cache.jsonl",
        "DECISIONS": "decisions.jsonl", "REPORT": "report.md",
        "CONFIG": "config.json", "RUN_LOG": "run.log",
    }.items():
        setattr(gate, attribute, out / filename)
    gate.MODEL = "primary"
    gate.SCHEMA_VERSION = "semantic_key_gate_primary_v1"
    v2._call_llm = _primary_only
    gate.main()


if __name__ == "__main__":
    main()
