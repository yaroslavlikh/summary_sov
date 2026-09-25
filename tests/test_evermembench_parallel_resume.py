import threading

import pandas as pd

from research import evermembench_parallel_resume as parallel


def test_parallel_work_keeps_jsonl_writes_on_caller_thread(monkeypatch):
    monkeypatch.setattr(parallel, "log", lambda *_args, **_kwargs: None)
    caller = threading.get_ident()
    written = []
    parallel._parallel(
        list(range(20)), lambda value: value * value,
        lambda value: written.append((threading.get_ident(), value)),
        workers=4, label="test",
    )
    assert {value for _thread, value in written} == {value * value for value in range(20)}
    assert {thread for thread, _value in written} == {caller}


def test_memory_shards_preserve_all_aggregate_counts():
    metas = [
        {"stats": {"networks": 1, "events_total": 3, "events_rejected": 1,
                   "rejected_reasons": {"bad_quote": 1}, "clusters_by_network": {"01": 2}}},
        {"stats": {"networks": 1, "events_total": 4, "events_rejected": 2,
                   "rejected_reasons": {"bad_quote": 1, "bad_type": 1},
                   "clusters_by_network": {"02": 3}}},
    ]
    merged = parallel._merge_stats(metas)
    assert merged["networks"] == 2
    assert merged["events_total"] == 7
    assert merged["events_rejected"] == 3
    assert merged["rejected_reasons"] == {"bad_quote": 2, "bad_type": 1}
    assert merged["clusters_by_network"] == {"01": 2, "02": 3}


def test_memory_worker_creates_shard_directory_before_opening_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(parallel, "SHARD_ROOT", tmp_path)
    monkeypatch.setattr(parallel.base, "load_messages", lambda: (
        pd.DataFrame([{"network_id": "01"}]), {}, {},
    ))

    class FakeAPI:
        def __init__(self, run_dir, **_kwargs):
            assert run_dir.is_dir()

    monkeypatch.setattr(parallel, "CachedAPI", FakeAPI)
    monkeypatch.setattr(parallel, "build_episode_snapshot", lambda *_args, **_kwargs: None)
    assert parallel._build_memory_shard("01") == {"topic": "01", "status": "ready"}
