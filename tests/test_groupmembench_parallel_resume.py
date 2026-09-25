import threading

from research import groupmembench_parallel_resume as parallel


def test_parallel_work_keeps_all_file_writes_on_caller_thread(monkeypatch):
    monkeypatch.setattr(parallel, "log", lambda *_args, **_kwargs: None)
    caller = threading.get_ident()
    written = []

    parallel._parallel(
        list(range(20)),
        lambda value: value * value,
        lambda value: written.append((threading.get_ident(), value)),
        workers=4,
        label="test",
    )

    assert {value for _thread, value in written} == {value * value for value in range(20)}
    assert {thread for thread, _value in written} == {caller}
