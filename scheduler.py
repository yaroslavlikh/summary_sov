import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from config import get_timezone, memory_worker_enabled
from database.db import get_conn
from episodic_memory import run_for_chat
from handlers.handlers import generate_and_send_summary, get_chat_ids

SCHEDULED_TIMES = ("14:00", "22:00")
CHECK_INTERVAL_SECONDS = 20
MEMORY_INTERVAL_SECONDS = 10 * 60


def claim_run(run_key):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO scheduler_runs (run_key) VALUES (%s) ON CONFLICT DO NOTHING RETURNING run_key",
            (run_key,),
        )
        claimed = cursor.fetchone() is not None
        conn.commit()
        return claimed


def run_scheduled_summaries(bot):
    for chat_id in get_chat_ids():
        try:
            generate_and_send_summary(bot, chat_id)
        except Exception as e:
            print(f"Ошибка при плановой суммаризации чата {chat_id}: {e}")


def _scheduler_loop(bot):
    tz = ZoneInfo(get_timezone())
    last_run_key = None
    while True:
        try:
            now = datetime.now(tz)
            current_time = now.strftime("%H:%M")
            run_key = f"{now.date()}:{current_time}"
            if current_time in SCHEDULED_TIMES and run_key != last_run_key and claim_run(run_key):
                last_run_key = run_key
                print(f"Плановая суммаризация ({current_time} {get_timezone()})")
                run_scheduled_summaries(bot)
        except Exception as e:
            print(f"Ошибка в планировщике суммаризаций: {e}")
        time.sleep(CHECK_INTERVAL_SECONDS)


def run_memory_worker():
    for chat_id in get_chat_ids():
        try:
            totals = run_for_chat(chat_id)
        except Exception as e:
            # The cursor only advances after a complete pass and every block is
            # committed atomically, so the next tick resumes where this one failed.
            print(f"Ошибка при построении памяти чата {chat_id}: {e}")
            _report_memory_pass(chat_id, {"blocks": 0, "events": 0, "failed": 1})
            continue
        if totals["blocks"]:
            _report_memory_pass(chat_id, {**totals, "failed": 0})


def _report_memory_pass(chat_id, totals):
    """Health of memory construction alongside the /ask metrics in Langfuse: how
    much was built, and what the validator threw away."""
    try:
        import ask_metrics
        from episodic_memory import memory_stats

        stats = memory_stats(chat_id)
        scores = {
            "memory_blocks_processed": float(totals.get("blocks", 0)),
            "memory_events_added": float(totals.get("events", 0)),
            "memory_pass_failed": float(totals.get("failed", 0)),
            "memory_events_total": float(stats["events"]),
            "memory_episodes_total": float(stats["episodes"]),
            "memory_events_rejected_total": float(sum(stats["rejected"].values())),
            "memory_events_per_100_messages": round(100 * stats["events"] / max(stats["messages"], 1), 2),
        }
        ask_metrics.push_scores_in_new_trace(
            "memory-worker", scores, comment=f"chat {chat_id}: {stats['rejected'] or 'без отклонений'}",
            payload={"chat_id": chat_id})
    except Exception as e:
        print(f"Не смог отправить метрики памяти: {e}")


def _memory_loop():
    while True:
        try:
            run_memory_worker()
        except Exception as e:
            print(f"Ошибка в воркере памяти: {e}")
        time.sleep(MEMORY_INTERVAL_SECONDS)


def start_scheduler(bot):
    thread = threading.Thread(target=_scheduler_loop, args=(bot,), daemon=True)
    thread.start()
    if memory_worker_enabled():
        threading.Thread(target=_memory_loop, daemon=True).start()
    return thread
