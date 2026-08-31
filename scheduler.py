import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from config import get_timezone
from handlers.handlers import generate_and_send_summary, get_chat_ids

SCHEDULED_TIMES = ("14:00", "22:00")
CHECK_INTERVAL_SECONDS = 20


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
            run_key = (now.date(), current_time)
            if current_time in SCHEDULED_TIMES and run_key != last_run_key:
                last_run_key = run_key
                print(f"Плановая суммаризация ({current_time} {get_timezone()})")
                run_scheduled_summaries(bot)
        except Exception as e:
            print(f"Ошибка в планировщике суммаризаций: {e}")
        time.sleep(CHECK_INTERVAL_SECONDS)


def start_scheduler(bot):
    thread = threading.Thread(target=_scheduler_loop, args=(bot,), daemon=True)
    thread.start()
    return thread
