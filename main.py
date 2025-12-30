import telebot
import config
from handlers.handlers import load_handlers
from database.init_db import init_db

import requests

import socket
socket.setdefaulttimeout(30)

print("TEST TELEGRAM CONNECT")
print(requests.get("https://api.telegram.org").status_code)


def start_up():
    try:
        init_db()

        telebot.apihelper.SESSION = None
        telebot.apihelper.API_URL = "https://api.telegram.org/bot{0}/{1}"
        TELEGRAM_TOKEN = config.get_key_bot()
        bot = telebot.TeleBot(TELEGRAM_TOKEN)

        load_handlers(bot)

        print("Бот запущен")
        bot.infinity_polling(
            timeout=60,
            long_polling_timeout=60,
            skip_pending=True
        )


    except Exception as e:
        print(f"Ошибка запуска: {e}")

if __name__ == '__main__':
    start_up()
