import os
import config
import telebot
from handlers.handlers import load_handlers
from database.init_db import init_db


def start_up():
    try:
        init_db()
        TELEGRAM_TOKEN = config.get_key_bot()
        bot = telebot.TeleBot(TELEGRAM_TOKEN)
        load_handlers(bot)
        print("Бот запущен")
        bot.polling()
    except Exception as e:
        print(f"Произошла ошибка при запуске бота: {e}")


if __name__ == '__main__':
    start_up()
