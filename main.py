import telebot
from config import get_key_bot
from handlers.handlers import load_handlers
from database.init_db import init_db

def start_app(token):
    try:
        init_db()
        print("База данных инициализирована")
    except Exception as e:
        print(f"Ошибка инициализации базы данных: {e}")
        return
    try:
        bot = telebot.TeleBot(token=token)
    except Exception as e:
       print(f"Ошибка инициализации бота: {e}")
       return
    try:
        load_handlers(bot)
    except Exception as e:
        print(f"Ошибка загрузки обработчиков: {e}")
        return
    try:
        print("Бот работает...")
        bot.infinity_polling()
    except Exception as e:
        print(f"Ошибка запуска бота: {e}")


if __name__ == '__main__':
    try:
        token = get_key_bot()
    except Exception as e:
       print(f"Ошибка получения токена бота: {e}")
    start_app(token)