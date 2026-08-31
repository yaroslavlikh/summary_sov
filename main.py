import telebot
from config import get_key_bot
from handlers.handlers import load_handlers
from database.init_db import init_db
from scheduler import start_scheduler

import urllib3
urllib3.util.connection.HAS_IPV6 = False

def start_app():
    try:
        init_db()
    except Exception as e:
        print(f"Ошибка инициализации базы данных: {e}")
        return
    try:
        token = get_key_bot()
        if not token:
            print("Ошибка: BOT_TOKEN не найден. Проверьте файл .env")
            return
    except Exception as e:
        print(f"Ошибка получения токена бота: {e}")
        return
    try:
        bot = telebot.TeleBot(token=token)
    except Exception as e:
        print(f"Ошибка создания бота: {e}")
        return  
    try:
        load_handlers(bot)
    except Exception as e:
        print(f"Ошибка загрузки обработчиков: {e}")
        return
    try:
        start_scheduler(bot)
    except Exception as e:
        print(f"Ошибка запуска планировщика саммари: {e}")
        return
    try:
        print("Бот запущен...")
        bot.infinity_polling()
    except Exception as e:
        print(f"Ошибка при запуске бота: {e}")

if __name__ == "__main__":
    start_app()