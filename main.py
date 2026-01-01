import telebot
from config import get_key_bot
from handlers.handlers import load_handlers
from database.init_db import init_db

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
        print("Бот запущен...")
        bot.polling()
    except Exception as e:
        print(f"Ошибка при запуске бота: {e}")

if __name__ == "__main__":
    start_app()