import os

import telebot
from waitress import serve

from config import get_key_bot
from handlers.handlers import load_handlers
from database.init_db import init_db
from scheduler import start_scheduler
from webhook_server import create_app

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
    domain = os.getenv('RAILWAY_PUBLIC_DOMAIN')
    if not domain:
        print("Ошибка: RAILWAY_PUBLIC_DOMAIN не найден — сервис должен быть exposed (сгенерируй domain в Railway)")
        return

    try:
        webhook_url = f"https://{domain}/webhook/{token}"
        bot.remove_webhook()
        bot.set_webhook(url=webhook_url)
        print(f"Webhook установлен: {webhook_url}")
    except Exception as e:
        print(f"Ошибка при установке webhook: {e}")
        return

    try:
        app = create_app(bot, token)
        port = int(os.getenv('PORT', 8080))
        print(f"Бот запущен... (слушаю на порту {port})")
        serve(app, host='0.0.0.0', port=port)
    except Exception as e:
        print(f"Ошибка при запуске веб-сервера: {e}")

if __name__ == "__main__":
    start_app()