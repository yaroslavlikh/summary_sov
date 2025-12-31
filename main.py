import telebot
from config import get_key_bot
from handlers.handlers import load_handlers
from database.init_db import init_db

import urllib3
import socket

# Отключаем IPv6
urllib3.util.connection.HAS_IPV6 = False

init_db()
token = get_key_bot()
bot = telebot.TeleBot(token=token)
load_handlers(bot)
print("Бот запущен...")
bot.polling()