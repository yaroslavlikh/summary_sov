import telebot
from config import get_key_bot
from handlers.handlers import load_handlers
from database.init_db import init_db

init_db()
token = get_key_bot()
bot = telebot.TeleBot(token=token)
load_handlers(bot)
bot.polling()