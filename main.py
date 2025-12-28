import os
import config
import telebot
from handlers.handlers import load_handlers


def start_up():
    TELEGRAM_TOKEN = config.get_key_bot()
    bot = telebot.TeleBot(TELEGRAM_TOKEN)
    load_handlers(bot)
    print("Бот запущен")
    bot.polling()


if __name__ == '__main__':
    start_up()
