import os
import config
import telebot


def start_up():
    TELEGRAM_TOKEN = config.get_key_bot()
    bot = telebot.TeleBot(TELEGRAM_TOKEN)
    bot.polling()


