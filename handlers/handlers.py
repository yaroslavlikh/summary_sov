import telebot
from google.auth import message
from ..llm.gemini import send_promte


def load_handlers(bot):
    file = open("messages.txt", "w")

    @bot.message_handler(func=lambda mess: True)
    def save_messages(message):
        message = message.text
        file.write(message + "\n")

    @bot.message_handler(commands=['summary'])
    def summary(message):
        promte = ". ".join(file.readlines())
        res = send_promte(promte)
        file.truncate(0)
        bot.send_message(message.chat.id, res)
