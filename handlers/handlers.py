import telebot
from llm.gemini import send_promte


def load_handlers(bot):

    @bot.message_handler(func=lambda mess: not mess.text[0] == "/")
    def save_messages(message):
        print(f"Получено сообщение: {message.text}")
        with open("handlers/messages.txt", "a", encoding="utf-8") as file:
            file.write(message.text + "\n")


    @bot.message_handler(commands=['summary'])
    def summary(message):
        print("Создание краткого содержания")
        with open("handlers/messages.txt", "r", encoding="utf-8") as file:
            prompt = ". ".join(file.readlines())
        res = send_promte(prompt)
        bot.send_message(message.chat.id, res)