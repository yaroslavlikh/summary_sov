import telebot
from flask import Flask, request
import config
from handlers.handlers import load_handlers
from database.init_db import init_db

TOKEN = config.get_key_bot()

bot = telebot.TeleBot(TOKEN)
load_handlers(bot)

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    update = telebot.types.Update.de_json(
        request.get_data().decode("utf-8")
    )
    bot.process_new_updates([update])
    return "OK", 200



@app.route("/")
def index():
    return "Bot is alive", 200


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=8080)