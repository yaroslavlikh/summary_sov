import telebot
from flask import Flask, request
import config
from handlers.handlers import load_handlers
from database.init_db import init_db

TOKEN = config.get_key_bot()
bot = telebot.TeleBot(TOKEN)

app = Flask(__name__)

load_handlers(bot)
init_db()

@app.route("/")
def health():
    return "BOT IS RUNNING", 200

@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    update = telebot.types.Update.de_json(
        request.get_data(as_text=True)
    )
    bot.process_new_updates([update])
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)