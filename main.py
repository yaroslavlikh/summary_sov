import telebot
from flask import Flask, request
import config
from handlers.handlers import load_handlers

TOKEN = config.get_key_bot()
WEBHOOK_URL = config.get_webhook_url()

bot = telebot.TeleBot(TOKEN)
load_handlers(bot)

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    json_str = request.get_data().decode("UTF-8")
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return "OK", 200


@app.route("/")
def index():
    return "Bot is alive", 200


if __name__ == "__main__":
    bot.remove_webhook()
    bot.set_webhook(url=WEBHOOK_URL)

    print("Webhook установлен:", WEBHOOK_URL)

    app.run(host="0.0.0.0", port=8080)