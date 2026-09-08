import telebot
from flask import Flask, abort, request


def create_app(bot, webhook_secret):
    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def health():
        return 'ok', 200

    @app.route('/webhook', methods=['POST'])
    def webhook():
        if not request.is_json or request.headers.get('X-Telegram-Bot-Api-Secret-Token') != webhook_secret:
            abort(403)
        update = telebot.types.Update.de_json(request.get_data().decode('utf-8'))
        bot.process_new_updates([update])
        return '', 200

    return app
