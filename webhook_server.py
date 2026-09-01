import telebot
from flask import Flask, abort, request


def create_app(bot, bot_token):
    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def health():
        return 'ok', 200

    @app.route(f'/webhook/{bot_token}', methods=['POST'])
    def webhook():
        if request.headers.get('content-type') != 'application/json':
            abort(403)
        update = telebot.types.Update.de_json(request.get_data().decode('utf-8'))
        bot.process_new_updates([update])
        return '', 200

    return app
