import telebot
from flask import Flask, request, abort
from config import get_key_bot, get_flask_url
from handlers.handlers import load_handlers
from database.init_db import init_db

app = Flask(__name__)
bot = None

def start_app(token, webhook_url):
    global bot
    try:
        init_db()
        print("База данных инициализирована")
    except Exception as e:
        print(f"Ошибка инициализации базы данных: {e}")
        return
    
    try:
        bot = telebot.TeleBot(token=token)
        load_handlers(bot)
        print("Обработчики загружены")
    except Exception as e:
        print(f"Ошибка инициализации бота: {e}")
        return
    
    try:
        # Удаляем старый webhook
        bot.delete_webhook()
        # Устанавливаем новый webhook
        webhook_path = f"{webhook_url}/webhook"
        bot.set_webhook(url=webhook_path)
        print(f"Webhook установлен: {webhook_path}")
    except Exception as e:
        print(f"Ошибка установки webhook: {e}")
        return

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return '', 200
    else:
        abort(403)

@app.route('/', methods=['GET'])
def index():
    return 'Бот работает!', 200

if __name__ == '__main__':
    try:
        token = get_key_bot()
        if not token:
            print("Ошибка: Токен бота не найден. Убедитесь, что BOT_TOKEN установлен в .env файле")
            exit(1)
        
        webhook_url = get_flask_url()
        if not webhook_url:
            print("Ошибка: WEBHOOK_URL не найден. Убедитесь, что WEBHOOK_URL установлен в .env файле")
            exit(1)
        
        start_app(token, webhook_url)
        print("Запуск Flask сервера...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
       print(f"Ошибка: {e}")
       exit(1)