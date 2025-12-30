import os
from dotenv import load_dotenv


load_dotenv()


def get_key_bot():
    try:
        token = os.getenv('BOT_TOKEN')
    except Exception as e:
        print(f"Ошибка при получении токена бота: {e}\nУбедитесь, что вы запускаете проект из корневой папки и в .env есть токен")
    return token


def get_token_gemini():
    try:
        token = os.getenv('GEMINI_API_KEY')
    except Exception as e:
        print(f"Ошибка при получении токена gemini: {e}\nУбедитесь, что вы запускаете проект из корневой папки и в .env есть токен")
    return token

def get_flask_url():
    try:
        url = os.getenv('WEBHOOK_URL')
    except Exception as e:
        print(f"Ошибка при получении WEBHOOK_URL: {e}\nУбедитесь, что вы запускаете проект из корневой папки и в .env есть WEBHOOK_URL")
    return url