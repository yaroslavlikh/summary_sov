import os
from dotenv import load_dotenv


load_dotenv()


def get_key_bot():
    try:
        token = os.getenv('BOT_TOKEN')
        if not token:
            print("Ошибка: BOT_TOKEN не найден в .env файле")
        return token
    except Exception as e:
        print(f"Ошибка при получении токена бота: {e}\nУбедитесь, что вы запускаете проект из корневой папки и в .env есть токен")
        return None


def get_token_gemini():
    try:
        token = os.getenv('GEMINI_API_KEY')
        if not token:
            print("Ошибка: GEMINI_API_KEY не найден в .env файле")
        return token
    except Exception as e:
        print(f"Ошибка при получении токена gemini: {e}\nУбедитесь, что вы запускаете проект из корневой папки и в .env есть токен")
        return None