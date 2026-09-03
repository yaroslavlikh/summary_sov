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


def get_groq_api_key():
    try:
        token = os.getenv('GROQ_API_KEY')
        if not token:
            print("Ошибка: GROQ_API_KEY не найден в .env файле")
        return token
    except Exception as e:
        print(f"Ошибка при получении ключа Groq: {e}\nУбедитесь, что вы запускаете проект из корневой папки и в .env есть токен")
        return None


def get_timezone():
    return os.getenv('TIMEZONE', 'Europe/Kyiv')


def get_database_url():
    url = os.getenv('DATABASE_URL')
    if not url:
        print("Ошибка: DATABASE_URL не найден в .env файле")
    return url


def get_encryption_key():
    key = os.getenv('MESSAGE_ENCRYPTION_KEY')
    if not key:
        print("Ошибка: MESSAGE_ENCRYPTION_KEY не найден в .env файле")
    return key