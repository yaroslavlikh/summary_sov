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


def _flag(name, default):
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


def episodes_in_ask_enabled():
    """Episodes ranked in /ask next to raw messages. Off until
    tests/evals/compare_episodes.py passes docs/EPISODES_ROLLOUT_PROTOCOL.md."""
    return _flag('MEMORY_EPISODES_IN_ASK', False)


def memory_worker_enabled():
    """Background construction of episodic memory (scheduler.py)."""
    return _flag('MEMORY_WORKER_ENABLED', True)
