import os
from dotenv import load_dotenv


load_dotenv()


def get_key_bot():
    token = os.getenv('BOT_TOKEN')
    return token


def get_token_gemini():
    token = os.getenv('GEMINI_API_KEY')
    return token
