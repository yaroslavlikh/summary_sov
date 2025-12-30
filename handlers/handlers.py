import sqlite3
from llm.gemini import send_prompt
#from llm.llama import send_prompt
counter = 0

def load_handlers(bot):
    global counter

    @bot.message_handler(func=lambda mess: not mess.text[0] == "/" and mess.content_type == 'text')
    def save_messages(message):
        global counter
        if message.from_user.username != "sglypa_tg_bot":
            print(f"Получено сообщение: {message.text}")
            user_name = message.from_user.first_name
            counter += 1
            conn = sqlite3.connect('database/messages.sql')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO messages (user_id, user_name, message) VALUES (?, ?, ?)",
                        (message.chat.id, user_name, message.text))
            conn.commit()
            if counter == 150:
                summary(message)
            conn.close()


    @bot.message_handler(commands=['help'])
    def help_command(message):
        help_text = """
        Доступные команды:

        /summary [количество] - Создать краткое содержание последних сообщений
        Пример: /summary 50 - создаст краткое содержание последних 50 сообщений
        По умолчанию: 100 сообщений

        /help - Показать это сообщение

        Бот автоматически сохраняет все ваши текстовые сообщения для последующего создания краткого содержания.
        """
        bot.send_message(message.chat.id, help_text.strip())


    @bot.message_handler(commands=['summary'])
    def summary(message, limit=100):
        global counter
        conn = sqlite3.connect('database/messages.sql')
        cursor = conn.cursor()
        _ = message.text.split()
        if len(_) > 1:
            if _[1].isdigit():
                limit = min(int(_[1]), int(cursor.execute("SELECT MAX(id) FROM messages").fetchone()[0]))

        print("Создание краткого содержания")

        cursor.execute("""SELECT user_name, message
                    FROM (
                        SELECT id, user_name, message
                        FROM messages
                        WHERE user_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC;""",
                       (message.chat.id, limit))
        messages = cursor.fetchall()
        conn.close()
        prompt = ". ".join([f"{msg[0]}: {msg[1]}" for msg in messages])
        print(prompt)
        try:
            res = f"{send_prompt(prompt)}\nИ напоминание от нашей компании Google: Гордей хуесос"
        except Exception as e:
            print(f"Ошибка при суммаризации: {e}")
            res = "Произошла ошибка при суммаризации ваших сообщений. Попробуйте позже"
        
        bot.send_message(message.chat.id, res)