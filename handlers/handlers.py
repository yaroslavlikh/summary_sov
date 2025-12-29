import sqlite3
from llm.gemini import send_promte


def load_handlers(bot):

    @bot.message_handler(func=lambda mess: not mess.text[0] == "/" and mess.content_type == 'text')
    def save_messages(message):
        print(f"Получено сообщение: {message.text}")
        user_name = message.from_user.first_name
        conn = sqlite3.connect('database/messages.sql')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO messages (user_id, user_name, message) VALUES (?, ?, ?)",
                       (message.from_user.id, user_name, message.text))
        conn.commit()

        conn.close()


    @bot.message_handler(commands=['summary'])
    def summary(message, limit: str = 150):
        conn = sqlite3.connect('database/messages.sql')
        cursor = conn.cursor()
        _ = message.text.split()
        if len(_) > 1:
            if _[1].isdigit():
                limit = min(int(_[1]), int(cursor.execute("SELECT MAX(id) FROM messages").fetchone()[0]))

        print("Создание краткого содержания")

        cursor.execute("""SELECT user_name, message FROM messages WHERE user_id = ?
                          ORDER BY id DESC 
                          LIMIT ?""",
                       (message.from_user.id, limit))
        messages = cursor.fetchall()
        conn.close()
        prompt = ". ".join([f"{msg[0]}: {msg[1]}" for msg in messages])
        res = f"{send_promte(prompt)}\n - Ну и Гордей хуесос"
        bot.send_message(message.chat.id, res)