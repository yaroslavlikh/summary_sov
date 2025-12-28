import sqlite3
from llm.gemini import send_promte


def load_handlers(bot):

    @bot.message_handler(func=lambda mess: not mess.text[0] == "/" and mess.content_type == 'text')
    def save_messages(message):
        print(f"Получено сообщение: {message.text}")
        conn = sqlite3.connect('tasks.sql')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO tasks (user_id, message) VALUES (?, ?)",
                       (message.from_user.id, message.text))
        conn.commit()
        conn.close()


    @bot.message_handler(commands=['summary'])
    def summary(message):
        print("Создание краткого содержания")
        conn = sqlite3.connect('tasks.sql')
        cursor = conn.cursor()
        cursor.execute("SELECT message FROM tasks WHERE user_id = ?",
                       (message.from_user.id,))
        messages = cursor.fetchall()
        conn.close()
        prompt = ". ".join([msg[0] for msg in messages])
        res = send_promte(prompt)
        bot.send_message(message.chat.id, res)
