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
    def summary(message):
        conn = sqlite3.connect('database/messages.sql')
        cursor = conn.cursor()

        row = cursor.execute(
            "SELECT last_id FROM summary_state WHERE user_id = ?",
            (message.chat.id,)
        ).fetchone()

        last_id = row[0] if row else 0

        cursor.execute("""
            SELECT id, user_name, message
            FROM messages
            WHERE user_id = ? AND id > ?
            ORDER BY id ASC
        """, (message.chat.id, last_id))

        messages = cursor.fetchall()

        if not messages:
            bot.send_message(message.chat.id, "Нет новых сообщений для суммаризации")
            conn.close()
            return

        prompt = ". ".join(f"{u}: {m}" for _, u, m in messages)

        res = send_prompt(prompt)

        last_msg_id = messages[-1][0]

        cursor.execute("""
            INSERT INTO summary_state (user_id, last_id)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET last_id = excluded.last_id
        """, (message.chat.id, last_msg_id))

        conn.commit()
        conn.close()
        print("last_id =", last_id)
        print("найдено сообщений:", len(messages))
        print(messages)
        bot.send_message(message.chat.id, res)
