import sqlite3
from llm.gemini import send_prompt
#from llm.llama import send_prompt
counter = 0

def load_handlers(bot):
    global counter

    @bot.message_handler(func=lambda mess: mess.text and not mess.text.startswith("/"))
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
        N = 100
        if len(message.text.split()) > 1:
            _ = message.text.split()[1]
            if _.isdigit():
                N = _
        
        conn = sqlite3.connect('database/messages.sql')
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_name, message
            FROM messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (message.chat.id, N))

        messages = cursor.fetchall()[::-1]  # обращаем обратно в хронологический порядок

        conn.close()

        if not messages:
            bot.send_message(message.chat.id, "Нет сообщений для суммаризации")
            return

        prompt = ". ".join(f"{u}: {m}" for u, m in messages)
        res = send_prompt(prompt)

        bot.send_message(message.chat.id, res)
