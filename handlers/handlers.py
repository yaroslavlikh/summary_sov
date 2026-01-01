import sqlite3
from llm.gemini import send_prompt
#from llm.llama import send_prompt
counter = 0
last_summary_id = 0

def load_handlers(bot):

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
                summary("/summary 150")
            conn.close()

    @bot.message_handler(commands=['help'])
    def help_command(message):
        help_text = """
        Доступные команды:

        /summary [количество] - Создать краткое содержание последних сообщений
        Пример: /summary 50 - создаст краткое содержание последних 50 сообщений
        По умолчанию: с последнего саммари

        /help - Показать это сообщение

        Бот автоматически сохраняет все ваши текстовые сообщения для последующего создания краткого содержания.
        """
        bot.send_message(message.chat.id, help_text.strip())

    def get_last_summary_id(user_id):
        conn = sqlite3.connect('database/messages.sql')
        cursor = conn.cursor()
        cursor.execute("SELECT last_id FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 0

    @bot.message_handler(commands=['summary'])
    def summary(message):
        global last_summary_id, counter
        conn = sqlite3.connect('database/messages.sql')
        cursor = conn.cursor()
        now_id_message = cursor.execute("SELECT id FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT 1", (message.chat.id,)).fetchone()[0]
        N = now_id_message - last_summary_id
        last_summary_id = now_id_message
        if len(message.text.split()) > 1:
            _ = message.text.split()[1]
            if _.isdigit():
                N = int(_)
                
        if N <= 10:
            bot.send_message(message.chat.id, f"Сообщений было написано слишком мало для суммаризации: {N}")
        else:
            cursor.execute("""
                SELECT user_name, message
                FROM messages
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
            """, (message.chat.id, N))

            messages = cursor.fetchall()[::-1]
            if not messages:
                bot.send_message(message.chat.id, "Нет сообщений для суммаризации")
                return
            prompt = ". ".join(f"{u}: {m}" for u, m in messages)
            res = send_prompt(prompt)
            counter = 0
            bot.send_message(message.chat.id, res)
        conn.close()