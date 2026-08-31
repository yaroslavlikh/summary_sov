import html
import re

from database.db import get_conn
from llm.groq_client import send_prompt
from mention_groups import MENTION_GROUPS

IGNORED_USERNAME = "sglypa_tg_bot"


def get_chat_ids():
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT user_id FROM messages")
        rows = cursor.fetchall()
    return [row[0] for row in rows]


def get_last_summary_msg_id(cursor, chat_id):
    cursor.execute(
        "SELECT last_summary_msg_id FROM chat_state WHERE chat_id = %s", (chat_id,)
    )
    row = cursor.fetchone()
    return row[0] if row else 0


def set_last_summary_msg_id(cursor, chat_id, last_id):
    cursor.execute(
        "INSERT INTO chat_state (chat_id, last_summary_msg_id) VALUES (%s, %s) "
        "ON CONFLICT (chat_id) DO UPDATE SET last_summary_msg_id = EXCLUDED.last_summary_msg_id",
        (chat_id, last_id),
    )


def build_message_link(chat_id, message_id):
    if not message_id:
        return None
    chat_id_str = str(chat_id)
    # t.me/c/<id>/<msg> links only work for supergroups/channels, whose
    # chat_id is always -100xxxxxxxxxxx. Regular (non-super) groups have no
    # working message permalink at all, so don't fabricate a dead URL.
    if not chat_id_str.startswith('-100'):
        return None
    internal_id = chat_id_str[4:]
    return f"https://t.me/c/{internal_id}/{message_id}"


def format_summary_html(raw_text, legend):
    escaped = html.escape(raw_text)

    def replace_citation(match):
        n = int(match.group(1))
        url = legend.get(n)
        return f'<a href="{url}">[{n}]</a> ' if url else ''

    text = re.sub(r'\[(\d+)\]', replace_citation, escaped)
    return '\n'.join(line.rstrip() for line in text.split('\n'))


def generate_and_send_summary(bot, chat_id, requested_n=None, requested_m=18):
    with get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id FROM messages WHERE user_id = %s ORDER BY id DESC LIMIT 1", (chat_id,)
        )
        last_row = cursor.fetchone()
        if last_row is None:
            bot.send_message(chat_id, "У вас нет сообщений для суммаризации.")
            return

        if requested_n is not None:
            N = requested_n
        else:
            last_summary_id = get_last_summary_msg_id(cursor, chat_id)
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE user_id = %s AND id > %s",
                (chat_id, last_summary_id),
            )
            N = cursor.fetchone()[0]

        if N <= 10:
            bot.send_message(chat_id, f"Сообщений было написано слишком мало для суммаризации: {N}")
            return

        cursor.execute(
            """
            SELECT id, message_id, user_name, message, replied_message
            FROM messages
            WHERE user_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (chat_id, N),
        )
        rows = cursor.fetchall()[::-1]

        if not rows:
            bot.send_message(chat_id, "Нет сообщений для суммаризации")
            return

        legend = {}
        lines = []
        for idx, (_, msg_id, user_name, text, replied) in enumerate(rows, start=1):
            legend[idx] = build_message_link(chat_id, msg_id)
            entry = f"[{idx}] {user_name}: {text}"
            if replied and replied != "Отмеченного сообщения нет":
                entry += f" (ответ на: {replied})"
            lines.append(entry)

        prompt_body = "\n".join(lines)
        newest_included_id = rows[-1][0]

        res = send_prompt(prompt_body, max_lines=requested_m)
        if not res:
            bot.send_message(chat_id, "LLM решил послать вас с ответом")
            return

        set_last_summary_msg_id(cursor, chat_id, newest_included_id)
        conn.commit()

    formatted = format_summary_html(res, legend)
    bot.send_message(chat_id, f'#summary\n\n{formatted}', parse_mode='HTML')


def load_handlers(bot):

    @bot.message_handler(func=lambda mess: mess.text and not mess.text.startswith("/"))
    def save_messages(message):
        try:
            if message.from_user.username == IGNORED_USERNAME:
                return
            print(f"Получено сообщение: {message.text}")
            reply_message = message.reply_to_message
            replied_text = reply_message.text if reply_message else "Отмеченного сообщения нет"
            user_name = message.from_user.first_name
            with get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO messages (user_id, user_name, message, replied_message, message_id) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (message.chat.id, user_name, message.text, replied_text, message.message_id),
                )
                conn.commit()
        except Exception as e:
            print(f"Ошибка при сохранении сообщения: {e}")

    @bot.message_handler(commands=['help'])
    def help_command(message):
        groups_list = ", ".join(f"/{name}" for name in MENTION_GROUPS)
        help_text = f"""
        Доступные команды:

        /summary [количество] [строк] - Создать краткое содержание последних сообщений
        Пример: /summary 50 - создаст краткое содержание последних 50 сообщений
        По умолчанию: все сообщения с последнего вызова /summary

        {groups_list} - позвать всех из соответствующей группы

        /help - Показать это сообщение

        Бот автоматически сохраняет все ваши текстовые сообщения для последующего создания краткого содержания
        и присылает саммари каждый день в 14:00 и 22:00, если сообщений было больше 10.
        """
        bot.send_message(message.chat.id, help_text.strip())

    @bot.message_handler(commands=list(MENTION_GROUPS.keys()))
    def mention_group(message):
        group = message.text.split()[0][1:].split('@')[0]
        usernames = MENTION_GROUPS.get(group)
        if not usernames:
            return
        bot.send_message(message.chat.id, f"🔔 {group}: {' '.join(usernames)}")

    @bot.message_handler(commands=['summary'])
    def summary(message):
        dt = message.text.split()

        requested_n = None
        if len(dt) > 1 and dt[1].isdigit():
            requested_n = int(dt[1])
            print(f"Пользователь запросил суммаризацию последних {requested_n} сообщений")

        requested_m = 18
        if len(dt) > 2 and dt[2].isdigit():
            requested_m = int(dt[2])
            print(f"Пользователь запросил суммаризацию в размере {requested_m} строк")

        generate_and_send_summary(bot, message.chat.id, requested_n, requested_m)
