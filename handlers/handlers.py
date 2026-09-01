import html
import re

from database.db import get_conn
from display_names import resolve_display_name
from llm.groq_client import answer_question, send_prompt
from llm.prompt import prompt_for_qa
from mention_groups import (
    add_to_group,
    delete_group,
    get_group,
    list_groups,
    remove_from_group,
)

IGNORED_USERNAME = "sglypa_tg_bot"


def get_chat_ids():
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT user_id FROM messages")
        rows = cursor.fetchall()
    return [row[0] for row in rows]


def get_chat_state(cursor, chat_id):
    cursor.execute(
        "SELECT last_summary_msg_id, last_summary_text FROM chat_state WHERE chat_id = %s", (chat_id,)
    )
    row = cursor.fetchone()
    return (row[0], row[1]) if row else (0, None)


def save_chat_state(cursor, chat_id, last_id, summary_text):
    cursor.execute(
        "INSERT INTO chat_state (chat_id, last_summary_msg_id, last_summary_text) VALUES (%s, %s, %s) "
        "ON CONFLICT (chat_id) DO UPDATE SET "
        "last_summary_msg_id = EXCLUDED.last_summary_msg_id, "
        "last_summary_text = EXCLUDED.last_summary_text",
        (chat_id, last_id, summary_text),
    )


def strip_citations(text):
    return re.sub(r'\s*\[\d+\]', '', text)


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

        last_summary_id, last_summary_text = get_chat_state(cursor, chat_id)

        if requested_n is not None:
            N = requested_n
        else:
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
            SELECT id, message_id, user_name, username, message, replied_message
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
        for idx, (_, msg_id, user_name, username, text, replied) in enumerate(rows, start=1):
            legend[idx] = build_message_link(chat_id, msg_id)
            author = resolve_display_name(username, user_name)
            entry = f"[{idx}] {author}: {text}"
            if replied and replied != "Отмеченного сообщения нет":
                entry += f" (ответ на: {replied})"
            lines.append(entry)

        prompt_body = ""
        if last_summary_text:
            prompt_body += f"Предыдущее саммари (контекст, не пересказывай его тезисы заново):\n{last_summary_text}\n\n"
        prompt_body += "Новые сообщения:\n" + "\n".join(lines)

        newest_included_id = rows[-1][0]

        res = send_prompt(prompt_body, max_lines=requested_m)
        if not res:
            bot.send_message(chat_id, "LLM решил послать вас с ответом")
            return

        save_chat_state(cursor, chat_id, newest_included_id, strip_citations(res))
        conn.commit()

    formatted = format_summary_html(res, legend)
    bot.send_message(chat_id, f'#summary\n\n{formatted}', parse_mode='HTML')


def answer_chat_question(bot, chat_id, question):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id FROM (
                SELECT id, ts_rank(search_vector, plainto_tsquery('russian', %s)) AS rank
                FROM messages
                WHERE user_id = %s AND search_vector @@ plainto_tsquery('russian', %s)
                ORDER BY rank DESC
                LIMIT 8
            ) top_matches
            """,
            (question, chat_id, question),
        )
        match_ids = [row[0] for row in cursor.fetchall()]

        if not match_ids:
            bot.send_message(chat_id, "В истории чата не нашёл ничего похожего на этот вопрос.")
            return

        # A keyword match on the question rarely IS the answer — the answer is
        # usually in a nearby reply that doesn't repeat the question's words
        # at all. Pull in a small window of surrounding messages per match.
        window_ids = set()
        for match_id in match_ids:
            window_ids.update(range(match_id - 1, match_id + 5))

        cursor.execute(
            """
            SELECT id, message_id, user_name, username, message
            FROM messages
            WHERE user_id = %s AND id = ANY(%s)
            ORDER BY id ASC
            """,
            (chat_id, list(window_ids)),
        )
        rows = cursor.fetchall()

    if not rows:
        bot.send_message(chat_id, "В истории чата не нашёл ничего похожего на этот вопрос.")
        return

    legend = {}
    lines = []
    for idx, (_, msg_id, user_name, username, text) in enumerate(rows, start=1):
        legend[idx] = build_message_link(chat_id, msg_id)
        author = resolve_display_name(username, user_name)
        lines.append(f"[{idx}] {author}: {text}")

    full_prompt = prompt_for_qa.format(question=question, messages="\n".join(lines))
    res = answer_question(full_prompt)
    if not res:
        bot.send_message(chat_id, "LLM решил послать вас с ответом")
        return

    formatted = format_summary_html(res, legend)
    bot.send_message(chat_id, formatted, parse_mode='HTML')


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
                    "INSERT INTO messages (user_id, user_name, username, message, replied_message, message_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (message.chat.id, user_name, message.from_user.username, message.text, replied_text, message.message_id),
                )
                conn.commit()
        except Exception as e:
            print(f"Ошибка при сохранении сообщения: {e}")

    @bot.message_handler(commands=['help'])
    def help_command(message):
        help_text = """
        Доступные команды:

        /summary [количество] [строк] - Создать краткое содержание последних сообщений
        Пример: /summary 50 - создаст краткое содержание последних 50 сообщений
        По умолчанию: все сообщения с последнего вызова /summary

        /ask <вопрос> - найти ответ в истории чата
        Пример: /ask во сколько мы собирались на бильярд

        /ping <группа> - позвать всех из группы
        /groups - список групп в этом чате
        /creategroup <группа> @user1 @user2 ... - создать новую группу
        /addto <группа> @user1 @user2 ... - добавить в группу
        /removefrom <группа> @user1 @user2 ... - убрать из группы
        /deletegroup <группа> - удалить группу целиком

        /help - Показать это сообщение

        Бот автоматически сохраняет все ваши текстовые сообщения для последующего создания краткого содержания
        и присылает саммари каждый день в 14:00 и 22:00, если сообщений было больше 10.
        """
        bot.send_message(message.chat.id, help_text.strip())

    @bot.message_handler(commands=['ping'])
    def ping_group(message):
        dt = message.text.split()
        if len(dt) < 2:
            bot.send_message(message.chat.id, "Укажи группу: /ping <имя>")
            return

        name = dt[1]
        usernames = get_group(message.chat.id, name)
        if not usernames:
            bot.send_message(message.chat.id, f"Нет такой группы: {name}")
            return

        bot.send_message(message.chat.id, f"{name}: {' '.join(usernames)}")

    @bot.message_handler(commands=['groups'])
    def groups_list(message):
        groups = list_groups(message.chat.id)
        if not groups:
            bot.send_message(message.chat.id, "Групп пока нет")
            return
        bot.send_message(message.chat.id, ", ".join(f"{name} ({count})" for name, count in groups))

    @bot.message_handler(commands=['creategroup'])
    def create_group_cmd(message):
        dt = message.text.split()
        if len(dt) < 3:
            bot.send_message(message.chat.id, "Формат: /creategroup <группа> @user1 @user2 ...")
            return

        name, usernames = dt[1], dt[2:]
        if get_group(message.chat.id, name):
            bot.send_message(message.chat.id, f'Группа "{name}" уже существует, используй /addto')
            return

        add_to_group(message.chat.id, name, usernames)
        bot.send_message(message.chat.id, f'Группа "{name}" создана: {" ".join(get_group(message.chat.id, name))}')

    @bot.message_handler(commands=['addto'])
    def add_to_group_cmd(message):
        dt = message.text.split()
        if len(dt) < 3:
            bot.send_message(message.chat.id, "Формат: /addto <группа> @user1 @user2 ...")
            return

        name, usernames = dt[1], dt[2:]
        add_to_group(message.chat.id, name, usernames)
        bot.send_message(message.chat.id, f'Добавлено в "{name}": {" ".join(usernames)}')

    @bot.message_handler(commands=['removefrom'])
    def remove_from_group_cmd(message):
        dt = message.text.split()
        if len(dt) < 3:
            bot.send_message(message.chat.id, "Формат: /removefrom <группа> @user1 @user2 ...")
            return

        name, usernames = dt[1], dt[2:]
        remove_from_group(message.chat.id, name, usernames)
        bot.send_message(message.chat.id, f'Удалено из "{name}": {" ".join(usernames)}')

    @bot.message_handler(commands=['deletegroup'])
    def delete_group_cmd(message):
        dt = message.text.split()
        if len(dt) < 2:
            bot.send_message(message.chat.id, "Формат: /deletegroup <группа>")
            return

        name = dt[1]
        delete_group(message.chat.id, name)
        bot.send_message(message.chat.id, f'Группа "{name}" удалена')

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

    @bot.message_handler(commands=['ask'])
    def ask_cmd(message):
        dt = message.text.split(maxsplit=1)
        if len(dt) < 2:
            bot.send_message(message.chat.id, "Формат: /ask <вопрос>")
            return

        question = dt[1]
        answer_chat_question(bot, message.chat.id, question)
