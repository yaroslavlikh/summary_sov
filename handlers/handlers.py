import html
import re
import threading

from chat_context import add_note, get_context_block, list_notes, remove_note
from chat_moments import search_moments
from context_learning import learn_context
from crypto_utils import decrypt, encrypt
from database.db import get_conn
from display_names import resolve_display_name
from embeddings import embed, to_vector_literal
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


def build_group_context_prompt_section(chat_id):
    notes = get_context_block(chat_id)
    if not notes:
        return ""
    return (
        "\nКонтекст о группе (не сами сообщения переписки, а накопленные заметки "
        "про участников, их характерные черты, повторяющиеся шутки/темы — используй "
        "только для лучшего понимания тона и отсылок, не пересказывай как содержание):\n"
        f"{notes}\n"
    )


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

    # The model cites messages by their raw index in the input batch (which
    # can jump around, e.g. [16][19][42]), since most input messages don't
    # end up cited at all. Renumber whatever actually appears, in order of
    # first appearance, to a clean 1, 2, 3... sequence for the reader.
    seen_order = []
    for match in re.finditer(r'\[(\d+)\]', escaped):
        n = int(match.group(1))
        if n in legend and n not in seen_order:
            seen_order.append(n)
    remap = {old: new for new, old in enumerate(seen_order, start=1)}

    def replace_citation(match):
        n = int(match.group(1))
        url = legend.get(n)
        if not url:
            return ''
        return f'<a href="{url}">[{remap[n]}]</a> '

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
            entry = f"[{idx}] {author}: {decrypt(text)}"
            replied_plain = decrypt(replied)
            if replied_plain and replied_plain != "Отмеченного сообщения нет":
                entry += f" (ответ на: {replied_plain})"
            lines.append(entry)

        prompt_body = ""
        if last_summary_text:
            prompt_body += f"Предыдущее саммари (контекст, не пересказывай его тезисы заново):\n{last_summary_text}\n\n"
        prompt_body += "Новые сообщения:\n" + "\n".join(lines)

        newest_included_id = rows[-1][0]

        group_context = build_group_context_prompt_section(chat_id)
        res = send_prompt(prompt_body, max_lines=requested_m, group_context=group_context)
        if not res:
            bot.send_message(chat_id, "LLM решил послать вас с ответом")
            return

        save_chat_state(cursor, chat_id, newest_included_id, strip_citations(res))
        conn.commit()

    formatted = format_summary_html(res, legend)
    bot.send_message(chat_id, f'#summary\n\n{formatted}', parse_mode='HTML')


# plainto_tsquery ANDs every word together, which fails as soon as the
# question includes meta-words ("менялся", "разговор") that describe the
# question itself rather than vocabulary the chat actually used. Build an
# OR-query instead so a message matching any one concept word can surface.
LATIN_CYRILLIC_JARGON = {
    "rag": "раг",
    "раг": "rag",
    "sql": "скл",
}


def _build_or_query(cursor, question):
    cursor.execute("SELECT plainto_tsquery('russian', %s)::text", (question,))
    row = cursor.fetchone()
    tsquery_text = row[0] if row else None
    if not tsquery_text:
        return None

    lexemes = set(re.findall(r"'([^']+)'", tsquery_text))
    for lex in list(lexemes):
        alias = LATIN_CYRILLIC_JARGON.get(lex.lower())
        if alias:
            lexemes.add(alias)

    return " | ".join(f"'{lex}'" for lex in lexemes) or None


def answer_chat_question(bot, chat_id, question, replied_message_id=None):
    with get_conn() as conn:
        cursor = conn.cursor()

        match_ids = set()
        anchor_id = None

        # A short/deictic question ("это правда?", "серьёзно?") carries no
        # useful search keywords on its own — it refers to whatever was just
        # replied to. Anchor on that directly instead of relying on search
        # to guess, and remember it so we can flag it for the model below.
        if replied_message_id:
            cursor.execute(
                "SELECT id FROM messages WHERE user_id = %s AND message_id = %s",
                (chat_id, replied_message_id),
            )
            row = cursor.fetchone()
            if row:
                anchor_id = row[0]
                match_ids.add(anchor_id)

        # Only fall back to "recent messages" when there's no explicit reply
        # anchor — otherwise it just floods the context with noise that can
        # outweigh the one message the user actually pointed at.
        if not anchor_id:
            cursor.execute(
                "SELECT id FROM messages WHERE user_id = %s ORDER BY id DESC LIMIT 10",
                (chat_id,),
            )
            match_ids.update(row[0] for row in cursor.fetchall())

        # Full-text: match each message's own precomputed search_vector.
        # (Previously this matched a live-concatenated +-3 chunk of raw
        # message text, but message is now stored encrypted, so tsvector
        # can no longer be computed on the fly from it -- only the
        # precomputed per-message search_vector, built from plaintext at
        # insert time, is available. The post-match +-3 window expansion
        # below still pulls in neighboring context either way.)
        or_query = _build_or_query(cursor, question)
        if or_query:
            cursor.execute(
                """
                SELECT id FROM (
                    SELECT id, ts_rank(search_vector, to_tsquery('russian', %s)) AS rank
                    FROM messages
                    WHERE user_id = %s AND search_vector @@ to_tsquery('russian', %s)
                    ORDER BY rank DESC
                    LIMIT 4
                ) top_matches
                """,
                (or_query, chat_id, or_query),
            )
            match_ids.update(row[0] for row in cursor.fetchall())

        # Semantic: catches paraphrased questions that share no vocabulary
        # with the messages that actually answer them.
        question_embedding = embed(question)
        cursor.execute(
            """
            SELECT id FROM messages
            WHERE user_id = %s AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 4
            """,
            (chat_id, to_vector_literal(question_embedding)),
        )
        match_ids.update(row[0] for row in cursor.fetchall())

        if not match_ids:
            bot.send_message(chat_id, "В истории чата не нашёл ничего похожего на этот вопрос.")
            return

        # Expand the same +-3 window for display, so the model sees the
        # actual per-message text (with correct citation numbers), not the
        # concatenated blob used just for matching.
        window_ids = set()
        for match_id in match_ids:
            window_ids.update(range(match_id - 3, match_id + 4))

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
    for idx, (row_id, msg_id, user_name, username, text) in enumerate(rows, start=1):
        legend[idx] = build_message_link(chat_id, msg_id)
        author = resolve_display_name(username, user_name)
        tag = " [СООБЩЕНИЕ, НА КОТОРОЕ ОТВЕЧАЛИ]" if row_id == anchor_id else ""
        lines.append(f"[{idx}] {author}: {decrypt(text)}{tag}")

    group_context = build_group_context_prompt_section(chat_id)
    relevant_moments = search_moments(chat_id, question_embedding, top_k=5)
    if relevant_moments:
        group_context += (
            "\nВозможно релевантные моменты из истории (мнения/настроения/шутки, "
            "не обязательно точный источник ответа, просто дополнительный фон):\n"
            + "\n".join(f"- {m}" for m in relevant_moments) + "\n"
        )

    full_prompt = prompt_for_qa.format(question=question, messages="\n".join(lines), group_context=group_context)
    res = answer_question(full_prompt)
    if not res:
        bot.send_message(chat_id, "LLM решил послать вас с ответом")
        return

    formatted = format_summary_html(res, legend)
    bot.send_message(chat_id, formatted, parse_mode='HTML')


def load_handlers(bot):
    bot_username = bot.get_me().username
    mention_tag = f"@{bot_username}".lower() if bot_username else None

    @bot.message_handler(func=lambda mess: mess.text and not mess.text.startswith("/"))
    def save_messages(message):
        if message.from_user.username == IGNORED_USERNAME:
            return

        try:
            print(f"Получено сообщение: {message.text}")
            reply_message = message.reply_to_message
            replied_text = reply_message.text if reply_message else "Отмеченного сообщения нет"
            user_name = message.from_user.first_name
            embedding_literal = to_vector_literal(embed(message.text))
            # search_vector must be computed from the plaintext explicitly
            # now -- message stores encrypted ciphertext, so Postgres can no
            # longer derive it automatically the way a GENERATED column did.
            with get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO messages (user_id, user_name, username, message, replied_message, message_id, embedding, search_vector) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, to_tsvector('russian', %s))",
                    (
                        message.chat.id, user_name, message.from_user.username,
                        encrypt(message.text), encrypt(replied_text), message.message_id,
                        embedding_literal, message.text,
                    ),
                )
                conn.commit()
        except Exception as e:
            print(f"Ошибка при сохранении сообщения: {e}")

        # Mentioning the bot anywhere in a regular message is treated as a
        # question, same as /ask, without needing the explicit command.
        if mention_tag and mention_tag in message.text.lower():
            question = re.sub(re.escape(mention_tag), '', message.text, flags=re.IGNORECASE).strip()
            if question:
                replied_message_id = message.reply_to_message.message_id if message.reply_to_message else None
                answer_chat_question(bot, message.chat.id, question, replied_message_id)

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

        /addcontext <заметка> - добавить заметку о группе вручную
        /context - показать все заметки
        /removecontext <id> - удалить заметку
        /learncontext - автоматически собрать портреты людей и повторяющиеся паттерны по всей истории чата (может занять время)

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
        replied_message_id = message.reply_to_message.message_id if message.reply_to_message else None
        answer_chat_question(bot, message.chat.id, question, replied_message_id)

    @bot.message_handler(commands=['addcontext'])
    def add_context_cmd(message):
        dt = message.text.split(maxsplit=1)
        if len(dt) < 2:
            bot.send_message(message.chat.id, "Формат: /addcontext <заметка>")
            return

        note_id = add_note(message.chat.id, dt[1])
        bot.send_message(message.chat.id, f'Добавлено (#{note_id}): {dt[1]}')

    @bot.message_handler(commands=['context'])
    def list_context_cmd(message):
        notes = list_notes(message.chat.id)
        if not notes:
            bot.send_message(message.chat.id, "Заметок пока нет")
            return
        bot.send_message(message.chat.id, "\n".join(f"#{note_id}: {note}" for note_id, note in notes))

    @bot.message_handler(commands=['removecontext'])
    def remove_context_cmd(message):
        dt = message.text.split()
        if len(dt) < 2 or not dt[1].isdigit():
            bot.send_message(message.chat.id, "Формат: /removecontext <id>")
            return

        if remove_note(message.chat.id, int(dt[1])):
            bot.send_message(message.chat.id, f'Заметка #{dt[1]} удалена')
        else:
            bot.send_message(message.chat.id, f'Нет заметки #{dt[1]}')

    @bot.message_handler(commands=['learncontext'])
    def learn_context_cmd(message):
        chat_id = message.chat.id
        bot.send_message(chat_id, "Начал сбор контекста по всей истории чата, это может занять время...")

        def run():
            try:
                portraits, patterns = learn_context(chat_id)
                blocks = []
                if portraits:
                    blocks.append("Портреты:\n" + "\n".join(portraits))
                if patterns:
                    blocks.append("Повторяющиеся паттерны:\n" + "\n".join(f"- {p}" for p in patterns))
                if not blocks:
                    bot.send_message(chat_id, "Не нашёл ничего значимого — возможно, истории пока маловато.")
                    return
                bot.send_message(chat_id, "Контекст обновлён:\n\n" + "\n\n".join(blocks))
            except Exception as e:
                print(f"Ошибка при сборе контекста: {e}")
                bot.send_message(chat_id, "Что-то пошло не так при сборе контекста, гляну логи.")

        threading.Thread(target=run, daemon=True).start()
