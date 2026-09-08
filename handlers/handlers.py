import base64
import re
import time
from concurrent.futures import ThreadPoolExecutor

from chat_context import add_note, list_notes, remove_note
from context_learning import learn_context
from crypto_utils import decrypt, encrypt
from database.db import get_conn
from display_names import resolve_display_name
from embeddings import embed, to_vector_literal
from llm.graphs import _format_citations, run_ask_graph_merged, run_summary_graph
from llm.groq_client import caption_image
from mention_groups import (
    add_to_group,
    delete_group,
    get_group,
    list_groups,
    remove_from_group,
)
from voice_transcription import transcribe

IGNORED_USERNAME = "sglypa_tg_bot"
MAX_SUMMARY_MESSAGES = 500
MAX_SUMMARY_LINES = 25
_background_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="summary-bot")


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
        "last_summary_msg_id = GREATEST(chat_state.last_summary_msg_id, EXCLUDED.last_summary_msg_id), "
        "last_summary_text = EXCLUDED.last_summary_text",
        (chat_id, last_id, summary_text),
    )


def strip_citations(text):
    return re.sub(r'\s*\[\d+\]', '', text)


def build_message_link(chat_id, message_id, thread_id=None):
    if not message_id:
        return None
    chat_id_str = str(chat_id)
    # t.me/c/<id>/<msg> links only work for supergroups/channels, whose
    # chat_id is always -100xxxxxxxxxxx. Regular (non-super) groups have no
    # working message permalink at all, so don't fabricate a dead URL.
    if not chat_id_str.startswith('-100'):
        return None
    internal_id = chat_id_str[4:]
    if thread_id:
        return f"https://t.me/c/{internal_id}/{thread_id}/{message_id}"
    return f"https://t.me/c/{internal_id}/{message_id}"


def format_summary_html(raw_text, legend):
    return _format_citations(raw_text, legend)


def generate_and_send_summary(bot, chat_id, requested_n=None, requested_m=18, thread_id=None):
    # One summary per chat at a time, including across multiple app replicas.
    # A session advisory lock is released explicitly before the pooled
    # connection is returned.
    with get_conn() as lock_conn:
        lock_cursor = lock_conn.cursor()
        lock_cursor.execute("SELECT pg_advisory_lock(%s)", (chat_id,))
        try:
            return _generate_and_send_summary(bot, chat_id, requested_n, requested_m, thread_id)
        finally:
            lock_cursor.execute("SELECT pg_advisory_unlock(%s)", (chat_id,))


def _generate_and_send_summary(bot, chat_id, requested_n=None, requested_m=18, thread_id=None):
    requested_m = max(1, min(requested_m, MAX_SUMMARY_LINES))
    with get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id FROM messages WHERE user_id = %s AND is_bot = FALSE ORDER BY id DESC LIMIT 1", (chat_id,)
        )
        last_row = cursor.fetchone()
        if last_row is None:
            bot.send_message(chat_id, "У вас нет сообщений для суммаризации.", message_thread_id=thread_id)
            return

        last_summary_id, last_summary_text = get_chat_state(cursor, chat_id)

        if requested_n is not None:
            N = min(requested_n, MAX_SUMMARY_MESSAGES)
        else:
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE user_id = %s AND id > %s AND is_bot = FALSE",
                (chat_id, last_summary_id),
            )
            N = min(cursor.fetchone()[0], MAX_SUMMARY_MESSAGES)

        if N <= 10:
            bot.send_message(
                chat_id, f"Сообщений было написано слишком мало для суммаризации: {N}", message_thread_id=thread_id
            )
            return

        if requested_n is None:
            # Process a large backlog oldest-first in bounded chunks, so
            # advancing the cursor never skips older unseen messages.
            cursor.execute(
                """
                SELECT id, message_id, message_thread_id, user_name, username, message, replied_message
                FROM messages
                WHERE user_id = %s AND id > %s AND is_bot = FALSE
                ORDER BY id ASC
                LIMIT %s
                """,
                (chat_id, last_summary_id, N),
            )
            rows = cursor.fetchall()
        else:
            cursor.execute(
                """
                SELECT id, message_id, message_thread_id, user_name, username, message, replied_message
                FROM messages
                WHERE user_id = %s AND is_bot = FALSE
                ORDER BY id DESC
                LIMIT %s
                """,
                (chat_id, N),
            )
            rows = cursor.fetchall()[::-1]

        if not rows:
            bot.send_message(chat_id, "Нет сообщений для суммаризации", message_thread_id=thread_id)
            return

        legend = {}
        lines = []
        for row_id, msg_id, msg_thread_id, user_name, username, text, replied in rows:
            # Keyed by the REAL Telegram message_id, not a synthetic local
            # index -- _format_citations only needs legend/citation numbers
            # to match, so this works as a citation key exactly like the old
            # 1..N scheme did, but also lets memory extraction (see
            # memory_facts.py / prompt_for_context_extraction) cite real,
            # externally-checkable source_message_ids instead of a number
            # that's meaningless outside this one summary batch. Rows
            # without a message_id (rare legacy edge case) get the internal
            # row id instead, purely so the citation number is never a
            # broken "[None]" -- they just won't resolve to a legend link.
            cite_id = msg_id if msg_id is not None else row_id
            legend[cite_id] = build_message_link(chat_id, msg_id, msg_thread_id)
            author = resolve_display_name(username, user_name)
            entry = f"[{cite_id}] {author}: {decrypt(text)}"
            replied_plain = decrypt(replied)
            if replied_plain and replied_plain != "Отмеченного сообщения нет":
                entry += f" (ответ на: {replied_plain})"
            lines.append(entry)

        prompt_body = ""
        if last_summary_text:
            prompt_body += f"Предыдущее саммари (контекст, не пересказывай его тезисы заново):\n{last_summary_text}\n\n"
        prompt_body += "Новые сообщения:\n" + "\n".join(lines)

        newest_included_id = rows[-1][0]

    def save_state(last_id, summary_text):
        with get_conn() as state_conn:
            save_chat_state(state_conn.cursor(), chat_id, last_id, summary_text)
            state_conn.commit()

    run_summary_graph({
        "chat_id": chat_id,
        "thread_id": thread_id,
        "bot": bot,
        "prompt_body": prompt_body,
        "lines": lines,
        "legend": legend,
        "max_lines": requested_m,
        "newest_included_id": newest_included_id,
        "save_summary_state": save_state,
    })


_CONVERSATION_GAP_SECONDS = 300  # how long a lull can be before a new
                                  # message starts a fresh conversation
                                  # instead of continuing the last one


def _compute_conversation_id(cursor, chat_id, message_id, reply_to_message_id, message_date):
    # Persisted alternative to the fixed +-3 message_id window used for
    # anchor context in llm/graphs.py: a message inherits its reply
    # target's conversation, else continues the chat's last conversation if
    # it follows closely enough in time, else starts a new one (its own
    # message_id). None (unresolved -- no message_date, e.g. pre-migration
    # rows) means the anchor-window query falls back to the old +-3 logic.
    if reply_to_message_id:
        cursor.execute(
            "SELECT conversation_id, message_id FROM messages WHERE user_id = %s AND message_id = %s",
            (chat_id, reply_to_message_id),
        )
        row = cursor.fetchone()
        if row:
            return row[0] if row[0] is not None else row[1]

    if message_date is not None:
        cursor.execute(
            "SELECT conversation_id, message_id, message_date FROM messages "
            "WHERE user_id = %s ORDER BY id DESC LIMIT 1",
            (chat_id,),
        )
        row = cursor.fetchone()
        if row and row[2] is not None and 0 <= message_date - row[2] < _CONVERSATION_GAP_SECONDS:
            return row[0] if row[0] is not None else row[1]
        return message_id

    return None


def _update_message_embedding(row_id, text):
    try:
        with get_conn() as conn:
            conn.cursor().execute(
                "UPDATE messages SET embedding = %s::vector WHERE id = %s",
                (to_vector_literal(embed(text)), row_id),
            )
            conn.commit()
    except Exception as e:
        print(f"Ошибка при построении embedding для сообщения {row_id}: {e}")


def _save_incoming_message(
    chat_id, user_name, username, text, replied_text, message_id,
    message_thread_id=None, reply_to_message_id=None, message_date=None,
    enqueue_embedding=True,
):
    # Shared by save_messages (typed text) and the voice/photo/sticker
    # handlers (transcribed/captioned text standing in for the original
    # media) -- all of them end up as a plain row here either way.
    try:
        conflict_action = "DO NOTHING" if not enqueue_embedding else """
            DO UPDATE SET
                user_name = EXCLUDED.user_name,
                username = EXCLUDED.username,
                message = EXCLUDED.message,
                replied_message = EXCLUDED.replied_message,
                message_thread_id = EXCLUDED.message_thread_id,
                reply_to_message_id = EXCLUDED.reply_to_message_id,
                message_date = EXCLUDED.message_date,
                search_vector = EXCLUDED.search_vector
        """
        with get_conn() as conn:
            cursor = conn.cursor()
            conversation_id = _compute_conversation_id(cursor, chat_id, message_id, reply_to_message_id, message_date)
            cursor.execute(
                f"""
                INSERT INTO messages (
                    user_id, user_name, username, message, replied_message,
                    message_id, message_thread_id, reply_to_message_id,
                    message_date, search_vector, conversation_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('russian', %s), %s)
                ON CONFLICT (user_id, message_id) WHERE message_id IS NOT NULL
                {conflict_action}
                RETURNING id
                """,
                (
                    chat_id, user_name, username,
                    encrypt(text), encrypt(replied_text), message_id,
                    message_thread_id, reply_to_message_id, message_date, text,
                    conversation_id,
                ),
            )
            row = cursor.fetchone()
            conn.commit()
        if enqueue_embedding and row:
            _background_pool.submit(_update_message_embedding, row[0], text)
    except Exception as e:
        print(f"Ошибка при сохранении сообщения: {e}")


def _save_bot_answer(
    chat_id, sent_message_id, bot_username, plain_text,
    message_thread_id=None, message_date=None, reply_to_message_id=None,
):
    # Outgoing bot messages never pass through save_messages (that only
    # fires on incoming Telegram updates), so without this a reply to the
    # bot's own answer -- or an implicit follow-up like "это правда?" --
    # has nothing in `messages` to anchor on. is_bot=TRUE keeps it out of
    # /summary and context_learning.py, which only care about actual
    # participants.
    try:
        with get_conn() as conn:
            cursor = conn.cursor()
            conversation_id = _compute_conversation_id(
                cursor, chat_id, sent_message_id, reply_to_message_id, message_date
            )
            cursor.execute(
                """
                INSERT INTO messages (
                    user_id, user_name, username, message, replied_message,
                    message_id, message_thread_id, message_date, search_vector, is_bot, conversation_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('russian', %s), TRUE, %s)
                ON CONFLICT (user_id, message_id) WHERE message_id IS NOT NULL
                DO UPDATE SET message = EXCLUDED.message, search_vector = EXCLUDED.search_vector
                RETURNING id
                """,
                (
                    chat_id, "Бот", bot_username, encrypt(plain_text), None, sent_message_id,
                    message_thread_id, message_date, plain_text, conversation_id,
                ),
            )
            row_id = cursor.fetchone()[0]
            conn.commit()
        _background_pool.submit(_update_message_embedding, row_id, plain_text)
    except Exception as e:
        print(f"Ошибка при сохранении ответа бота: {e}")


def answer_chat_question(
    bot, chat_id, question, replied_message_id=None, bot_username=None,
    asker_name="неизвестный", thread_id=None,
):
    run_ask_graph_merged({
        "bot": bot,
        "chat_id": chat_id,
        "question": question,
        "asker_name": asker_name,
        "replied_message_id": replied_message_id,
        "bot_username": bot_username,
        "thread_id": thread_id,
        "save_bot_answer": _save_bot_answer,
    })


def load_handlers(bot):
    bot_username = bot.get_me().username
    mention_tag = f"@{bot_username}".lower() if bot_username else None

    def _ask_if_mentioned(message, text):
        # Mentioning the bot anywhere in a regular message is treated as a
        # question, same as /ask, without needing the explicit command.
        if mention_tag and mention_tag in text.lower():
            question = re.sub(re.escape(mention_tag), '', text, flags=re.IGNORECASE).strip()
            if question:
                replied_message_id = message.reply_to_message.message_id if message.reply_to_message else None
                asker_name = resolve_display_name(message.from_user.username, message.from_user.first_name)
                answer_chat_question(
                    bot, message.chat.id, question, replied_message_id, bot_username,
                    asker_name, getattr(message, 'message_thread_id', None),
                )

    @bot.message_handler(func=lambda mess: mess.text and not mess.text.startswith("/"))
    def save_messages(message):
        if message.from_user.username == IGNORED_USERNAME:
            return

        print(f"Получено сообщение {message.message_id}")
        reply_message = message.reply_to_message
        replied_text = reply_message.text if reply_message else "Отмеченного сообщения нет"
        _save_incoming_message(
            message.chat.id, message.from_user.first_name, message.from_user.username,
            message.text, replied_text, message.message_id,
            getattr(message, 'message_thread_id', None),
            reply_message.message_id if reply_message else None,
            message.date,
        )
        _ask_if_mentioned(message, message.text)

    def _download_bytes(file_id, attempts=3):
        # Transient connect timeouts to api.telegram.org happen occasionally
        # from cloud hosts -- without a retry, a single blip permanently
        # loses that voice/photo/sticker instead of just being slow once.
        last_error = None
        for attempt in range(attempts):
            try:
                file_info = bot.get_file(file_id)
                return bot.download_file(file_info.file_path)
            except Exception as e:
                last_error = e
                print(f"Попытка {attempt + 1}/{attempts} скачать файл не удалась: {e}")
                if attempt < attempts - 1:
                    time.sleep(2)
        raise last_error

    @bot.message_handler(content_types=['voice', 'video_note'])
    def save_voice_message(message):
        if message.from_user.username == IGNORED_USERNAME:
            return

        reply_message = message.reply_to_message
        replied_text = reply_message.text if reply_message else "Отмеченного сообщения нет"
        media_label = "кружок" if message.video_note else "голосовое"
        placeholder = message.caption or f"[{media_label}]"
        _save_incoming_message(
            message.chat.id, message.from_user.first_name, message.from_user.username,
            placeholder, replied_text, message.message_id,
            getattr(message, 'message_thread_id', None),
            reply_message.message_id if reply_message else None,
            message.date, enqueue_embedding=False,
        )

        def run():
            try:
                media = message.voice or message.video_note
                audio_bytes = _download_bytes(media.file_id)
                transcript = transcribe(audio_bytes)
                if not transcript:
                    return
                print(f"Расшифровано голосовое {message.message_id}")
                # Voice messages (not video notes) can carry their own typed
                # caption alongside the audio -- keep it, same as photos.
                text = f"{message.caption}\n{transcript}" if message.caption else transcript
                _save_incoming_message(
                    message.chat.id, message.from_user.first_name, message.from_user.username,
                    text, replied_text, message.message_id,
                    getattr(message, 'message_thread_id', None),
                    reply_message.message_id if reply_message else None,
                    message.date,
                )
                _ask_if_mentioned(message, text)
            except Exception as e:
                print(f"Ошибка при обработке голосового/кружка: {e}")

        _background_pool.submit(run)

    @bot.message_handler(content_types=['photo', 'sticker'])
    def save_visual_message(message):
        if message.from_user.username == IGNORED_USERNAME:
            return

        reply_message = message.reply_to_message
        replied_text = reply_message.text if reply_message else "Отмеченного сообщения нет"
        tag = "изображение" if message.photo else "стикер"
        placeholder = f"{message.caption}\n[{tag}]" if message.caption else f"[{tag}]"
        _save_incoming_message(
            message.chat.id, message.from_user.first_name, message.from_user.username,
            placeholder, replied_text, message.message_id,
            getattr(message, 'message_thread_id', None),
            reply_message.message_id if reply_message else None,
            message.date, enqueue_embedding=False,
        )

        def run():
            try:
                if message.photo:
                    file_id = message.photo[-1].file_id
                    tag = "изображение"
                else:
                    # Animated/video stickers (TGS/WEBM) aren't a plain
                    # decodable image -- their thumbnail is, and every
                    # sticker has one, static or not, so use it uniformly.
                    sticker = message.sticker
                    file_id = sticker.thumbnail.file_id if sticker.thumbnail else sticker.file_id
                    tag = "стикер"

                image_bytes = _download_bytes(file_id)
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                image_description = caption_image(image_b64)
                if not image_description:
                    return
                print(f"Описано {tag} в сообщении {message.message_id}")
                # message.caption is the human's own typed text alongside
                # the photo/sticker (Telegram keeps it separate from
                # message.text) -- without it, whatever they actually said
                # is silently dropped and only the AI-generated description
                # gets saved.
                text = f"[{tag}: {image_description.strip()}]"
                if message.caption:
                    text = f"{message.caption}\n{text}"
                _save_incoming_message(
                    message.chat.id, message.from_user.first_name, message.from_user.username,
                    text, replied_text, message.message_id,
                    getattr(message, 'message_thread_id', None),
                    reply_message.message_id if reply_message else None,
                    message.date,
                )
                if message.caption:
                    _ask_if_mentioned(message, message.caption)
            except Exception as e:
                print(f"Ошибка при обработке {'фото' if message.photo else 'стикера'}: {e}")

        _background_pool.submit(run)

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

        generate_and_send_summary(bot, message.chat.id, requested_n, requested_m, message.message_thread_id)

    @bot.message_handler(commands=['ask'])
    def ask_cmd(message):
        dt = message.text.split(maxsplit=1)
        if len(dt) < 2:
            bot.send_message(message.chat.id, "Формат: /ask <вопрос>")
            return

        question = dt[1]
        replied_message_id = message.reply_to_message.message_id if message.reply_to_message else None
        asker_name = resolve_display_name(message.from_user.username, message.from_user.first_name)
        answer_chat_question(
            bot, message.chat.id, question, replied_message_id, bot_username,
            asker_name, getattr(message, 'message_thread_id', None),
        )

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

        _background_pool.submit(run)
