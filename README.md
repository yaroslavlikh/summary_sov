# summary_sov

Telegram-бот для группового чата: саммаризация истории, ответы на вопросы
по контексту переписки (RAG), накопление знаний о людях и группе, и
исследовательский трек по provenance-bearing памяти для LLM-агентов.

Построен на LangGraph (оркестрация `/ask` и `/summary`), Postgres + pgvector
(гибридный full-text + семантический поиск), Groq (`gpt-oss-120b`/`20b`),
с шифрованием сообщений at rest и опциональной трассировкой через Langfuse.

## Возможности

- **Автосохранение** — все текстовые сообщения, голосовые (транскрибируются
  локально через `faster-whisper`), фото и стикеры (описываются vision-моделью)
  сохраняются в Postgres с `message_id` для последующих ссылок; текст
  шифруется (Fernet) at rest.
- **`/summary [N] [M]`** — саммари последних N сообщений (по умолчанию — все
  с прошлого вызова) в M тезисах (по умолчанию 18), разбито на тематические
  блоки с кликабельными ссылками на исходные сообщения. Автоматически дважды
  в день (14:00 и 22:00 по `TIMEZONE`), если новых сообщений больше 10.
- **`/ask <вопрос>`** — вопрос по истории чата: резолвится anchor (если это
  reply), гибридный FTS+vector retrieval, LLM rerank, ответ с цитатами на
  реальные сообщения. Бота можно не вызывать командой — простое упоминание
  `@bot_username` в любом сообщении тоже триггерит `/ask`.
- **Память о людях и группе** — `@bot запомни, что...`/`обращайся ко мне
  как...` распознаётся семантически (не по ключевым словам) в том же
  LLM-вызове, что уже классифицирует вопрос, без дополнительной задержки.
  `/addcontext`, `/context`, `/removecontext` — заметки вручную;
  `/learncontext` — автоматически строит портреты участников и находит
  повторяющиеся паттерны по всей истории.
- **Группы упоминаний** — `/creategroup`, `/addto`, `/removefrom`,
  `/deletegroup`, `/groups`, `/ping <группа>` — позвать сразу несколько
  человек.

Полный список команд — `/help` в самом боте.

## Установка

### Требования

- Python 3.10+
- Telegram Bot Token ([@BotFather](https://t.me/BotFather))
- Groq API Key ([console.groq.com/keys](https://console.groq.com/keys))
- Postgres с расширением `pgvector` (например, Railway)

### Шаги

```bash
pip install -r requirements.txt
```

Создайте `.env` в корне (см. `example.env`):

```env
BOT_TOKEN=токен_telegram_бота
GROQ_API_KEY=ключ_groq_api
TIMEZONE=Europe/Kyiv
DATABASE_URL=postgresql://user:password@host:port/dbname
MESSAGE_ENCRYPTION_KEY=ключ_Fernet
RAILWAY_PUBLIC_DOMAIN=your-service.up.railway.app

# опционально — включает трассировку LangGraph-узлов и LLM-вызовов
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

`MESSAGE_ENCRYPTION_KEY` — сгенерировать: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.

### Запуск

```bash
python main.py
```

Бот поднимает webhook-сервер (Flask + waitress) на `RAILWAY_PUBLIC_DOMAIN`,
инициализирует/самопочиняет схему БД и запускает планировщик авто-саммари.

## Структура проекта

```
summary_sov/
├── main.py                    # точка входа, webhook-сервер, init
├── config.py                  # переменные окружения
├── scheduler.py                # авто-саммари 14:00 / 22:00
├── handlers/handlers.py        # все message/command-обработчики
├── llm/
│   ├── graphs.py                # LangGraph-графы /ask и /summary
│   ├── groq_client.py           # ChatGroq-клиент, Langfuse tracing
│   └── prompt.py                # шаблоны промптов
├── database/
│   ├── db.py                    # пул соединений
│   └── init_db.py               # самопочинающаяся схема
├── memory_facts.py             # provenance-bearing память о состоянии людей
├── participants.py             # резолюция subject → реальный участник чата
├── chat_context.py             # заметки/портреты (/addcontext, /learncontext)
├── chat_moments.py             # разовые моменты, ретрив по вектору
├── context_learning.py         # построение портретов/moments из истории
├── mention_groups.py           # группы для /ping
├── crypto_utils.py             # шифрование сообщений at rest
├── display_names.py            # маппинг username → отображаемое имя
├── embeddings.py               # локальная sentence-transformers модель
├── voice_transcription.py      # faster-whisper (локально, CPU)
├── webhook_server.py           # Flask-приложение для Telegram webhook
├── tests/                      # unittest, реальный sandboxed Postgres
│   └── evals/                  # LLM-judge бенчмарк /ask на реальных данных
├── research/                   # исследовательский трек памяти (см. ниже)
└── docs/
    ├── CHANGELOG.md             # хронология каждого изменения и почему
    ├── eval_incidents.md        # сырой лог найденных инцидентов
    └── paper_material.md        # материал для статьи
```

## Исследовательский трек: prospective memory

Отдельная линия работы (`research/`, задокументирована в
`research/PROSPECTIVE_MEMORY_RETRIEVAL.md`) поверх основного бота: можно ли
надёжно извлекать и переиспользовать provenance-bearing факты о людях/группе
(`memory_facts`), и как честно это измерить, не обманывая себя на каждом шаге
измерения. Кратко:

- **Gold-датасет** — human-verified факты, независимые от production-экстракции
  (иначе оценка циклична), заморожен с sha256-checksum.
- **Forced-JSON extractor** — честная декомпозиция recall (не одно число).
- **Oracle-memory эксперимент** — paired baseline vs oracle `/ask` на
  memory-dependent вопросах.

Все находки, включая реальные баги, пойманные по пути (circular evaluation,
attribution collapse внутри собственного инструмента, LangGraph молча
роняющий поле стейта) — в `docs/CHANGELOG.md` и `docs/paper_material.md`.
Сырые датасеты и результаты экспериментов вне git (реальный контент чата) —
см. `research/` скрипты для воспроизведения.

## База данных

Схема создаётся и самопочиняется автоматически при старте (`init_db()`).
Основные таблицы: `messages` (шифрованный текст, `message_id` для ссылок,
`search_vector`/`embedding` для гибридного поиска, `conversation_id` для
группировки реплаев), `chat_state`, `chat_context`, `chat_moments`,
`memory_facts`, `mention_groups`, `scheduler_runs`. Пара
`(user_id, message_id)` уникальна — повторная доставка Telegram webhook не
создаёт дубликаты.

## Тесты

```bash
python -m unittest discover -s tests -v
```

unittest (не pytest) против реального sandboxed Postgres (изолированная
схема на каждый прогон, никогда не моки). LLM-judge бенчмарк `/ask` на
реальных исторических вызовах бота:

```bash
python3 tests/evals/run_eval.py
```

## Модели

- Основная: `openai/gpt-oss-120b`, фолбэк: `openai/gpt-oss-20b` (Groq).
- Локальные: `sentence-transformers` (эмбеддинги, 384-мерные), `faster-whisper`
  medium (транскрипция голосовых, CPU).
- `/ask` и `/summary` — LangGraph-графы; с заданными `LANGFUSE_PUBLIC_KEY`/
  `LANGFUSE_SECRET_KEY` каждый узел и LLM-вызов трассируется автоматически.

## Устранение проблем

**Бот не отвечает** — проверьте `BOT_TOKEN` в переменных окружения сервера
(не только локальный `.env`), что webhook принят Telegram (`RAILWAY_PUBLIC_DOMAIN`
корректен) и бот добавлен в чат.

**Ошибки при генерации** — проверьте `GROQ_API_KEY` и его лимиты.

**Проблемы с БД** — `DATABASE_URL` должен указывать на internal-адрес, если
бот и Postgres в одном Railway-проекте; для подключения снаружи используйте
публичный proxy-адрес, не internal.

## Лицензия

Проект создан в образовательных целях.
