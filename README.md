# summary_sov

Telegram-бот для группового чата: саммаризация истории, ответы на вопросы
по контексту переписки (RAG), накопление знаний о людях и группе, и
исследовательский трек по provenance-bearing памяти для LLM-агентов.

Построен на LangGraph (оркестрация `/ask` и `/summary`), Postgres + pgvector
(гибридный full-text + семантический поиск), Groq (`gpt-oss-120b`/`20b`),
с шифрованием сообщений at rest и опциональной трассировкой через Langfuse.

## Коротко для технического интервью

- **Agentic/RAG:** LangGraph workflow, hybrid retrieval (PostgreSQL FTS +
  pgvector), RRF, LLM-reranking, восстановление контекста и цитирование
  исходных сообщений.
- **Memory:** событийная память с адресуемыми источниками (`episodic_memory.py`):
  событие хранит дословную цитату и id исходного сообщения, цитата проверяется
  механически, обновления складываются в append-only эпизоды, найденный эпизод
  разворачивается обратно в первичные сообщения.
- **Quality:** раздельная оценка retrieval, linking/provenance и answer
  quality; human-verified gold, замороженные артефакты и paired bootstrap.
- **Reliability:** Langfuse tracing, тесты, анализ реальных failure modes,
  идемпотентная обработка webhook и шифрование текста сообщений at rest.
- **Проверяемые результаты:** 1 031 вопрос, 43 независимые сети, четыре
  одинаковых условия и 4 124 сохранённых результата. Подробности и ограничения
  — в [PORTFOLIO_EVIDENCE.md](PORTFOLIO_EVIDENCE.md).

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
инициализирует/самопочиняет схему БД и запускает планировщик авто-саммари вместе
с фоновым воркером эпизодической памяти.

Дополнительные переменные окружения (все со значениями по умолчанию):

```env
MEMORY_WORKER_ENABLED=true        # фоновое построение памяти
MEMORY_EPISODES_IN_ASK=false      # эпизоды в выдаче /ask (см. протокол ниже)
ONLINE_JUDGE_SAMPLE_RATE=0.25     # доля ответов, которые оценивает судья онлайн
JUDGE_MODEL_KIND=primary          # модель судей
```

## Память чата

Память строится **до** вопросов, отдельно от `/summary`: воркер берёт осевшие
сообщения, режет их на сессии (`conversation_id`, иначе календарный день) и на
каждый блок делает один вызов извлечения. Событие принимается, только если у
него есть цитата, дословно найденная в указанном сообщении; время берётся из
реальных дат сообщений, а не из ответа модели. Дальше событие либо продолжает
эпизод (revise / reaffirm / augment), либо открывает новый — при любой
неуверенности создаётся новый, потому что ложное слияние дороже пропущенной
связи.

В `/ask` эпизоды конкурируют с сырыми сообщениями в одном векторном пуле и, если
их выбрал rerank, разворачиваются в исходные сообщения своих последних событий.
Слой выключен флагом `MEMORY_EPISODES_IN_ASK`, пока парное сравнение с текущим
поиском не пройдёт правило из `docs/EPISODES_ROLLOUT_PROTOCOL.md`:

```bash
python3 tests/evals/compare_episodes.py            # RAW vs RAW+EPISODES, парно
python3 -m episodic_memory <chat_id>               # разовое построение памяти
```

Метод описан в препринте `paper/preprint_memory_ru/`; там же граница того, что
установлено: слой событий улучшает доставку размеченных источников, а средний
прирост accuracy и добавка от связывания в эпизоды — нет. Прод использует другие
модели (Groq, локальный MiniLM), поэтому результат статьи сюда автоматически не
переносится — для этого и нужен парный прогон.

## Структура проекта

```
summary_sov/
├── main.py                    # точка входа, webhook-сервер, init
├── config.py                  # переменные окружения
├── scheduler.py                # авто-саммари 14:00 / 22:00
├── handlers/handlers.py        # все message/command-обработчики
├── llm/
│   ├── graphs.py                # LangGraph-графы /ask и /summary
│   ├── answer_judges.py         # LLM-судьи качества ответа (оффлайн и онлайн)
│   ├── groq_client.py           # ChatGroq-клиент, Langfuse tracing
│   └── prompt.py                # шаблоны промптов
├── database/
│   ├── db.py                    # пул соединений
│   └── init_db.py               # самопочинающаяся схема
├── episodic_memory.py          # событийная память с адресуемыми источниками + эпизоды
├── ask_metrics.py              # механические метрики ответа, скоры в Langfuse
├── memory_facts.py             # прежняя память о состоянии (read-only, не пишется)
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
    ├── EPISODES_ROLLOUT_PROTOCOL.md  # протокол выкатки эпизодов в /ask
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
