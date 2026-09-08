# Prospective memory retrieval для долгоживущего группового чата

## Статус документа

Это проект исследовательской гипотезы и следующей реализации для `summary_sov`.
Он не описывает уже доказанный результат и не заменяет
[`research/REPORT.md`](REPORT.md): текущий semantic-disentanglement pilot дал
слабый и смешанный эффект и пока не готов к продакшену.

Предлагаемая система не использует memory graph или PageRank. Основные
примитивы — сырые сообщения, динамически обновляемые атомарные факты, несколько
простых retrieval-веток и один общий reranker.

---

## 1. Исследовательская проблема

Долгоживущий групповой чат содержит несколько качественно разных источников
памяти:

1. Сырые реплики с точным автором, временем и `message_id`.
2. Текущие состояния людей: где человек живёт, чем занимается, какие у него
   планы и ограничения.
3. Эпизоды: отдельные прошлые обсуждения, события и конфликты.
4. Производные социальные знания: портреты, групповые нормы, повторяющиеся
   шутки и паттерны.

Существующий `/ask` хорошо ищет прямые лексические или семантические совпадения
в сырых сообщениях, но плохо восстанавливает скрытые предпосылки реплики.

Пример:

```text
14 августа:
Миша: Я окончательно переехал из Москвы в Казань.

7 сентября:
Миша: Ну я, конечно, пойду на бильярд.
```

Чтобы распознать шутку, системе недостаточно найти старые сообщения со словом
«бильярд». Она должна восстановить актуальное состояние Миши — он находится в
другом городе, — хотя поверхностно эти две реплики семантически не похожи.

Отсюда два связанных требования:

- поиск должен поднимать не только тематически похожий текст, но и скрытые
  факты, способные изменить интерпретацию текущей реплики;
- любой производный факт должен раскрываться назад в исходные сообщения, чтобы
  ответ оставался проверяемым и цитируемым.

---

## 2. Что есть сейчас

### 2.1 Сырые сообщения

Обычный путь `/ask` использует:

```text
classify/rewrite
    -> FTS по messages
    -> vector search по messages
    -> RRF
    -> LLM rerank
    -> generate answer
```

Это хороший baseline для прямого поиска и его следует сохранить.

### 2.2 Anchor-контекст

При явном reply `_resolve_anchor` находит строку сообщения, после чего текущий
граф сразу переходит к `generate_answer`. Окно строится по равенству
`conversation_id` или через fallback `±3`.

Текущий `conversation_id` вычисляется по reply-наследованию и временному разрыву
в пять минут. Реальные инциденты и полный скан показали, что в активном чате
разные разговоры могут непрерывно склеиваться на несколько суток.

Semantic-disentanglement pilot уменьшает часть contamination, но создаёт
собственные ошибки:

- объединяет формульные реплики разных людей;
- плохо присоединяет короткие сообщения;
- иногда оставляет слишком узкий контекст;
- на исправленном end-to-end eval улучшение слабое и статистически неубедительное.

### 2.3 Производная память

Сейчас есть два разных пути:

- `chat_context`: портреты и lore всегда добавляются в prompt целиком;
- `chat_moments`: заметки ищутся только через vector similarity к тексту
  вопроса.

Оба пути теряют исходные `message_id`. Поэтому модель может использовать
производный факт, но пользовательская ссылка назад к доказательству невозможна.

Кроме того, портреты ориентированы на повторяющиеся черты человека, а
`chat_moments` — на яркие события. Между ними нет отдельного представления
актуального изменяемого состояния человека.

---

## 3. Главные архитектурные решения

### 3.1 Не считать `conversation_id` семантической истиной

Постоянное непересекающееся разбиение группового чата в общем случае
неоднозначно:

- несколько разговоров идут одновременно;
- одна реплика может иметь несколько функций;
- старая тема возвращается через недели;
- нужная граница зависит от текущего вопроса.

Поэтому `conversation_id` можно оставить как технический сигнал или заменить
на нейтральный `episode_id`, но нельзя считать его окончательным ответом на
вопрос «какой контекст релевантен».

### 3.2 Conversation embedding — retriever, а не classifier

Embedding эпизода полезен, чтобы найти похожие прошлые обсуждения. Он не должен:

- навсегда объединять два похожих эпизода;
- единолично назначать сообщения разговору;
- заменять reply-связи, участников и временную последовательность.

Короткая формула:

> Retrieve by episode embedding; verify and cite at message level.

### 3.3 Область контекста строится во время запроса

Для конкретного вопроса система должна сформировать минимальный достаточный
`evidence bundle`, а не слепо передавать целый кластер:

```text
EvidenceBundle
    serving_text       текст raw-сообщения или производного факта
    kind               raw | state | moment | episode | lore
    raw_evidence_ids   исходные message_id
    role               функция в интерпретации вопроса
```

Один и тот же raw message может участвовать в разных bundles. Это нормальнее,
чем принудительная глобальная кластеризация.

### 3.4 Скрытые предпосылки готовятся при записи памяти

Нельзя полагаться на то, что query-time модель всегда превратит реплику про
бильярд в скрытый запрос «где сейчас Миша». Поэтому вместе с фактом сохраняются
возможные будущие ситуации его использования — `retrieval_cues`.

Это prospective indexing: индексируется не только содержание прошлого факта,
но и классы будущих реплик, которые он поможет интерпретировать.

---

## 4. Динамическая fact memory

### 4.1 Минимальная схема

Для пилота нужна отдельная таблица, не ломающая существующие `chat_context` и
`chat_moments`:

```sql
CREATE TABLE memory_facts (
    id BIGSERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    subject_key TEXT,
    state_key TEXT,
    kind TEXT NOT NULL,
    claim TEXT NOT NULL,
    retrieval_text TEXT NOT NULL,
    embedding vector(384),
    importance SMALLINT NOT NULL DEFAULT 1,
    observed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    superseded_by BIGINT REFERENCES memory_facts(id),
    source_message_ids BIGINT[] NOT NULL
);
```

`claim` и `retrieval_text` должны шифроваться так же, как остальные сохранённые
тексты. Embedding вычисляется по plaintext до шифрования.

Начальный набор `kind`/`state_key` должен оставаться небольшим:

```text
location
work_study
availability
relationship
preference
plan
identity
group_lore
other
```

Сложную онтологию вводить не нужно. `state_key` нужен прежде всего для
обновления очевидно изменяемых состояний, например `current_location`.

### 4.2 Пример записи

```json
{
  "subject_key": "misha",
  "state_key": "current_location",
  "kind": "location",
  "claim": "Миша переехал из Москвы в Казань",
  "retrieval_cues": [
    "где сейчас Миша",
    "может ли Миша прийти на встречу в Москве",
    "почему Миша не сможет присутствовать",
    "участие Миши в местных мероприятиях",
    "шутки Миши о том, что он куда-то придёт"
  ],
  "importance": 3,
  "source_message_ids": [68121, 68125]
}
```

`retrieval_text` — конкатенация `claim` и `retrieval_cues`. Vector search идёт
по нему, а не только по короткому тексту факта.

### 4.3 Обновления и противоречия

Факты хранятся как события, а не переписывают историю.

Если появляется «я вернулся в Москву»:

```text
старый current_location=Казань -> active=false
новый  current_location=Москва -> active=true
старый.superseded_by = новый.id
```

Временные планы могут иметь `expires_at`. Если точную дату определить нельзя,
факт остаётся активным до явного обновления, а генератор получает дату
наблюдения и обязан учитывать более свежие свидетельства.

### 4.4 Извлечение через существующий summary flow

После генерации `/summary` уже запускается отдельный tool-calling extraction
проход. Его следует расширить инструментом:

```text
upsert_state(
    subject,
    state_key,
    kind,
    claim,
    retrieval_cues,
    importance,
    expires_at,
    source_message_ids
)
```

Изменения текущего контракта:

1. Extraction-модель должна видеть реальные `message_id`, а не только локальные
   номера строк.
2. Каждый tool call обязан передавать хотя бы один `source_message_id`.
3. Сервер валидирует, что указанные сообщения существуют в том же `chat_id` и
   находились во входном summary batch.
4. Инструмент сам закрывает предыдущий активный факт с тем же
   `(chat_id, subject_key, state_key)` и добавляет новую версию.
5. События, шутки и мнения без состояния продолжают записываться как
   `moment`/`group_lore`, но тоже получают provenance.

Summary для пользователя и extraction памяти остаются двумя разными
результатами одного batch: красивое summary не должно одновременно служить
структурированным storage format.

---

## 5. Retrieval pipeline

### 5.1 Seed entities

Система определяет небольшой набор вовлечённых людей:

- автор reply-anchor;
- человек, задающий вопрос;
- явно названные участники;
- участники, восстановленные из местоимений текущим rewrite-вызовом.

Для anchor-пути автор доступен детерминированно из БД. Для вопроса без anchor
существующий `_classify_and_rewrite` может дополнительно возвращать
`memory_subjects`; отдельный LLM-вызов для этого не требуется.

### 5.2 Параллельные ветки поиска

```text
1. raw_fts
   Прямые лексические совпадения в messages.

2. raw_vector
   Семантические совпадения в messages.

3. entity_state
   Активные факты вовлечённых людей без cosine threshold.

4. memory_vector
   Поиск по embedding(claim + retrieval_cues) среди всех memory_facts.

5. moment_or_episode
   Похожие прошлые моменты и разговорные эпизоды.
```

Начальные бюджеты кандидатов:

```text
raw FTS                 8
raw vector              8
active entity facts    12 суммарно
memory vector           8
moments/episodes        5
```

После дедупликации reranker должен видеть не больше 20–30 объектов.

### 5.3 Entity-state retrieval

Эта ветка отвечает за высокий recall скрытых предпосылок:

```sql
SELECT *
FROM memory_facts
WHERE chat_id = :chat_id
  AND subject_key = ANY(:subjects)
  AND active = TRUE
  AND (expires_at IS NULL OR expires_at > now())
ORDER BY importance DESC, observed_at DESC
LIMIT 12;
```

Она принципиально не требует совпадения слов вопроса с фактом. Поэтому переезд
Миши попадёт в candidates даже при вопросе про бильярд.

### 5.4 Anchor-путь

Сейчас найденный anchor ведёт прямо в `generate_answer`. Новый путь должен быть:

```text
resolve_anchor
    -> retrieve_entity_state
    -> retrieve_memory_vector
    -> collect_raw_context_candidates
    -> rerank_evidence
    -> expand_provenance
    -> generate_answer
```

Сам anchor всегда остаётся обязательным кандидатом. `conversation_id` может
расширять candidate pool, но больше не должен целиком определять финальное окно.

### 5.5 Episode embeddings

Их следует добавлять после fact-memory pilot. Назначение:

- поднять похожие прошлые обсуждения;
- найти повторяющиеся групповые ситуации;
- восстановить старый эпизод с другой формулировкой.

Episode embedding не решает скрытые entity-state зависимости и потому не должен
быть первой реализацией.

---

## 6. Reranking как центральный этап

### 6.1 Почему текущего критерия недостаточно

Формулировка «выбери сообщения, которые помогают ответить» склоняет модель к
тематическому совпадению. Для косвенной реплики нужен другой критерий:

> Выбери факты и сообщения, без которых смысл, истинность, серьёзность,
> адресат или временная актуальность текущей реплики могут быть поняты иначе.

Практический removal test:

> Если убрать кандидата, может ли разумный читатель иначе понять ответ или тон
> сообщения?

### 6.2 Функциональные роли кандидатов

Reranker возвращает JSON, а не голый список чисел:

```json
{
  "selected": [
    {
      "type": "raw",
      "id": 73012,
      "role": "utterance_being_interpreted"
    },
    {
      "type": "memory",
      "id": 41,
      "role": "contradicting_current_state"
    }
  ],
  "interpretation": "likely_joke"
}
```

Начальный список ролей:

```text
direct_evidence
referent_resolution
current_state
contradiction
temporal_update
shared_joke
speaker_trait
utterance_being_interpreted
irrelevant
```

Роли нужны не как сложная онтология, а как наблюдаемое объяснение решения
reranker. Они позволяют отдельно диагностировать ложное тематическое совпадение
и полезную скрытую предпосылку.

### 6.3 Формат входа

```text
Текущая реплика:
[A] Миша: Я пойду на бильярд, конечно.

Кандидаты:
[R1 raw] Гордей: Бильярд в пятницу в 20:00.
[M1 current_state] Миша переехал в Казань.
    Наблюдалось: 14 августа.
    Источники: 68121, 68125.
[M2 preference] Миша любит русский бильярд.
[M3 speaker_trait] Миша часто иронизирует.
```

Желаемый выбор:

```json
{
  "selected": [
    {"id": "R1", "role": "direct_evidence"},
    {"id": "M1", "role": "contradiction"},
    {"id": "M3", "role": "speaker_trait"}
  ],
  "interpretation": "likely_joke"
}
```

### 6.4 Provenance expansion

После rerank каждый выбранный derived item раскрывается в его
`source_message_ids`. Генератор получает и производную формулировку, и исходные
реплики.

User-facing цитаты разрешены только на raw messages. Производный факт без
валидного provenance не может быть единственным основанием фактического ответа.

---

## 7. End-to-end пример

### Запись памяти

```text
[68121] Миша: Я окончательно переехал в Казань.
[68125] Миша: В Москве теперь только иногда бываю.
```

Summary extraction вызывает `upsert_state` и создаёт current-location fact с
retrieval cues и двумя source IDs.

### Запрос через несколько недель

```text
[73012] Миша: Ну я, конечно, пойду на бильярд.
[73013] Ярослав -> 73012: Он серьёзно?
```

Retrieval:

1. Anchor даёт автора `Миша`.
2. Raw search находит обсуждение бильярда.
3. Entity-state retrieval гарантированно поднимает активный `current_location`.
4. Prospective embedding дополнительно совпадает с cue про участие в локальной
   встрече.
5. Reranker помечает переезд как `contradiction/current_state`.
6. Provenance expansion загружает `68121` и `68125`.

Ответ:

```text
Скорее всего, Миша шутит: он говорил, что переехал в Казань и бывает в Москве
только иногда [1][2].
```

---

## 8. Оценка

### 8.1 Декомпозиция отказов

Для каждого eval-кейса сохраняются стадии:

```text
нужный факт не попал в candidates
    -> retrieval failure

факт попал, но reranker его отбросил
    -> ranking failure

факт выбран, но ответ неверный
    -> reasoning failure

ответ верный, но sources/citations неверны
    -> provenance failure
```

Это важнее одной итоговой оценки ответа: система становится диагностируемой.

### 8.2 Retrieval ablation

Основной эксперимент — факторный:

| Вариант | Entity retrieval | Retrieval cues |
|---|---:|---:|
| Claim embedding | нет | нет |
| Entity only | да | нет |
| Prospective indexing | нет | да |
| Полная система | да | да |

Отдельно сравниваются:

- текущий positional/`conversation_id` baseline;
- raw FTS + vector;
- always-injected `chat_context`;
- fact memory без provenance expansion;
- полная fact memory с provenance.

### 8.3 Типы вопросов

1. **Direct:** «Где сейчас Миша?»
2. **Paraphrased:** «В каком городе теперь живёт Миша?»
3. **Implicit constraint:** «Сможет ли Миша прийти на московскую встречу?»
4. **Pragmatic/sarcastic:** «Миша говорит, что точно придёт на московский
   бильярд. Это всерьёз?»
5. **Temporal update:** вопрос, где старый и новый факты конфликтуют.
6. **Mixed evidence:** производный факт плюс новая raw-реплика.

### 8.4 Метрики

Retrieval:

- raw evidence Recall@K;
- derived fact Recall@K;
- candidate contamination;
- candidate count/token count;
- recall отдельно для direct и implicit запросов.

Reranking:

- evidence selection precision/recall;
- recall нужной функциональной роли;
- доля случаев, где reranker сохраняет provenance-bearing item.

Generation:

- answer correctness;
- citation precision;
- citation completeness;
- provenance-chain validity;
- correctness non-inferiority относительно production baseline.

Raw, derived и source retrieval следует отчитывать отдельно: автоматическое
зачисление derived item как эквивалента исходному свидетельству может скрыть
ошибки поддержки.

### 8.5 Как получить достаточно кейсов

Нельзя опираться только на редкие настоящие `/ask` с длинными отсылками.

Нужны три набора:

1. **Reply reconstruction:** скрыть reply edge у обычных человеческих реплик и
   проверить восстановление parent/цепочки.
2. **Natural anchor selection:** использовать реальные параллельные reply-chain
   как положительные и естественные отрицательные сообщения.
3. **Counterfactual presupposition probes:** брать реальные provenance-backed
   facts и создавать поверх них будущие прямые и имплицитные запросы. Генерацию
   можно масштабировать LLM, но тестовая часть требует человеческой проверки
   вопроса, ответа и минимального evidence set.

Реальные `/ask` остаются небольшим end-to-end safety-набором, а не единственным
источником статистической мощности.

### 8.6 Online shadow evaluation

После offline-проверки candidate работает на каждом `/ask` в shadow-режиме:

- production отправляет пользователю настоящий ответ;
- candidate сохраняет candidates, rerank и черновой ответ, но не отправляет их;
- для ручной оценки откладываются прежде всего случаи, где системы выбрали
  разные evidence или citations;
- оценка проводится вслепую и попарно.

Так ручная разметка концентрируется на информативных disagreement cases, а не
на сотнях одинаковых ответов.

---

## 9. Порядок реализации

### Этап 0. Контракт данных и offline harness

- добавить `memory_facts`;
- определить JSON-контракт `upsert_state`;
- валидировать provenance против текущего summary batch;
- написать deterministic retrieval trace;
- production behavior пока не менять.

### Этап 1. Извлечение памяти

- расширить context extraction после `/summary`;
- извлекать state facts, retrieval cues и source IDs;
- реализовать простое superseding по `subject_key + state_key`;
- прогнать на истории в read-only/sidecar режиме и вручную проверить выборку.

### Этап 2. Retrieval без episode embeddings

- entity-state retrieval;
- vector retrieval по prospective `retrieval_text`;
- объединение с текущим raw FTS/vector;
- логирование origin каждого кандидата.

### Этап 3. Evidence reranker

- JSON-выход с функциональными ролями;
- минимальный evidence bundle;
- provenance expansion;
- генератор цитирует только raw messages.

### Этап 4. Эксперимент

- прямые и имплицитные probes;
- факторная абляция entity retrieval × retrieval cues;
- stage-level failure analysis;
- небольшой end-to-end `/ask` eval.

### Этап 5. Episode retrieval

Добавлять только если fact-memory не покрывает реальные провалы:

- episode representation;
- semantic search прошлых эпизодов;
- message-level rerank после раскрытия эпизода;
- никаких автоматических глобальных merge по cosine similarity.

### Этап 6. Shadow deployment

- candidate не влияет на ответы;
- собираются disagreement cases;
- после достаточной проверки — ограниченный A/B или постепенное включение.

---

## 10. Что сознательно не делаем в первой версии

- Не строим memory graph.
- Не внедряем PageRank.
- Не пытаемся идеально кластеризовать всю историю.
- Не заменяем `conversation_id` новым hard semantic ID.
- Не создаём большую онтологию отношений.
- Не добавляем отдельный LLM-вызов только для entity extraction, пока это можно
  вернуть из существующего classify/rewrite call.
- Не переносим старые непроверяемые портреты в provenance-aware memory как
  будто они уже имеют доказательства.
- Не отправляем candidate-ответы пользователям до offline и shadow проверки.

---

## 11. Формулировка исследовательской гипотезы

### Рабочая формулировка

> В долгоживущем многопользовательском чате семантическое сходство с текущим
> вопросом недостаточно для извлечения памяти, необходимой для интерпретации
> косвенных и прагматических отсылок. Subject-gated retrieval и prospective
> indexing производных фактов повышают recall скрытых предпосылок, а раскрытие
> выбранных фактов в исходные сообщения улучшает проверяемость цитат без
> ухудшения корректности ответа.

### Возможное название подхода

**Prospective, Provenance-Resolving Memory Retrieval**

или более предметно:

**Retrieving Hidden Conversational Preconditions from Long-Lived Multi-Party
Chat Memory**

### Предполагаемые вклады статьи

1. Формализация скрытой предпосылки как отдельного retrieval target в
   multi-party conversational memory.
2. Prospective indexing: генерация retrieval cues в момент формирования
   производной памяти.
3. Subject-gated retrieval изменяемых entity states без необходимости
   лексического совпадения с вопросом.
4. Единый evidence reranking raw и derived memories с раскрытием provenance до
   исходных реплик.
5. Реальные production-инциденты и benchmark с direct, implicit, temporal и
   mixed-evidence вопросами.

Новизну пунктов 1–3 перед сильным заявлением нужно дополнительно проверить
отдельным литературным поиском по document expansion, anticipatory retrieval,
pragmatic inference и event/state memory. Уже найденные близкие работы по
provenance graph и multi-party benchmarks не делают эту реализацию автоматически
новой, но и не совпадают с ней целиком.

---

## 12. Критерий успеха первой версии

Первая версия считается успешной, если на проверенной выборке implicit-запросов:

1. Нужный state fact попадает в candidates заметно чаще, чем при raw/vector и
   claim-only memory search.
2. Reranker сохраняет его без резкого роста contamination.
3. Ответ использует правильную временную версию факта.
4. Каждая фактическая ссылка раскрывается в реальные human-authored messages.
5. На обычных прямых `/ask` correctness не становится хуже production baseline.

Если entity retrieval + retrieval cues не улучшают именно первый пункт, не нужно
строить поверх них episode embeddings или более сложную память: сначала следует
пересмотреть базовую гипотезу поиска скрытых предпосылок.
