# Как из этого проекта сделать статью

Этот файл — не текст статьи, а рабочая инструкция автору. Основной черновик:
`research/PAPER_DRAFT_RU.md`. Зафиксированный SocialMemBench-прогон:
`.research_runs/frozen/socialmembench_temporal_episodes_official_harness_v1_20260912_122626_MSK/`.

## Что такое статья простыми словами

Статья — не полная документация проекта и не рассказ обо всех трёх днях
экспериментов. Это один проверяемый аргумент:

> В долгоживущем multi-party чате изменяющиеся факты полезно хранить как
> небольшие append-only temporal episodes с provenance к исходным сообщениям и
> извлекать совместно с raw evidence; такая конструкция должна особенно помогать
> вопросам об изменениях во времени, не требуя глобального knowledge graph.

Каждый раздел отвечает на один вопрос читателя:

1. **Abstract:** что болит, что предложено, как проверено, какой главный
   результат и главное ограничение. Пишется последним.
2. **Introduction:** почему обычного raw retrieval и статических summary
   недостаточно; какой научный вопрос; 3–4 конкретных вклада.
3. **Related Work:** что уже умеют conversational memory, graph memory,
   temporal RAG и attribution; где именно остаётся незакрытая щель.
4. **Method:** точный query-independent pipeline от turn до event, episode,
   retrieval и provenance expansion. Здесь не должно быть результатов.
5. **Experimental Setup:** datasets, split/protocol, baselines, модели,
   top-k, prompts, метрики, цена и latency. Должно быть достаточно деталей для
   воспроизведения.
6. **Results:** таблицы и доверительные интервалы без объяснений задним числом.
7. **Analysis:** где метод помог, где проиграл и почему; false split/false merge,
   extraction loss, temporal question types и citation behavior.
8. **Limitations and Ethics:** LLM extraction, англоязычные synthetic benchmarks,
   малая статистическая мощность SocialMemBench, приватность реальных чатов и
   отсутствие официальной leaderboard submission.
9. **Conclusion:** один абзац — что установлено фактически, а не что хотелось
   установить.

## Что уже готово

- Сформулирована проблема и границы исследования.
- Описана архитектура Temporal Episodes и её формальная модель.
- Зафиксирован benchmark protocol: Social — discovery/external, Group —
  confirmatory, EverMemBench — optional robustness.
- SocialMemBench завершён на 1 031 QA и 43 networks; сырьё, код, predictions,
  scores, logs, embeddings и report запечатаны с SHA256.
- Есть controlled ablations RAW, RAW+FLAT, RAW+VERSIONED и
  RAW+TEMPORAL EPISODES.
- Есть честный результат: общий paired delta против RAW `+0.010`, 95% CI
  `[-0.008, +0.028]`; это положительный, но не статистически значимый сигнал.

## Что пока нельзя утверждать

- Нельзя писать, что Temporal Episodes доказанно превосходят RAW в целом.
- Нельзя называть сравнение с опубликованными числами официальным leaderboard
  result.
- Нельзя выбирать только выигрышные типы вопросов как главный результат после
  просмотра SocialMemBench.
- Нельзя выдавать benchmark harness за production Telegram `/ask` pipeline.
- Нельзя менять метод или основной confirmatory protocol по результатам Group и
  затем называть этот же прогон confirmatory.

## В каком порядке писать

Не начинать с Abstract. Практический порядок такой:

1. Заморозить Method и Experimental Setup по фактически запущенному коду.
2. Вставить SocialMemBench Results без рекламных формулировок.
3. Закончить заранее заявленный GroupMemBench confirmatory run.
4. Один раз сформулировать итоговый claim по Group; при отрицательном результате
   написать качественный failure analysis, не переделывать цель постфактум.
5. Решить, оправдан ли EverMemBench как robustness check.
6. Написать Related Work с точными ссылками.
7. Написать Introduction и contributions под реально полученные результаты.
8. Последними написать Conclusion, Abstract и название.

## Минимальный publishable пакет

- Статья в LaTeX по шаблону выбранной конференции или arXiv.
- Отдельный чистый repository с implementation, configs и командами запуска.
- Frozen manifests/checksums для каждого заявленного численного результата.
- Таблица качества с network-level confidence intervals и paired deltas.
- Таблица ресурсов: число LLM/embedding calls, tokens, wall-clock time, API cost,
  размер памяти и retrieval latency.
- Приложение с prompts, schemas, validation rules и несколькими полными
  provenance-backed examples.
- Data/privacy statement: production-чат используется только как мотивация или
  отдельный приватный pilot, если его нельзя безопасно выпустить.

## Критерий готовности первого препринта

Первый честный препринт готов, когда завершён GroupMemBench и для каждого числа
в основной таблице существует неизменяемый артефакт, а Methods описывает ровно
тот код, который это число произвёл. EverMemBench, production deployment и
идеальная генерация памяти могут остаться future work.
