from collections import defaultdict

from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize

from chat_context import add_note, delete_auto_notes
from database.db import get_conn
from display_names import resolve_display_name
from llm.groq_client import answer_context_question
from llm.prompt import (
    prompt_for_cluster_description,
    prompt_for_portrait_map,
    prompt_for_portrait_reduce,
)

PERSON_BATCH_SIZE = 200
MIN_MESSAGES_FOR_PORTRAIT = 10
MIN_CLUSTER_SIZE = 3
MAX_CLUSTER_SAMPLE = 15


def _parse_vector(value):
    # psycopg2 has no built-in adapter for pgvector's type, so it comes back
    # as the raw text representation, e.g. "[0.1,0.2,...]".
    if isinstance(value, str):
        return [float(x) for x in value.strip('[]').split(',')]
    return list(value)


def _generate_person_portraits(chat_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT user_name, username FROM messages WHERE user_id = %s",
            (chat_id,),
        )
        people = cursor.fetchall()

    portraits = []
    for user_name, username in people:
        with get_conn() as conn:
            cursor = conn.cursor()
            if username:
                cursor.execute(
                    "SELECT message FROM messages WHERE user_id = %s AND username = %s ORDER BY id",
                    (chat_id, username),
                )
            else:
                cursor.execute(
                    "SELECT message FROM messages WHERE user_id = %s AND username IS NULL AND user_name = %s ORDER BY id",
                    (chat_id, user_name),
                )
            texts = [row[0] for row in cursor.fetchall()]

        if len(texts) < MIN_MESSAGES_FOR_PORTRAIT:
            continue

        observations = []
        for i in range(0, len(texts), PERSON_BATCH_SIZE):
            batch = texts[i:i + PERSON_BATCH_SIZE]
            obs = answer_context_question(prompt_for_portrait_map.format(messages="\n".join(batch)))
            if obs:
                observations.append(obs)

        if not observations:
            continue

        display_name = resolve_display_name(username, user_name)
        portrait = answer_context_question(
            prompt_for_portrait_reduce.format(name=display_name, observations="\n\n".join(observations))
        )
        if portrait:
            portraits.append(f"Портрет: {display_name} — {portrait.strip()}")

    return portraits


def _generate_cluster_patterns(chat_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT message, embedding FROM messages WHERE user_id = %s AND embedding IS NOT NULL ORDER BY id",
            (chat_id,),
        )
        rows = cursor.fetchall()

    if len(rows) < MIN_CLUSTER_SIZE:
        return []

    texts = [r[0] for r in rows]
    vectors = normalize([_parse_vector(r[1]) for r in rows])

    # L2-normalized vectors make euclidean distance equivalent to cosine
    # distance, and sklearn's HDBSCAN is numerically unstable with
    # metric='cosine' directly (produces overflow/divide-by-zero warnings).
    clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric='euclidean')
    labels = clusterer.fit_predict(vectors)

    clusters = defaultdict(list)
    for label, text in zip(labels, texts):
        if label == -1:
            continue
        clusters[label].append(text)

    patterns = []
    for members in clusters.values():
        if len(members) <= MAX_CLUSTER_SAMPLE:
            sample = members
        else:
            step = len(members) // MAX_CLUSTER_SAMPLE
            sample = members[::step][:MAX_CLUSTER_SAMPLE]

        desc = answer_context_question(
            prompt_for_cluster_description.format(messages="\n".join(f"- {m}" for m in sample))
        )
        if desc and "НЕЗНАЧИМО" not in desc:
            patterns.append(desc.strip())

    return patterns


def learn_context(chat_id):
    portraits = _generate_person_portraits(chat_id)
    patterns = _generate_cluster_patterns(chat_id)

    delete_auto_notes(chat_id)
    for note in portraits + patterns:
        add_note(chat_id, note, source='auto')

    return portraits, patterns
