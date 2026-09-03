from collections import defaultdict

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from chat_context import add_note, delete_auto_notes
from chat_moments import add_moment, delete_all_moments
from crypto_utils import decrypt
from database.db import get_conn
from display_names import resolve_display_name
from embeddings import embed
from llm.groq_client import answer_context_question
from llm.prompt import (
    prompt_for_cluster_description,
    prompt_for_moments_map,
    prompt_for_portrait_map,
    prompt_for_portrait_reduce,
)

PERSON_BATCH_SIZE = 200
MIN_MESSAGES_FOR_PORTRAIT = 10
MIN_CLUSTER_SIZE = 3
MAX_CLUSTER_SAMPLE = 15
MOMENTS_BATCH_SIZE = 500


def _parse_vector(value):
    # psycopg2 has no built-in adapter for pgvector's type, so it comes back
    # as the raw text representation, e.g. "[0.1,0.2,...]".
    if isinstance(value, str):
        return [float(x) for x in value.strip('[]').split(',')]
    return list(value)


def build_portrait(display_name, texts):
    """Map-reduce a person's texts (from any source) into one portrait note, or None."""
    if len(texts) < MIN_MESSAGES_FOR_PORTRAIT:
        return None

    observations = []
    for i in range(0, len(texts), PERSON_BATCH_SIZE):
        batch = texts[i:i + PERSON_BATCH_SIZE]
        obs = answer_context_question(prompt_for_portrait_map.format(messages="\n".join(batch)))
        if obs:
            observations.append(obs)

    if not observations:
        return None

    portrait = answer_context_question(
        prompt_for_portrait_reduce.format(name=display_name, observations="\n\n".join(observations))
    )
    return f"Портрет: {display_name} — {portrait.strip()}" if portrait else None


def cluster_and_describe(texts_with_vectors):
    """texts_with_vectors: list of (text, embedding_vector). Returns pattern note strings.

    Embeddings are only used in-memory for this clustering pass -- nothing
    here writes bulk data anywhere, only the resulting short descriptions do.
    """
    if len(texts_with_vectors) < MIN_CLUSTER_SIZE:
        return []

    texts = [t for t, _ in texts_with_vectors]
    vectors = normalize([v for _, v in texts_with_vectors])

    # HDBSCAN's distance computations degrade badly in high dimensions (our
    # embeddings are 384-d) -- on tens of thousands of points this can run
    # for hours. Reducing to ~50 dims via PCA keeps most of the semantic
    # structure while making the actual clustering tractable.
    if len(vectors) > 200:
        target_dim = min(50, len(vectors) - 1)
        vectors = PCA(n_components=target_dim, random_state=42).fit_transform(vectors)

    # L2-normalized vectors make euclidean distance equivalent to cosine
    # distance, and sklearn's HDBSCAN is numerically unstable with
    # metric='cosine' directly (produces overflow/divide-by-zero warnings).
    clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric='euclidean')
    labels = clusterer.fit_predict(vectors)

    clusters = defaultdict(list)
    for label, text in zip(labels, texts):
        if label != -1:
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


def generate_moments(chat_id, chronological_lines):
    """chronological_lines: list of "Автор: текст" strings, in time order.

    Runs a map-only pass (no reduce) extracting notable moods/opinions/jokes
    per chunk, embeds each one, and stores it directly into chat_moments --
    retrieved later via RAG, not always injected, so there's no need to
    compress this down to a small fixed set.
    """
    count = 0
    for i in range(0, len(chronological_lines), MOMENTS_BATCH_SIZE):
        batch = chronological_lines[i:i + MOMENTS_BATCH_SIZE]
        res = answer_context_question(prompt_for_moments_map.format(messages="\n".join(batch)))
        if not res or "НЕТ" in res.strip()[:10]:
            continue

        for line in res.split("\n"):
            line = line.strip().lstrip("-").strip()
            if line:
                add_moment(chat_id, line, embed(line))
                count += 1

    return count


def learn_context(chat_id):
    """Analyze whatever this bot has itself captured in the messages table."""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT user_name, username FROM messages WHERE user_id = %s AND is_bot = FALSE",
            (chat_id,),
        )
        people = cursor.fetchall()

    portraits = []
    for user_name, username in people:
        with get_conn() as conn:
            cursor = conn.cursor()
            if username:
                cursor.execute(
                    "SELECT message FROM messages WHERE user_id = %s AND username = %s AND is_bot = FALSE ORDER BY id",
                    (chat_id, username),
                )
            else:
                cursor.execute(
                    "SELECT message FROM messages WHERE user_id = %s AND username IS NULL AND user_name = %s AND is_bot = FALSE ORDER BY id",
                    (chat_id, user_name),
                )
            texts = [decrypt(row[0]) for row in cursor.fetchall()]

        display_name = resolve_display_name(username, user_name)
        portrait = build_portrait(display_name, texts)
        if portrait:
            portraits.append(portrait)

    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT message, embedding FROM messages WHERE user_id = %s AND embedding IS NOT NULL AND is_bot = FALSE ORDER BY id",
            (chat_id,),
        )
        rows = cursor.fetchall()

    patterns = cluster_and_describe([(decrypt(r[0]), _parse_vector(r[1])) for r in rows])

    delete_auto_notes(chat_id)
    for note in portraits + patterns:
        add_note(chat_id, note, source='auto')

    return portraits, patterns
