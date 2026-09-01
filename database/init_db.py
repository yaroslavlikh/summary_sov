from database.db import get_conn


def init_db():
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                user_name TEXT NOT NULL,
                message TEXT NOT NULL,
                last_id INTEGER DEFAULT 0,
                replied_message TEXT DEFAULT NULL,
                message_id BIGINT
            );""")

        # CREATE TABLE IF NOT EXISTS above won't add columns to a table that
        # already exists with an older schema, so heal it explicitly here.
        cursor.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS replied_message TEXT DEFAULT NULL;")
        cursor.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS message_id BIGINT;")
        cursor.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS username TEXT;")
        cursor.execute("ALTER TABLE messages ALTER COLUMN user_id TYPE BIGINT;")

        # A pre-existing table can also have an `id` column with no
        # auto-increment default (e.g. created as plain INTEGER PRIMARY KEY),
        # which makes every insert fail with a not-null violation.
        cursor.execute("""
            DO $$
            DECLARE
                tbl_oid oid := 'messages'::regclass;
                has_default boolean;
            BEGIN
                SELECT EXISTS (
                    SELECT 1 FROM pg_attrdef ad
                    JOIN pg_attribute a ON a.attrelid = ad.adrelid AND a.attnum = ad.adnum
                    WHERE ad.adrelid = tbl_oid AND a.attname = 'id'
                ) INTO has_default;

                IF NOT has_default THEN
                    CREATE SEQUENCE IF NOT EXISTS messages_id_seq OWNED BY messages.id;
                    PERFORM setval('messages_id_seq', COALESCE((SELECT MAX(id) FROM messages), 0) + 1, false);
                    ALTER TABLE messages ALTER COLUMN id SET DEFAULT nextval('messages_id_seq');
                END IF;
            END $$;
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_state (
                chat_id BIGINT PRIMARY KEY,
                last_summary_msg_id INTEGER NOT NULL DEFAULT 0
            );""")
        cursor.execute("ALTER TABLE chat_state ADD COLUMN IF NOT EXISTS last_summary_text TEXT DEFAULT NULL;")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mention_groups (
                id SERIAL PRIMARY KEY,
                chat_id BIGINT NOT NULL,
                group_name TEXT NOT NULL,
                username TEXT NOT NULL,
                UNIQUE (chat_id, group_name, username)
            );""")

        # Full-text search over message content, for /ask.
        cursor.execute("""
            ALTER TABLE messages ADD COLUMN IF NOT EXISTS search_vector tsvector
                GENERATED ALWAYS AS (to_tsvector('russian', coalesce(message, ''))) STORED;
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS messages_search_idx ON messages USING GIN (search_vector);
        """)

        # Semantic search (embeddings), for paraphrased /ask questions that
        # full-text search can't match on shared vocabulary alone.
        cursor.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS embedding vector(384);")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS messages_embedding_idx ON messages
                USING hnsw (embedding vector_cosine_ops);
        """)

        conn.commit()
