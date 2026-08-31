from database.db import get_conn


def init_db():
    with get_conn() as conn:
        cursor = conn.cursor()
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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_state (
                chat_id BIGINT PRIMARY KEY,
                last_summary_msg_id INTEGER NOT NULL DEFAULT 0
            );""")

        conn.commit()
