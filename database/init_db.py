import sqlite3

def init_db():
    conn = sqlite3.connect('database/messages.sql')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            user_name TEXT NOT NULL,
            message TEXT NOT NULL,
            last_id INTEGER DEFAULT 0
        );""")
    conn.commit()
    conn.close()