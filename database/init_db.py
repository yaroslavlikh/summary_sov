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
    
    try:
        cursor.execute("ALTER TABLE messages ADD COLUMN replied_message TEXT DEFAULT NULL;")
    except sqlite3.OperationalError:
        pass
    
    conn.commit()
    conn.close()