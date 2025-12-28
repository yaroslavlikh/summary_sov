import sqlite3

def init_db():
    conn = sqlite3.connect('tasks.sql')
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS tasks(
        id INT PRIMARY KEY AUTOINCREMENT,
        user_id INT NOT NULL,
        message TEXT NOT NULL);""")
    conn.commit()
    conn.close()